"""Drug index generation.

Combines molecule data with indications, mechanisms of action, chemical probes,
and warnings to produce the final drug index. Filters to include only molecules
that qualify as "drugs" and generates human-readable descriptions.
"""

from typing import Any

import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType

from pts.pyspark.common.session import Session


def drug_molecule(
    source: dict[str, str],
    destination: str,
    _settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Generate the drug molecule index.

    Args:
        source: Dictionary with paths to:
            - molecule: Processed molecule parquet
            - chemical_probes: Chemical probes parquet
            - mechanism_of_action: Mechanism of action parquet
            - drug_warning: Drug warnings parquet
            - indication: ChEMBL indication JSONL
            - disease: Disease/EFO parquet
        destination: Path to write the output parquet file.
        _settings: Custom settings (not used).
        properties: Spark configuration options.
    """
    spark = Session(app_name='drug_molecule', properties=properties)

    logger.info(f'Loading data from {source}')
    molecule_df = spark.load_data(source['molecule'])
    chemical_probes_df = spark.load_data(source['chemical_probes'])
    mechanism_df = spark.load_data(source['mechanism_of_action'])
    warning_df = spark.load_data(source['drug_warning'])
    indication_raw_df = spark.load_data(source['indication'], format='json')
    disease_df = spark.load_data(source['disease'])

    logger.info('Processing drug index')
    output_df = process_drug_index(
        molecule_df,
        chemical_probes_df,
        mechanism_df,
        warning_df,
        indication_raw_df,
        disease_df,
    )

    logger.info(f'Writing drug index to {destination}')
    output_df.write.parquet(destination, mode='overwrite')


def process_drug_index(
    molecule: DataFrame,
    chemical_probes: DataFrame,
    mechanism_of_action: DataFrame,
    drug_warning: DataFrame,
    indication_raw: DataFrame,
    disease: DataFrame,
) -> DataFrame:
    """Process and combine all drug data into the final index.

    Args:
        molecule: Processed molecule data.
        chemical_probes: Chemical probes data.
        mechanism_of_action: Mechanism of action data.
        drug_warning: Drug warnings data.
        indication_raw: Raw ChEMBL indication data.
        disease: Disease/EFO data for indication mapping.

    Returns:
        Final drug index DataFrame.
    """
    # Process indications
    indications = _process_indications(indication_raw, disease)

    # Get chemical probe drug IDs
    probes = (
        chemical_probes.filter(f.col('drugId').isNotNull())
        .select(f.col('drugId').alias('chemicalProbeDrugId'))
        .distinct()
    )

    # Get molecules with mechanism of action
    has_mechanism = (
        mechanism_of_action.select(f.explode(f.col('chemblIds')).alias('id'))
        .distinct()
        .withColumn('hasMechanismOfAction', f.lit(True))
    )

    # Process warnings to get blackBoxWarning and hasBeenWithdrawn flags
    warnings = _process_warnings(drug_warning)

    # Join all data together
    drug_df = (
        molecule.join(probes, molecule['id'] == probes['chemicalProbeDrugId'], 'left_outer')
        .join(has_mechanism, on='id', how='left_outer')
        .join(indications.select('id', 'indications'), on='id', how='left_outer')
        .join(warnings, on='id', how='left_outer')
    )

    # Filter to only include "drugs" - molecules that have:
    # - a drugbank cross-reference, OR
    # - indications, OR
    # - mechanism of action, OR
    # - are a chemical probe
    is_drug = (
        f.expr("array_contains(transform(crossReferences, x -> x.source), 'drugbank')")
        | f.col('indications').isNotNull()
        | f.col('hasMechanismOfAction').isNotNull()
        | f.col('chemicalProbeDrugId').isNotNull()
    )

    # Compute maximumClinicalTrialPhase from indications
    drug_df = drug_df.withColumn(
        'maximumClinicalTrialPhase',
        f.expr('aggregate(indications, cast(0 as double), (acc, x) -> greatest(acc, x.maxPhaseForIndication))'),
    )

    # Add description
    drug_df = _add_description(drug_df)

    # Filter and cleanup
    return (
        drug_df.filter(is_drug)
        .drop(
            'chemicalProbeDrugId',
            'hasMechanismOfAction',
            'indications',
            'maximumClinicalTrialPhase',
            'blackBoxWarning',
            'hasBeenWithdrawn',
        )
        .transform(_cleanup)
        .dropDuplicates(['id'])
    )


def _process_indications(indication_raw: DataFrame, disease: DataFrame) -> DataFrame:
    """Process raw ChEMBL indication data.

    Args:
        indication_raw: Raw ChEMBL indication JSONL data.
        disease: Disease/EFO data for name mapping.

    Returns:
        DataFrame with id and indications columns.
    """
    # Prepare EFO lookup - handle obsolete terms
    efo = disease.select(
        f.col('id').alias('updatedEfo'),
        f.trim(f.lower(f.col('name'))).alias('efoName'),
        f.array_union(
            f.array(f.col('id')),
            f.coalesce(f.col('obsoleteTerms'), f.array()),
        ).alias('allEfoIds'),
    ).withColumn(
        'allEfoIds',
        f.transform(f.col('allEfoIds'), lambda x: f.translate(x, ':', '_')),
    )

    # Process raw indication data
    indication = (
        indication_raw.select(
            f.col('_metadata.all_molecule_chembl_ids').alias('ids'),
            f.explode(f.col('indication_refs')).alias('ref'),
            f.col('max_phase_for_ind').alias('maxPhaseForIndication'),
            f.translate(f.col('efo_id'), ':', '_').alias('disease'),
        )
        .withColumn('ref_id', f.split(f.col('ref.ref_id'), ','))
        .withColumn('source', f.col('ref.ref_type'))
        .drop('ref')
        .groupBy('ids', 'maxPhaseForIndication', 'disease', 'source')
        .agg(f.collect_set('ref_id').alias('ref_id'))
        .withColumn(
            'references',
            f.struct(
                f.col('source'),
                f.flatten(f.col('ref_id')).alias('ids'),
            ),
        )
        .groupBy('ids', 'maxPhaseForIndication', 'disease')
        .agg(f.collect_set('references').alias('references'))
    )

    # Join with EFO to get disease names and updated IDs
    indication_with_disease = (
        indication.join(
            efo,
            f.array_contains(efo['allEfoIds'], indication['disease']),
        )
        .drop('allEfoIds', 'disease')
        .withColumnRenamed('updatedEfo', 'disease')
    )

    # Create final indication structure grouped by molecule ID
    return (
        indication_with_disease.withColumn('id', f.explode(f.col('ids')))
        .withColumn(
            'indications',
            f.struct(
                f.col('disease'),
                f.col('efoName'),
                f.col('references'),
                f.col('maxPhaseForIndication').cast('double').alias('maxPhaseForIndication'),
            ),
        )
        .groupBy('id')
        .agg(f.collect_set('indications').alias('indications'))
    )


def _process_warnings(drug_warning: DataFrame) -> DataFrame:
    """Process drug warnings to extract blackBoxWarning and hasBeenWithdrawn flags.

    Args:
        drug_warning: Drug warnings data.

    Returns:
        DataFrame with id, blackBoxWarning, and hasBeenWithdrawn columns.
    """
    return (
        drug_warning.select(
            f.explode(f.col('chemblIds')).alias('id'),
            f.col('warningType'),
        )
        .groupBy('id')
        .agg(
            f.max(f.when(f.col('warningType') == 'Black Box Warning', True).otherwise(False)).alias(
                'blackBoxWarning'
            ),
            f.max(f.when(f.col('warningType') == 'Withdrawn', True).otherwise(False)).alias(
                'hasBeenWithdrawn'
            ),
        )
    )


def _add_description(df: DataFrame) -> DataFrame:
    """Add human-readable description to drug data.

    The description summarizes key drug attributes including type, clinical phase,
    indications, and safety information.

    Args:
        df: Drug DataFrame with required columns.

    Returns:
        DataFrame with description column added.
    """
    # Register UDF for description generation
    from pyspark.sql import SparkSession

    spark = SparkSession.getActiveSession()
    if spark is None:
        msg = 'No active SparkSession found'
        raise RuntimeError(msg)
    spark.udf.register('generate_description', _generate_description, StringType())

    # Prepare indication data for description
    df = df.withColumn(
        '_indication_phases',
        f.expr('transform(coalesce(indications, array()), x -> x.maxPhaseForIndication)'),
    ).withColumn(
        '_indication_labels',
        f.expr('transform(coalesce(indications, array()), x -> x.efoName)'),
    )

    # Generate description using UDF
    df = df.withColumn(
        'description',
        f.expr(
            """generate_description(
            drugType,
            coalesce(maximumClinicalTrialPhase, cast(-1 as double)),
            _indication_phases,
            _indication_labels,
            coalesce(hasBeenWithdrawn, false),
            coalesce(blackBoxWarning, false)
        )"""
        ),
    )

    return df.drop('_indication_phases', '_indication_labels')


def _generate_description(
    drug_type: str | None,
    max_phase: float | None,
    indication_phases: list[float] | None,
    indication_labels: list[str] | None,
    is_withdrawn: bool,
    black_box_warning: bool,
) -> str:
    """Generate a human-readable description of a drug.

    Args:
        drug_type: Type of drug (e.g., "Small molecule").
        max_phase: Maximum clinical trial phase (-1 if unknown).
        indication_phases: List of indication phases.
        indication_labels: List of indication disease names.
        is_withdrawn: Whether the drug has been withdrawn.
        black_box_warning: Whether the drug has a black box warning.

    Returns:
        Human-readable description string.
    """
    if drug_type is None:
        drug_type = 'Unknown'

    roman_numbers = {4.0: 'IV', 3.0: 'III', 2.0: 'II', 1.0: 'I', 0.5: 'I (Early)'}

    # Main drug type note
    main_note = f'{drug_type.capitalize()} drug'

    # Clinical phase
    phase_str = ''
    if max_phase is not None and max_phase > 0:
        phase_roman = roman_numbers.get(max_phase, '')
        if phase_roman:
            label_count = len(indication_labels) if indication_labels else 0
            multi_indication = ' (across all indications)' if label_count > 1 else ''
            phase_str = f' with a maximum clinical trial phase of {phase_roman}{multi_indication}'

    # Process indications
    indication_str = ''
    if indication_phases is not None and indication_labels is not None:
        indications = list(zip(indication_phases, indication_labels, strict=False))
        indications = [
            (phase, label)
            for phase, label in indications
            if phase is not None and label is not None
        ]
        indications = list(set(indications))

        approved = [label for phase, label in indications if phase == 4.0]
        investigational_count = sum(1 for phase, _ in indications if phase < 4.0)

        if approved and not investigational_count:
            if len(approved) <= 2:
                indication_str = f' and is indicated for {_join_semantic(approved)}'
            else:
                indication_str = f' and has {len(approved)} approved indications'
        elif not approved and investigational_count:
            s = 's' if investigational_count > 1 else ''
            indication_str = f' and has {investigational_count} investigational indication{s}'
        elif approved and investigational_count:
            s = 's' if investigational_count > 1 else ''
            if len(approved) <= 2:
                approved_str = _join_semantic(approved)
                indication_str = (
                    f' and is indicated for {approved_str}'
                    f' and has {investigational_count} investigational indication{s}'
                )
            else:
                indication_str = (
                    f' and has {len(approved)} approved'
                    f' and {investigational_count} investigational indication{s}'
                )

    # Main sentence
    main_sentence = f'{main_note}{phase_str}{indication_str}.'

    # Withdrawn note
    withdrawn_note = ' It was withdrawn in at least one region.' if is_withdrawn else ''

    # Black box warning note
    black_box_note = ' This drug has a black box warning from the FDA.' if black_box_warning else ''

    return f'{main_sentence}{withdrawn_note}{black_box_note}'


def _join_semantic(items: list[str]) -> str:
    """Join items in a grammatically correct way.

    Args:
        items: List of strings to join.

    Returns:
        Joined string (e.g., "a, b and c").
    """
    if not items:
        return ''
    if len(items) == 1:
        return items[0]
    return f'{", ".join(items[:-1])} and {items[-1]}'


def _cleanup(df: DataFrame) -> DataFrame:
    """Ensure array columns have empty arrays instead of nulls.

    Args:
        df: DataFrame to clean up.

    Returns:
        Cleaned DataFrame.
    """
    for column in ['tradeNames', 'synonyms']:
        if column in df.columns:
            df = df.withColumn(column, f.coalesce(f.col(column), f.array()))
    return df
