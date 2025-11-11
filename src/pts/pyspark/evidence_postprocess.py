"""Module to process Platform evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from itertools import chain
from typing import Any

from loguru import logger
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql.window import Window

from pts.pyspark.common.session import Session
from pts.pyspark.common.utils import (
    linear_rescaling,
    pvalue_linear_rescaling,
    required_columns,
    update_quality_flag,
)

MECHANISM_OF_ACTION_MAP = {
    # Inhibitor list:
    'RNAI INHIBITOR': 'LoF',
    'NEGATIVE MODULATOR': 'LoF',
    'NEGATIVE ALLOSTERIC MODULATOR': 'LoF',
    'ANTAGONIST': 'LoF',
    'ANTISENSE INHIBITOR': 'LoF',
    'BLOCKER': 'LoF',
    'INHIBITOR': 'LoF',
    'DEGRADER': 'LoF',
    'INVERSE AGONIST': 'LoF',
    'ALLOSTERIC ANTAGONIST': 'LoF',
    'DISRUPTING AGENT': 'LoF',
    # Activator list:
    'PARTIAL AGONIST': 'GoF',
    'ACTIVATOR': 'GoF',
    'POSITIVE ALLOSTERIC MODULATOR': 'GoF',
    'POSITIVE MODULATOR': 'GoF',
    'AGONIST': 'GoF',
    'SEQUESTERING AGENT': 'GoF',
    'STABILISER': 'GoF',
}


@dataclass
class LookUpTables:
    session: Session
    settings: dict[str, Any]
    disease_lut: DataFrame | None = None
    target_lut: DataFrame | None = None
    publication_lut: DataFrame | None = None
    mechanism_of_action: DataFrame | None = None

    def __post_init__(self: LookUpTables) -> None:
        # Process disease:
        if disease_lut_path := self.settings.get('disease_path'):
            self.disease_lut = self._prepare_disease_lut(self.session.load_data(disease_lut_path, 'parquet'))

        # Process target:
        if target_lut_path := self.settings.get('target_path'):
            self.target_lut = self._prepare_target_lut(self.session.load_data(target_lut_path, 'parquet'))

        # Process publication:
        if publication_lut_path := self.settings.get('publication_date_lut'):
            self.publication_lut = self._prepare_publication_lut(self.session.load_data(publication_lut_path, 'json'))

        # mechanism of action is optional. Read if available:
        if mechanism_of_action_lut_path := self.settings.get('mechanism_of_action'):
            self.mechanism_of_action_lut = self._prepare_moa_lut(
                self.session.load_data(mechanism_of_action_lut_path, 'parquet')
            )
        else:
            self.mechanism_of_action_lut = None

    @staticmethod
    # @required_columns(['id', 'obsoleteTerms'])
    def _prepare_disease_lut(df: DataFrame) -> DataFrame:
        """Prepare disease index for disease validation.

        Args:
            df (DataFrame):raw, unprocessed disease index.

        Returns:
            DataFrame: dataframe with column `diseaseId` and `diseaseFromSourceMappedId`.
        """
        return (
            df.select(
                f.col('id').alias('diseaseId'),
                f.explode(
                    f.concat(
                        f.array(f.col('id')),
                        f.coalesce(f.col('obsoleteTerms'), f.lit([]).cast(t.ArrayType(t.StringType()))),
                    )
                ).alias('diseaseFromSourceMappedId'),
            )
            .orderBy(f.col('diseaseFromSourceMappedId').asc())
            .repartition(f.col('diseaseFromSourceMappedId'))
        )

    @staticmethod
    # @required_columns(['id', 'biotype', 'proteinIds', 'approvedSymbol'])
    def _prepare_target_lut(df: DataFrame) -> DataFrame:
        """Prepare target index for target validation.

        Args:
            df (DataFrame): raw, unprocessed target index.

        Returns:
            DataFrame: dataframe with column `targetId` and `targetFromSourceId`.
        """
        return (
            df.select(
                f.col('id').alias('targetId'),
                'biotype',
                f.array_distinct(
                    f.flatten(
                        f.array(
                            f.array(f.col('id')),
                            f.coalesce(
                                f.transform(f.col('proteinIds'), lambda protein: protein.id),
                                f.array().cast(t.ArrayType(t.StringType())),
                            ),
                            f.array(f.col('approvedSymbol')),
                        )
                    )
                ).alias('targetFromSourceIds'),
                LookUpTables._get_cancer_gene_assessment(f.col('hallmarks.attributes')).alias('TSorOncogene'),
            )
            .select('targetId', 'biotype', 'TSorOncogene', f.explode('targetFromSourceIds').alias('targetFromSourceId'))
            .distinct()
            .orderBy(f.col('targetFromSourceId').asc())
            .repartition(f.col('targetFromSourceId'))
        )

    @staticmethod
    # @required_columns(['source', 'firstPublicationDate', 'pmid', 'id', 'pmcid'])
    def _prepare_publication_lut(df: DataFrame) -> DataFrame:
        """Perform modification on the raw publication lookup table.

        Args:
            df (DataFrame): raw literature dataset

        Returns:
            DataFrame: processed lookup table.
        """
        return (
            # Filtering for selected literature sources:
            df.filter(f.col('source').isin('MED', 'PPR', 'AGR'))
            # Extracing relevant coclumns:
            .select(
                f.col('firstPublicationDate').alias('publicationDate'),
                f.explode(f.expr('filter(array(pmid, id, pmcid), x -> x is not null)')).alias('publicationId'),
            )
        )

    @staticmethod
    def _get_cancer_gene_assessment(hallmark_attributes: Column) -> Column:
        """Extract gene role in cancerous development based on cancer hallmark annotation.

        Cancer hallmark annotation provides a list of assessments of the gene. If they suggest
        oncogenic role we flag as `oncogene`, if they suggest tumor supressor function it is flagged as
        `tsg`. In case a gene has both `oncogene` and `tsg` annotation, it is flagged as `bivalent`.

        Args:
            hallmark_attributes (Column): hallmark column from the target index

        Returns:
            Column: with oncogenic/tumor supressor asessment.
        """
        # Get a cleaned version of the hallmark attribute description:
        hallmark_descriptions = f.transform(hallmark_attributes, lambda attribute: f.lower(attribute.description))

        # Flagging 'oncogene' and 'tsg' (tumor suppressor) annotations:
        flagged = f.array_distinct(
            f.transform(
                hallmark_descriptions,
                lambda desc: f.when(desc.contains('oncogene') & desc.contains('tsg'), f.lit('bivalent'))
                .when(desc.contains('oncogene'), f.lit('oncogene'))
                .when(desc.contains('tsg'), f.lit('tsg')),
            )
        )

        # Resolving direction:
        return (
            f.when(f.array_contains(flagged, 'bivalent'), f.lit('bivalent'))
            .when(f.array_contains(flagged, 'oncogene') & f.array_contains(flagged, 'tsg'), f.lit('bivalent'))
            .when(f.array_contains(flagged, 'oncogene'), f.lit('oncogene'))
            .when(f.array_contains(flagged, 'tsg'), f.lit('tsg'))
        )

    @staticmethod
    def _prepare_moa_lut(moa_df: DataFrame) -> DataFrame:
        """Generate look up table for chembl evidence with unified MoA assessment.

        Args:
            moa_df (DataFrame): raw mechanism of action dataset.

        Returns:
            DataFrame: drug/target look up -> moa type.
        """
        # Create moa map:
        moa_mapping_expr = f.create_map([f.lit(x) for x in chain(*MECHANISM_OF_ACTION_MAP.items())])

        return (
            moa_df
            # Mapping action types:
            .withColumn('moaType', f.coalesce(moa_mapping_expr.getItem(f.col('actionType')), f.lit('noEvaluable')))
            # Exploding drugs:
            .select(
                f.explode_outer('chemblIds').alias('drugId'),
                'moaType',
                'targets',
            )
            # Exploding diseases:
            .select(
                f.explode_outer('targets').alias('targetId'),
                'drugId',
                'moaType',
            )
            # Aggregate action types by drug/target pairs:
            .groupBy('targetId', 'drugId')
            .agg(
                f.collect_set('moaType').alias('moaTypes'),
            )
            .withColumn(
                'moaType',
                f.when(
                    f.array_contains('moaTypes', 'GoF') & f.array_contains('moaTypes', 'LoF'),
                    f.lit(None).cast(t.StringType()),
                )
                .when(f.array_contains('moaTypes', 'GoF'), f.lit('GoF'))
                .when(f.array_contains('moaTypes', 'LoF'), f.lit('LoF')),
            )
            .select('targetId', 'drugId', 'moaType')
        )


class EvidenceFlags(StrEnum):
    INVALID_DISEASE = 'No valid disease'
    INVALID_TARGET = 'No valid target'
    DUPLICATED = 'Duplicated'
    NO_VALID_SCORE = 'No valid score'
    INVALID_BIOTYPE = 'Invalid biotype'


@dataclass
class Evidence:
    # Incoming evidence:
    _df: DataFrame

    # Hardcoded column name to capture direction on target:
    DIRECTION_ON_TARGET_COLUMN_NAME: str = 'directionOnTarget'

    # Hardcoded values for evidence processing:
    QC_COLUMN: str = 'qualityControls'

    # Default lenght of variant identifier to start hashing:
    VARIANT_HASH_LENGHT: int = 300

    # Columns we consider for dating evidence:
    EVIDENCE_DATE_COLUMNS: list[str] = field(
        default_factory=lambda: ['publicationDate', 'curationDate', 'studyStartDate']
    )

    def __post_init__(self: Evidence) -> None:
        """Initial process of the evidence."""
        # Initialise QC column if not given:
        if self.QC_COLUMN not in self.df.columns:
            self._df = self._df.withColumn(self.QC_COLUMN, f.lit([]).cast(t.ArrayType(t.StringType())))

    @required_columns(['targetFromSourceId'])
    def validate_target(self: Evidence, target_lut: DataFrame, invalid_biotypes: list[str] | None = None) -> Evidence:
        """Validation of targets.

        Args:
            target_lut (DataFrame): processed target look-up-table.
            invalid_biotypes (list[str] | None): if provided, evidence will be flagged with target of invalid biotype

        Returns:
            Evidence: evidence with mapped targets, and flaggeed evidence without target.
        """
        # If invalid biotypes are not provided, create empty array:
        if not invalid_biotypes:
            invalid_biotypes = []

        return Evidence(
            self.df
            # Resolve target identifiers:
            .join(
                f.broadcast(target_lut).select('targetId', 'biotype', 'targetFromSourceId'),
                on='targetFromSourceId',
                how='leftouter',
            )
            # Flag evidence without mapped targets:
            .withColumn(
                self.QC_COLUMN,
                update_quality_flag(f.col(self.QC_COLUMN), f.col('targetId').isNull(), EvidenceFlags.INVALID_TARGET),
            )
            # Flag evidence without target of invalid biotype:
            .withColumn(
                self.QC_COLUMN,
                update_quality_flag(
                    f.col(self.QC_COLUMN), f.col('biotype').isin(invalid_biotypes), EvidenceFlags.INVALID_BIOTYPE
                ),
            )
            .drop('biotype')
        )

    @required_columns(['diseaseFromSourceMappedId'])
    def validate_diseases(self: Evidence, disease_lut: DataFrame) -> Evidence:
        """Validation of diseases.

        Args:
            disease_lut (DataFrame): processed disease look-up-table.

        Returns:
            Evidence: evidence with mapped targets, and flaggeed evidence without target.
        """
        return Evidence(
            self.df
            # Resolve target identifiers:
            .join(disease_lut, on='diseaseFromSourceMappedId', how='leftouter')
            # Flag evidence without mapped targets:
            .withColumn(
                self.QC_COLUMN,
                update_quality_flag(f.col(self.QC_COLUMN), f.col('diseaseId').isNull(), EvidenceFlags.INVALID_DISEASE),
            )
        )

    @required_columns(['id'])
    def validate_uniqueness(self: Evidence) -> Evidence:
        """Validate uniqueness of evidence.

        Grouping evidence by evidence identifiers. If more than one evidence share the same ID all but one
        will be flagged.

        Returns:
            Evidence: where non-unique evidence is flagged.
        """
        return Evidence(
            self.df.withColumn('evidence_unique_rank', f.rank().over(Window.partitionBy('id').orderBy(f.rand())))
            .withColumn(
                self.QC_COLUMN,
                update_quality_flag(
                    f.col(self.QC_COLUMN), f.col('evidence_unique_rank') != 1, EvidenceFlags.DUPLICATED
                ),
            )
            .drop('evidence_unique_rank')
        )

    @required_columns(['datasourceId'])
    def validate_datasource(self: Evidence, datasource_id: str) -> Evidence:
        """Validate evidence for datasource identifier.

        Only the specified datasourceId will be kept.

        Args:
            datasource_id (str): identifier of the datasource we want to work with.

        Returns:
            Evidence: with only one datasource
        """
        return Evidence(self.df.filter(f.col('datasourceId') == datasource_id))

    def assign_evidence_identifier(self: Evidence, unique_fields: list[str]) -> Evidence:
        """Adding unique identifier to each evidence based on source specific fields.

        Args:
            unique_fields (list[str]): list of column names that define evidence uniqueness.

        Returns:
            Evidence: with new column (id)
        """
        # Maybe not all columns are actually in the dataframe:
        valid_columns = [
            f.coalesce(f.col(col).cast(t.StringType()), f.lit('null'))
            for col in unique_fields
            if col in self.df.columns
        ]

        # Generate a hash based on the concatenated fields:
        hash_expression = f.sha1(f.concat(*valid_columns))

        return Evidence(self.df.withColumn('id', hash_expression))

    def hash_long_variant_identifiers(self: Evidence) -> Evidence:
        """Hash long variant identifier.

        If `variantId` column is present in the dataset, long ids need to be hashed.
        """
        if 'variantId' not in self.df.columns:
            return self

        return Evidence(
            self.df.withColumn('variantId', self._hash_long_variant_ids(f.col('variantId'), self.VARIANT_HASH_LENGHT))
        )

    def assign_direction_on_target(
        self: Evidence,
        direction_on_target_expression: str | None,
        mechanism_of_action_lut: DataFrame | None,
        target_lut: DataFrame | None,
    ) -> Evidence:
        """Assign direction of target for evidence.

        Args:
            direction_on_target_expression (str | None): spark sql expression to compute direction
            mechanism_of_action_lut (DataFrame | None): drug mechanism of action look up table
            target_lut (DataFrame | None): target look up table

        Returns:
            Evidence: when expression is provided a new column is assigned.

        Raises:
            ValueError: if "drugId" column is missing if moa dataset is used.
        """
        unused_columns = ['moaType', 'TSorOncogene']

        # If no expression is provided return evidence unchanged:
        if direction_on_target_expression is None:
            return self

        evidence_df = self.df

        # Mechanism of action requires the presence of drugId column:
        if mechanism_of_action_lut:
            if 'drugId' not in evidence_df.columns:
                raise ValueError(
                    'To use mechanism of action to annotate effect on target, "drugId" column must be in evidence.'
                )

            # Update evidence data frame, apply expression, and drop unused column:
            evidence_df = evidence_df.join(mechanism_of_action_lut, on=['targetId', 'drugId'], how='left')

        if target_lut:
            evidence_df = evidence_df.join(
                target_lut.select('targetId', 'TSorOncogene').distinct(), on='targetId', how='left'
            )

        return Evidence(
            evidence_df.withColumn(self.DIRECTION_ON_TARGET_COLUMN_NAME, f.expr(direction_on_target_expression)).drop(
                *unused_columns
            )
        )

    def calculate_evidence_score(self: Evidence, score_expression: str) -> Evidence:
        """Using the provided score expression, assign evidence score to evidence.

        Args:
            score_expression (str): valid spark expression as string.

        Returns:
            Evidence: with assigned score
        """
        return Evidence(
            self.df
            # Calculate evidence score and make sure the type is consistent across different datasets:
            .withColumn('score', f.expr(score_expression).cast(t.DoubleType()))
            # Flag evidence invalid scores (missing, zero or negative, more than 1)
            .withColumn(
                self.QC_COLUMN,
                update_quality_flag(
                    f.col(self.QC_COLUMN),
                    f.when(
                        f.col('score').isNull() | (f.col('score') < 0) | (f.col('score') > 1), f.lit(True)
                    ).otherwise(f.lit(False)),
                    EvidenceFlags.NO_VALID_SCORE,
                ),
            )
        )

    @required_columns(['id'])
    def resolve_publication_date(self: Evidence, publication_date_lut: DataFrame) -> Evidence:
        """Date evidence based on literature references.

        ID is a mandatory column to run the logic, however the presence of literature column
        is not. If literature column is missing, self is returned.

        Args:
            publication_date_lut (DataFrarme): look up table with publication identifiers and date

        Returns:
            Evidence: with a new column `publicationDate` if literature is presents.
        """
        # Return self if publication column is not present in the dataset:
        if 'literature' not in self.df.columns:
            return self

        #  Extract publication IDs/evidence ID:
        evidence_with_pub_ids = (
            self.df.select(f.col('id'), f.explode(f.col('literature')).alias('publicationId'))
            .withColumn('publicationId', f.upper(f.trim(f.col('publicationId'))))
            .distinct()
        )

        # Join evidence with publication mapping:
        dated_evidence = (
            evidence_with_pub_ids.join(publication_date_lut, on='publicationId', how='inner')
            # For each evidence identifier find the earliest publication date:
            .withColumn('rank', f.row_number().over(Window.partitionBy('id').orderBy(f.col('publicationDate').asc())))
            .filter(f.col('rank') == 1)
            .select('id', 'publicationDate')
            .distinct()
        )

        # Broadcast for efficiency and join back to main evidence
        dated_evidence_lut = f.broadcast(dated_evidence.orderBy(f.col('id').asc()))

        return Evidence(self.df.join(dated_evidence_lut, on='id', how='left_outer'))

    def resolve_evidence_date(self: Evidence) -> Evidence:
        """Generate evidenceDate for each evidence based on a number of columns.

        Evidence date column has to be presented even if none of the eligible columns are present in the dataset.

        Returns:
            Evidence: with new column `evidenceDate`.
        """
        # Get which columns do we have:
        dating_columns = [f.col(col) for col in self.EVIDENCE_DATE_COLUMNS if col in self.df.columns]

        # If no columns are there, we just return:
        if len(dating_columns) == 0:
            dating_columns = [f.lit(None).cast(t.StringType())]

        # Assign date:
        return Evidence(self.df.withColumn('evidenceDate', f.array_min(f.array(dating_columns))))

    def resolve_direction_of_effect(self: Evidence, mechanism_of_action: DataFrame) -> Evidence:
        raise NotImplementedError

    @staticmethod
    def _hash_long_variant_ids(variant_id: Column, threshold: int = 300) -> Column:
        """Generate hash for long variant identifiers.

        Args:
            variant_id (Column): variant identifier column.
            threshold (int): lenght of the variant identifier above which we hash.

        Returns:
            Column: variant id column where long variants are hashed.
        """
        # Extract chromosome and position from variantId
        chr_col = f.regexp_extract(variant_id, r'^([0-9XYMT]{1,2})_([0-9]+)_([ACGTN]+)_([ACGTN]+)$', 1)
        pos_col = f.regexp_extract(variant_id, r'^([0-9XYMT]{1,2})_([0-9]+)_([ACGTN]+)_([ACGTN]+)$', 2)

        # Apply transformation logic
        return (
            f.when(chr_col.isNull() | pos_col.isNull(), f.concat(f.lit('OTVAR_'), f.md5(variant_id).cast('string')))
            .when(
                f.length(variant_id) > threshold,
                f.concat_ws('_', f.lit('OTVAR'), chr_col, pos_col, f.md5(variant_id).cast('string')),
            )
            .otherwise(variant_id)
        )

    @property
    def df(self: Evidence) -> DataFrame:
        return self._df

    @df.setter
    def df(self: Evidence, new_df: DataFrame) -> Evidence:
        return Evidence(new_df)

    def get_invalid_evidence(self: Evidence) -> DataFrame:
        """Return invalid evidence.

        Returns:
            DataFrame: evidence with non empty QC column.
        """
        return self.df.filter(f.size(self.QC_COLUMN) != 0)

    def get_valid_evidence(self: Evidence) -> DataFrame:
        """Return valid evidence.

        Returns:
            DataFrame: evidence with empty QC column.
        """
        return self.df.filter(f.size(self.QC_COLUMN) == 0)

    def assign_direction_on_trait(self: Evidence, direction_expression: str | None) -> Evidence:
        """Assigning direction on trait.

        Based on a datasource specific expression, the effect on the disease is assessed.

        Args:
            direction_expression (str | None): spark sql expression on how a given source is processed.

        Returns:
            Evidence: 'directionOnTrait' column if directions are provided.
        """
        # If no expresssion is provided, evidence returned unchanged:
        if direction_expression is None:
            return self

        return Evidence(self.df.withColumn('directionOnTrait', f.expr(direction_expression)))


def evidence_postprocess(
    source: dict[str, str], destination: dict[str, str], settings: dict[str, Any], properties: dict[str, str]
) -> None:
    datasource_id = settings['datasource_id']
    evidence_format = settings['evidence_format']
    score_expression = settings['score_expression']

    logger.info(f'processing "{datasource_id}" evidence')
    # Initialise session:
    session = Session(app_name='evidence', properties=properties)

    # Registering UDFs in the spark session:
    session.spark.udf.register('linear_rescale', linear_rescaling)
    session.spark.udf.register('pvalue_linear_score', pvalue_linear_rescaling)

    # Read input data:
    source_evidence = session.load_data(source['evidence_path'], format=evidence_format)

    # Reading look up tables and make surer all looks good:
    lookup_tables = LookUpTables(session, source)
    assert lookup_tables.disease_lut is not None, 'disease_lut not generated'
    assert lookup_tables.target_lut is not None, 'target_lut not generated'
    assert lookup_tables.publication_lut is not None, 'publication_lut not generated'

    # Processing evidence:
    processed_evidence = (
        Evidence(source_evidence)
        # Validate entities:
        .validate_diseases(lookup_tables.disease_lut)
        .validate_target(lookup_tables.target_lut, settings.get('excluded_biotypes'))
        .validate_datasource(settings['datasource_id'])
        # Assing evidence identifier + check duplication:
        .assign_evidence_identifier(settings['unique_fields'])
        .validate_uniqueness()
        # Resolving evidence date:
        .resolve_publication_date(lookup_tables.publication_lut)
        .resolve_evidence_date()
        # Calculating score:
        .calculate_evidence_score(score_expression)
        # Calculate direction of effect:
        .assign_direction_on_trait(settings.get('direction_on_trait_expression'))
        .assign_direction_on_target(
            direction_on_target_expression=settings.get('direction_on_target_expression'),
            mechanism_of_action_lut=lookup_tables.mechanism_of_action_lut,
            target_lut=lookup_tables.target_lut,
        )
        # Hash long variant identifiers:
        .hash_long_variant_identifiers()
    )

    # Writing outputs:
    processed_evidence.get_invalid_evidence().write.mode('overwrite').parquet(destination['failed_evidence'])
    processed_evidence.get_valid_evidence().write.mode('overwrite').parquet(destination['evidence'])
