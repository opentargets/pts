"""ChEMBL Molecule processing.

Processes raw ChEMBL molecule data into the Open Targets molecule format,
including synonyms, cross-references, and molecule hierarchy.
"""

from typing import Any

import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, MapType, StringType, StructField, StructType

from pts.pyspark.common.session import Session

CHEMBL_SOURCE = 'ChEMBL'
AACT_SOURCE = 'AACT'

LABEL_SOURCE_SCHEMA = ArrayType(
    StructType([
        StructField('label', StringType()),
        StructField('source', StringType()),
    ])
)


def _as_label_source(label_col, source_val):
    """Wrap a string column as a {label, source} struct."""
    return f.struct(label_col.alias('label'), f.lit(source_val).alias('source'))


def _normalize_name(col):
    """Lowercase, strip trademark symbols, trim, collapse internal whitespace."""
    stripped = f.regexp_replace(col, r'[®™©℠]', '')
    collapsed = f.regexp_replace(f.trim(stripped), r'\s+', ' ')
    return f.lower(collapsed)


_DRUG_LIST_SCHEMA = ArrayType(StructType([
    StructField('drug', StringType()),
    StructField('synonyms', ArrayType(StringType())),
]))

BATCH_INNER_SCHEMA = StructType([
    StructField('investigated_drugs', _DRUG_LIST_SCHEMA),
    StructField('comparator_drugs', _DRUG_LIST_SCHEMA),
    StructField('supportive_drugs', _DRUG_LIST_SCHEMA),
])


def _parse_aact_batch(batch_raw):
    """Parse OpenAI batch output into one row per drug entry with a normalized member set.

    Returns DataFrame[nct_id, members: array<string>] (normalized, deduped, non-empty).
    """
    messages = (
        batch_raw
        .select(
            f.col('custom_id').alias('nct_id'),
            f.explode('response.body.output').alias('out'),
        )
        .filter(f.col('out.type') == 'message')
        .select('nct_id', f.explode('out.content').alias('content'))
        # content.text is itself a JSON string (OpenAI structured output is double-encoded);
        # decode it into BATCH_INNER_SCHEMA.
        .select('nct_id', f.from_json(f.col('content.text'), BATCH_INNER_SCHEMA).alias('parsed'))
    )

    roles = f.array_union(
        f.array_union(
            f.coalesce(f.col('parsed.investigated_drugs'), f.array().cast(_DRUG_LIST_SCHEMA)),
            f.coalesce(f.col('parsed.comparator_drugs'), f.array().cast(_DRUG_LIST_SCHEMA)),
        ),
        f.coalesce(f.col('parsed.supportive_drugs'), f.array().cast(_DRUG_LIST_SCHEMA)),
    )

    return (
        messages
        .withColumn('entry', f.explode(roles))
        .withColumn(
            'members',
            f.array_union(
                f.array(f.col('entry.drug')),
                f.coalesce(f.col('entry.synonyms'), f.array().cast('array<string>')),
            ),
        )
        .withColumn(
            'members',
            f.array_distinct(
                f.filter(
                    f.transform(f.col('members'), _normalize_name),
                    lambda m: (m.isNotNull()) & (f.length(m) > 0),
                )
            ),
        )
        .filter(f.size('members') > 0)
        .select('nct_id', 'members')
    )


def _build_chembl_indexes(mol_df):
    """Build (name_index, regimen_index, parent_child) from ChEMBL-source names.

    name_index:    DataFrame[name_norm, ids: array<string>]
    regimen_index: DataFrame[regimen_norm, ids: array<string>]  (suppression only)
    parent_child:  DataFrame[id, related: array<string>]  (parent + children)
    """
    empty_ls = f.array().cast(LABEL_SOURCE_SCHEMA)
    labels = (
        mol_df
        .select(
            'id',
            f.array_union(
                f.array(f.col('name')),
                f.array_union(
                    f.transform(f.coalesce(f.col('synonyms'), empty_ls), lambda s: s['label']),  # noqa: FURB118
                    f.transform(f.coalesce(f.col('tradeNames'), empty_ls), lambda t: t['label']),  # noqa: FURB118
                ),
            ).alias('labels'),
        )
        .select('id', f.explode('labels').alias('label'))
        .withColumn('name_norm', _normalize_name(f.col('label')))
        .filter(f.length('name_norm') > 0)
    )

    name_index = labels.groupBy('name_norm').agg(f.collect_set('id').alias('ids'))

    # "<ingredient> COMPONENT OF <regimen>" -> regimen token (normalized text is lowercased)
    regimen_index = (
        labels
        .withColumn(
            'regimen_norm',
            f.regexp_extract(f.col('name_norm'), r'\bcomponent of\s+(.+)$', 1),
        )
        .filter(f.length('regimen_norm') > 0)
        .groupBy('regimen_norm')
        .agg(f.collect_set('id').alias('ids'))
    )

    empty_str_arr = f.array().cast('array<string>')
    children = mol_df.select(
        'id',
        f.coalesce(f.col('childChemblIds'), empty_str_arr).alias('related'),
    )
    parents = (
        mol_df
        .filter(f.col('parentId').isNotNull())
        .select('id', f.array(f.col('parentId')).alias('related'))
    )
    parent_child = (
        children.union(parents)
        .groupBy('id')
        .agg(f.array_distinct(f.flatten(f.collect_list('related'))).alias('related'))
    )

    return name_index, regimen_index, parent_child


AMBIGUITY_CAP = 10

# v1 port of the experiment's cleanup blacklists — expected to grow with corpus coverage.
CODE_REGEX = r'\b[a-z]{1,6}-?\d{3,}[a-z0-9]*\b'

# v1 port of the experiment's cleanup blacklists — expected to grow with corpus coverage.
CONTROL_TERMS = {
    'placebo', 'vehicle', 'saline', 'sham', 'soc', 'standard of care', 'study drug',
    'sodium chloride', 'water', 'air', 'normal saline',
}
# v1 port of the experiment's cleanup blacklists — expected to grow with corpus coverage.
CLASS_KEYWORDS = [
    'inhibitor', 'agonist', 'antagonist', 'antibody', 'analogue', 'analog', 'therapy',
    'statin', 'steroid', 'nsaid', 'cell', 'cells', 'lymphocyte', 'lymphocytes',
    'mesenchymal', 'stromal', 'progenitor', 'fibroblast',
]
_CLASS_PATTERN = r'\b(' + '|'.join(CLASS_KEYWORDS) + r')\b'


def _has_class_keyword(col):
    """True when the candidate text contains any drug-class / cell-therapy keyword as a whole word."""
    return col.rlike(_CLASS_PATTERN)


def _apply_cleanup_rules(cand, regimen_index, existing_per_id):
    """Apply rules #5-#11 + drop PARENT_CHILD. Returns DataFrame[id, candidate, nct_id].

    Args:
        cand: DataFrame[id, candidate, nct_id, status]
        regimen_index: DataFrame[regimen_norm, ids: array<string>]
        existing_per_id: DataFrame[id, existing: array<string>]

    Returns:
        DataFrame[id, candidate, nct_id] with noise filtered out.
    """
    # drop PARENT_CHILD (keep NOVEL + CONFLICT)
    cand = cand.filter(f.col('status') != 'PARENT_CHILD')

    # #8: descriptor-wrapped code -> bare code (when phrase has a class word AND a code)
    cand = cand.withColumn('code', f.regexp_extract(f.col('candidate'), CODE_REGEX, 0))
    cand = cand.withColumn(
        'candidate',
        f.when(
            (f.length('code') > 0) & _has_class_keyword(f.col('candidate')),
            f.col('code'),
        ).otherwise(f.col('candidate')),
    ).drop('code')

    # #10: single-character
    cand = cand.filter(f.length('candidate') > 1)

    # #9: insulin units + any '%'
    cand = cand.filter(~f.col('candidate').rlike(r'^(u|gla)[- ]?\d{2,3}$'))
    cand = cand.filter(~f.col('candidate').contains('%'))

    # #5: control noise
    control_array = f.array([f.lit(t) for t in sorted(CONTROL_TERMS)])
    cand = cand.filter(~f.array_contains(control_array, f.col('candidate')))

    # #6: drug-class / cell-therapy keyword present, UNLESS a code survived (#8 kept the code)
    cand = cand.filter(~_has_class_keyword(f.col('candidate')) | f.col('candidate').rlike(CODE_REGEX))

    # #7: regimen suppression (candidate equals a known regimen token)
    regimen_keys = regimen_index.select(f.col('regimen_norm').alias('candidate')).distinct()
    cand = cand.join(regimen_keys.withColumn('_is_regimen', f.lit(True)), on='candidate', how='left')
    cand = cand.filter(f.col('_is_regimen').isNull()).drop('_is_regimen')

    # #11: plural suppression (singular already on M)
    cand = cand.withColumn(
        'singular',
        f.when(f.col('candidate').endswith('ies'),
               f.concat(f.expr('left(candidate, length(candidate) - 3)'), f.lit('y')))
        .when(f.col('candidate').endswith('es'), f.expr('left(candidate, length(candidate) - 2)'))
        .when(f.col('candidate').endswith('s'), f.expr('left(candidate, length(candidate) - 1)'))
        .otherwise(f.col('candidate')),
    )
    cand = cand.join(existing_per_id, on='id', how='left')
    cand = cand.filter(
        (f.col('singular') == f.col('candidate'))
        | ~f.array_contains(f.coalesce(f.col('existing'), f.array().cast('array<string>')), f.col('singular'))
    ).drop('singular', 'existing')

    return cand.select('id', 'candidate', 'nct_id').distinct()


def _anchor_candidates(entries, name_index, parent_child):
    """Anchor member sets to molecules and emit (id, candidate, nct_id, status).

    For each trial drug entry (a normalized member set), resolve members against
    name_index to find which ChEMBL molecule(s) the entry anchors to, then emit
    each member that is NOT already on an anchored molecule as a candidate
    synonym, classified by status.

    Entries where any single member resolves to more than AMBIGUITY_CAP molecules
    are dropped entirely.

    Args:
        entries: DataFrame[nct_id, members: array<string>]
        name_index: DataFrame[name_norm, ids: array<string>]
        parent_child: DataFrame[id, related: array<string>]

    Returns:
        DataFrame[id, candidate, nct_id, status] where id is an anchored
        molecule, candidate is a member not already on id, and status is one of
        NOVEL / PARENT_CHILD / CONFLICT.
    """
    entries = entries.withColumn('entry_id', f.monotonically_increasing_id())

    members = entries.select('entry_id', 'nct_id', f.explode('members').alias('member'))

    # resolve each member against name_index; unresolved -> empty ids array
    resolved = members.join(name_index, members['member'] == name_index['name_norm'], 'left').select(
        'entry_id',
        'nct_id',
        'member',
        f.coalesce(f.col('ids'), f.array().cast('array<string>')).alias('ids'),
    )

    # poison: drop any entry where a single member resolves to > AMBIGUITY_CAP molecules
    poisoned = (
        resolved
        .groupBy('entry_id')
        .agg(f.max(f.size('ids')).alias('max_ids'))
        .filter(f.col('max_ids') > AMBIGUITY_CAP)
        .select('entry_id')
    )
    resolved = resolved.join(poisoned, on='entry_id', how='left_anti')

    # collect the union of all resolved molecule ids per entry (anchor set)
    anchors = (
        resolved
        .select('entry_id', f.explode('ids').alias('anchor_id'))
        .groupBy('entry_id')
        .agg(f.collect_set('anchor_id').alias('anchor_ids'))
    )

    # cross each member with each anchored molecule of its entry
    cand = resolved.join(anchors, on='entry_id', how='inner')
    cand = cand.withColumn('anchor_id', f.explode('anchor_ids'))

    # drop members already belonging to the anchor molecule (not candidates for it)
    cand = cand.filter(~f.array_contains(f.col('ids'), f.col('anchor_id')))

    # join parent_child info for the anchor to determine status
    pc = parent_child.withColumnRenamed('id', 'anchor_id').withColumnRenamed('related', 'pc_related')
    cand = cand.join(pc, on='anchor_id', how='left')

    empty_str_arr = f.array().cast('array<string>')
    cand = cand.withColumn(
        'status',
        f.when(f.size('ids') == 0, f.lit('NOVEL'))
        .when(
            f.arrays_overlap(f.col('ids'), f.coalesce(f.col('pc_related'), empty_str_arr)),
            f.lit('PARENT_CHILD'),
        )
        .otherwise(f.lit('CONFLICT')),
    )

    return cand.select(
        f.col('anchor_id').alias('id'),
        f.col('member').alias('candidate'),
        'nct_id',
        'status',
    ).distinct()


MIN_TRIALS = 2


def _mine_aact_synonyms(mol_df, entries):
    """Full AACT mining: anchor -> cleanup -> n_trials>=MIN_TRIALS -> DataFrame[id, label].

    The stored label is the normalized candidate string (v1: normalized form, which
    matches the anchor index; surface-form refinement is deferred).
    """
    name_index, regimen_index, parent_child = _build_chembl_indexes(mol_df)

    # Per-molecule set of normalized existing names (name + synonym/tradeName labels),
    # used by rule #11 plural suppression. Intentionally parallels the label collection
    # in _build_chembl_indexes (different shape: grouped array vs exploded rows).
    empty_ls = f.array().cast(LABEL_SOURCE_SCHEMA)
    existing_per_id = mol_df.select(
        'id',
        f.array_union(
            f.array(_normalize_name(f.col('name'))),
            f.array_union(
                f.transform(f.coalesce(f.col('synonyms'), empty_ls), lambda s: _normalize_name(s['label'])),
                f.transform(f.coalesce(f.col('tradeNames'), empty_ls), lambda t: _normalize_name(t['label'])),
            ),
        ).alias('existing'),
    )

    anchored = _anchor_candidates(entries, name_index, parent_child)
    cleaned = _apply_cleanup_rules(anchored, regimen_index, existing_per_id)

    return (
        cleaned
        .groupBy('id', 'candidate')
        .agg(f.countDistinct('nct_id').alias('n_trials'))
        .filter(f.col('n_trials') >= MIN_TRIALS)
        .select('id', f.col('candidate').alias('label'))
    )


def chembl_molecule(
    source: dict[str, str],
    destination: str,
    _settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Process ChEMBL molecule data.

    Args:
        source: Dictionary with paths to:
            - chembl_molecule: ChEMBL molecule JSONL
            - drugbank: Drugbank to ChEMBL ID mapping CSV
        destination: Path to write the output parquet file.
        _settings: Custom settings (not used).
        properties: Spark configuration options.
    """
    spark = Session(app_name='chembl_molecule', properties=properties)

    logger.info(f'Loading data from {source}')
    molecule_df = spark.load_data(source['chembl_molecule'], format='json')
    drugbank_df = spark.load_data(
        source['drugbank'],
        format='csv',
        header=True,
        sep='\t',
    )

    logger.info('Processing molecules')
    output_df = process_molecules(molecule_df, drugbank_df)

    logger.info(f'Writing molecules to {destination}')
    output_df.write.parquet(destination, mode='overwrite')


def process_molecules(
    molecule_raw: DataFrame,
    drugbank_lookup: DataFrame,
) -> DataFrame:
    """Process raw ChEMBL molecule data.

    Args:
        molecule_raw: Raw ChEMBL molecule data.
        drugbank_lookup: Drugbank to ChEMBL ID mapping.

    Returns:
        Processed molecule DataFrame.
    """
    # Prepare drugbank lookup - rename columns to match expected format
    drugbank = drugbank_lookup.select(
        f.col("From src:'1'").alias('id'),
        f.col("To src:'2'").alias('drugbank_id'),
    )

    # Preprocess molecules
    mols = _molecule_preprocess(molecule_raw, drugbank)

    # Process components
    synonyms = _process_molecule_synonyms(mols)
    cross_references = _process_molecule_cross_references(mols)
    hierarchy = _process_molecule_hierarchy(mols)

    # Combine all components
    mol_combined = (
        mols
        .drop('cross_references', 'syns')
        .join(synonyms, on='id', how='left_outer')
        .join(cross_references, on='id', how='left_outer')
        .join(hierarchy, on='id', how='left_outer')
    )

    empty_label_source = f.array().cast(LABEL_SOURCE_SCHEMA)

    # Final processing - ensure name is populated and deduplicate
    return (
        mol_combined
        .withColumn('synonyms', f.coalesce(f.col('synonyms'), empty_label_source))
        .withColumn('tradeNames', f.coalesce(f.col('tradeNames'), empty_label_source))
        .withColumn(
            'name',
            f.coalesce(
                f.col('name'),
                f.element_at(
                    f.filter(f.col('synonyms'), lambda s: s['source'] == CHEMBL_SOURCE),
                    1,
                )['label'],
                f.col('id'),
            ),
        )
        .drop('drugbank_id')
        .dropDuplicates(['id'])
    )


def _molecule_preprocess(
    molecule_raw: DataFrame,
    drugbank: DataFrame,
) -> DataFrame:
    """Preprocess raw molecule data.

    Args:
        molecule_raw: Raw ChEMBL molecule data.
        drugbank: Drugbank lookup table.

    Returns:
        Preprocessed molecule DataFrame.
    """
    return (
        molecule_raw
        .select(
            f.col('molecule_chembl_id').alias('id'),
            f.col('molecule_structures.canonical_smiles').alias('canonicalSmiles'),
            f.col('molecule_structures.standard_inchi_key').alias('inchiKey'),
            # ChEMBL ships molfile as a full SD-file record (molblock + appended
            # SDF property tags). Truncate to the bare molblock by dropping
            # everything after the `M  END` terminator. If `M  END` is absent the
            # string is left unchanged.
            f.regexp_replace(
                f.col('molecule_structures.molfile'),
                r'(?s)(\nM  END\n).*',
                '$1',
            ).alias('molblock'),
            f.coalesce(f.col('molecule_type'), f.lit('Unknown')).alias('drugType'),
            f.col('pref_name').alias('name'),
            f.col('cross_references'),
            f.col('molecule_hierarchy.parent_chembl_id').alias('parentId'),
            f.col('molecule_synonyms.molecule_synonym').alias('mol_synonyms'),
            f.col('molecule_synonyms.syn_type').alias('synonym_type'),
        )
        .withColumn('syns', f.arrays_zip(f.col('mol_synonyms'), f.col('synonym_type')))
        # Remove circular references
        .withColumn(
            'parentId',
            f.when(f.col('parentId') == f.col('id'), f.lit(None)).otherwise(f.col('parentId')),
        )
        .drop('mol_synonyms', 'synonym_type')
        .join(drugbank, on='id', how='left_outer')
    )


def _process_molecule_synonyms(preprocessed_mols: DataFrame) -> DataFrame:
    """Group synonyms into sorted sets of trade names and other synonyms.

    Args:
        preprocessed_mols: Preprocessed molecule DataFrame.

    Returns:
        DataFrame with id, tradeNames, and synonyms columns.
    """
    synonyms = (
        preprocessed_mols
        .select(f.col('id'), f.explode('syns').alias('col'))
        .withColumn('syn_type', f.upper(f.col('col.synonym_type')))
        .withColumn('synonym', f.col('col.mol_synonyms'))
    )

    trade_names = (
        synonyms
        .filter(f.col('syn_type') == 'TRADE_NAME')
        .groupBy('id')
        .agg(f.collect_set('synonym').alias('_trade'))
    )

    other_synonyms = (
        synonyms.filter(f.col('syn_type') != 'TRADE_NAME').groupBy('id').agg(f.collect_set('synonym').alias('_syn'))
    )

    full = trade_names.join(other_synonyms, on='id', how='full_outer')

    return full.withColumn(
        'synonyms',
        f.array_sort(
            f.transform(f.coalesce(f.col('_syn'), f.array()), lambda c: _as_label_source(c, CHEMBL_SOURCE))
        ).cast(LABEL_SOURCE_SCHEMA),
    ).withColumn(
        'tradeNames',
        f.array_sort(
            f.transform(f.coalesce(f.col('_trade'), f.array()), lambda c: _as_label_source(c, CHEMBL_SOURCE))
        ).cast(LABEL_SOURCE_SCHEMA),
    ).drop('_syn', '_trade')


def _process_molecule_hierarchy(preprocessed_mols: DataFrame) -> DataFrame:
    """Group all child molecules by parent chembl_id.

    Args:
        preprocessed_mols: Preprocessed molecule DataFrame.

    Returns:
        DataFrame with id and childChemblIds columns.
    """
    return (
        preprocessed_mols
        .select('id', 'parentId')
        .filter(f.col('id') != f.col('parentId'))
        .filter(f.col('parentId').isNotNull())  # ty:ignore[missing-argument]
        .groupBy('parentId')
        .agg(f.collect_set('id').alias('childChemblIds'))
        .withColumnRenamed('parentId', 'id')
    )


def _process_molecule_cross_references(preprocessed_mols: DataFrame) -> DataFrame:
    """Group cross references for each molecule id.

    Args:
        preprocessed_mols: Preprocessed molecule DataFrame.

    Returns:
        DataFrame with id and crossReferences columns.
    """
    chembl_xrefs = _process_chembl_cross_references(preprocessed_mols)
    drugbank_xrefs = _process_singleton_cross_references(preprocessed_mols, 'drugbank_id', 'drugbank')

    # Merge cross reference maps
    merged = _merge_cross_reference_maps(chembl_xrefs, drugbank_xrefs)
    merged = merged.filter(f.col('xref').isNotNull()).withColumnRenamed('xref', 'crossReferences')  # ty:ignore[missing-argument]

    # Transform to array of structs format
    return (
        merged
        .select(f.col('id'), f.explode('crossReferences').alias('key', 'ids'))
        .withColumnRenamed('key', 'source')
        .groupBy('id')
        .agg(f.collect_set(f.struct(f.col('source'), f.col('ids'))).alias('crossReferences'))
    )


def _process_chembl_cross_references(preprocessed_mols: DataFrame) -> DataFrame:
    """Process ChEMBL cross references into a map structure.

    Args:
        preprocessed_mols: Preprocessed molecule DataFrame.

    Returns:
        DataFrame with id and xref map columns.
    """
    chembl_xr = (
        preprocessed_mols
        .select(
            f.col('id'),
            f.explode(
                f.arrays_zip(
                    f.col('cross_references.xref_id'),
                    f.col('cross_references.xref_src'),
                )
            ).alias('sources'),
        )
        .withColumn('ref_id', f.col('sources.xref_id'))
        .withColumn('ref_src', f.col('sources.xref_src'))
        .drop('sources')
    )

    # Group by id and source to create map
    return (
        chembl_xr
        .groupBy('id', 'ref_src')
        .agg(f.collect_list('ref_id').alias('ref_ids'))
        .groupBy('id')
        .agg(f.map_from_entries(f.collect_list(f.struct('ref_src', 'ref_ids'))).alias('xref'))
    )


def _process_singleton_cross_references(
    preprocessed_mols: DataFrame,
    reference_id_column: str,
    source: str,
) -> DataFrame:
    """Process singleton cross references (e.g., drugbank_id).

    Args:
        preprocessed_mols: Preprocessed molecule DataFrame.
        reference_id_column: Column name containing the reference ID.
        source: Name of the source for the cross reference.

    Returns:
        DataFrame with id and xref map columns.
    """
    return (
        preprocessed_mols
        .filter(f.col(reference_id_column).isNotNull())  # ty:ignore[missing-argument]
        .select(f.col('id'), f.col(reference_id_column).cast('string'))
        .groupBy('id')
        .agg(f.collect_set(reference_id_column).alias(reference_id_column))
        .withColumn('xref', f.create_map(f.lit(source), f.col(reference_id_column)))
        .drop(reference_id_column)
    )


def _merge_cross_reference_maps(ref1: DataFrame, ref2: DataFrame) -> DataFrame:
    """Merge two cross reference map DataFrames.

    Args:
        ref1: First DataFrame with id and xref columns.
        ref2: Second DataFrame with id and xref columns.

    Returns:
        Merged DataFrame with combined xref maps.
    """
    empty_map = f.create_map().cast(MapType(StringType(), ArrayType(StringType())))

    r1 = ref1.select(f.col('id'), f.coalesce(f.col('xref'), empty_map).alias('x'))
    r2 = ref2.select(f.col('id'), f.coalesce(f.col('xref'), empty_map).alias('y'))

    return (
        r1
        .join(r2, on='id', how='full_outer')
        .select(
            f.col('id'),
            f.coalesce(f.col('x'), empty_map).alias('x'),
            f.coalesce(f.col('y'), empty_map).alias('y'),
        )
        .withColumn('xref', f.map_concat(f.col('x'), f.col('y')))
        .drop('x', 'y')
    )
