"""Module to process Platform evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import log10

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql.window import Window

from pts.pyspark.common.session import Session
from pts.pyspark.common.utils import update_quality_flag


class EvidenceFlags(Enum):
    INVALID_DISEASE = 'Evidence has no valid disease'
    INVALID_TARGET = 'Evidence has no valid target'
    DUPLICATED = 'Evidence is duplicated'
    NO_VALID_SCORE = 'Evidence has no valid score'
    INVALID_BIOTYPE = 'Evidence has invalid biotype'


@dataclass
class Evidence:
    # Incoming evidence:
    evidence_df: DataFrame

    # Hardcoded values for evidence processing:
    QC_COLUMN: str = 'qualityControls'

    # List of columns that are used for generate evidence identifier:
    UNIQUE_FIELDS: list[str] = field(default_factory=list)
    DEFAULT_UNIQUE_FIELDS: list[str] = field(
        default_factory=lambda: [
            'targetId',
            'diseaseId',
            'datasourceId',
            'targetFromSourceId',
            'diseaseFromSourceMappedId',
        ]
    )

    # Default lenght of variant identifier to start hashing:
    VARIANT_HASH_LENGHT: int = 300

    # Columns we consider for dating evidence:
    EVIDENCE_DATE_COLUMNS: list[str] = field(
        default_factory=lambda: ['publicationDate', 'curationDate', 'studyStartDate']
    )

    def __post_init__(self: Evidence) -> None:
        """Initial process of the evidence."""
        # Initialise QC column:
        self.evidence_df = self.evidence_df.withColumn(self.QC_COLUMN, f.lit([]).cast(t.ArrayType(t.StringType())))

    def validate_target(self: Evidence, target_lut: DataFrame, invalid_biotypes: list[str] | None) -> Evidence:
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

        self.evidence_df = (
            self.evidence_df
            # Resolve target identifiers:
            .join(f.broadcast(target_lut), on='targetFromSourceId', how='leftouter')
            # Flag evidence without mapped targets:
            .withColumn(
                self.QC_COLUMN,
                update_quality_flag(f.col(self.QC_COLUMN), f.col('targetId').isNull(), EvidenceFlags.INVALID_TARGET),
            )
            # Flag evidence without target of invalid biotype:
            .withColumn(
                self.QC_COLUMN,
                update_quality_flag(
                    f.col(self.QC_COLUMN), f.col('targetId').isin(invalid_biotypes), EvidenceFlags.INVALID_TARGET
                ),
            )
            .drop('biotype')
        )

        return self

    def validate_diseases(self: Evidence, disease_lut: DataFrame) -> Evidence:
        """Validation of diseases.

        Args:
            disease_lut (DataFrame): processed disease look-up-table.

        Returns:
            Evidence: evidence with mapped targets, and flaggeed evidence without target.
        """
        self.evidence_df = (
            self.evidence_df
            # Resolve target identifiers:
            .join(disease_lut, on='diseaseFromSourceMappedId', how='leftouter')
            # Flag evidence without mapped targets:
            .withColumn(
                self.QC_COLUMN,
                update_quality_flag(f.col(self.QC_COLUMN), f.col('diseaseId').isNull(), EvidenceFlags.INVALID_DISEASE),
            )
        )

        return self

    def validate_uniqueness(self: Evidence) -> Evidence:
        """Validate uniqueness of evidence.

        Grouping evidence by evidence identifiers. If more than one evidence share the same ID all but one
        will be flagged.

        Returns:
            Evidence: where non-unique evidence is flagged.
        """
        self.evidence_df = (
            self.evidence_df.withColumn(
                'evidence_unique_rank', f.rank().over(Window.partitionBy('id').orderBy(f.rand()))
            )
            .withColumn(
                self.QC_COLUMN,
                update_quality_flag(
                    f.col(self.QC_COLUMN), f.col('evidence_unique_rank') != 1, EvidenceFlags.DUPLICATED
                ),
            )
            .drop('evidence_unique_rank')
        )
        return self

    def assign_evidence_identifier(self: Evidence) -> Evidence:
        """Adding unique identifier to each evidence based on source specific fields."""
        hash_columns = self.DEFAULT_UNIQUE_FIELDS + self.UNIQUE_FIELDS

        hash_expression = f.sha1(f.concat(*[f.col(col).cast(t.StringType()) for col in hash_columns]))

        self.evidence_df = self.evidence_df.withColumn('id', hash_expression)

        return self

    def hash_long_variant_identifiers(self: Evidence) -> Evidence:
        """Hash long variant identifier.

        If `variantId` column is present in the dataset, long ids need to be hashed.
        """
        if 'variantId' not in self.evidence_df.columns:
            return self

        self.evidence_df = self.evidence_df.withColumn(
            'variantId', self._hash_long_variant_ids(f.col('variantId'), self.VARIANT_HASH_LENGHT)
        )

        return self

    def calculate_evidence_score(self: Evidence, score_expression: str) -> Evidence:
        """Using the provided score expression, assign evidence score to evidence.

        Args:
            score_expression (str): valid spark expression as string.

        Returns:
            Evidence: with assigned score
        """
        self.evidence_df = (
            self.evidence_df
            # Calculate evidence score and make sure the type is consistent across different datasets:
            .withColumn('score', f.expr(score_expression).cast(t.DoubleType()))
            # Flag evidence invalid scores (missing, zero or negative, more than 1)
            .withColumn(
                self.QC_COLUMN,
                update_quality_flag(
                    f.col(self.QC_COLUMN),
                    f.when(
                        f.col('score').isNull() | (f.col('score') <= 0) | (f.col('score') > 1), f.lit(True)
                    ).otherwise(f.lit(False)),
                    EvidenceFlags.NO_VALID_SCORE,
                ),
            )
        )
        return self

    def resolve_publication_date(self: Evidence, publication_date_lut: DataFrame) -> Evidence:
        # Return self if publication column is not present in the dataset:
        if not {'literature', 'id'}.issubset(self.evidence_df.columns):
            return self

        # 1. Filter for MED, PPR, AGR and explode valid publication identifiers
        processed_publication_data = publication_date_lut.filter(f.col('source').isin('MED', 'PPR', 'AGR')).select(
            f.col('firstPublicationDate').alias('publicationDate'),
            f.explode(f.expr('filter(array(pmid, id, pmcid), x -> x is not null)')).alias('publicationId'),
        )

        # 2. Process evidence DataFrame and extract publication IDs
        evidence_with_pub_ids = (
            self.evidence_df.select(f.col('id'), f.explode(f.col('literature')).alias('publicationId'))
            .withColumn('publicationId', f.upper(f.trim(f.col('publicationId'))))
            .distinct()
        )

        # 3. Join evidence with publication mapping
        dated_evidence = (
            evidence_with_pub_ids.join(processed_publication_data, on='publicationId', how='inner')
            # For each evidence identifier we find the earliest publication date:
            .withColumn('rank', f.row_number().over(Window.partitionBy('id').orderBy(f.col('publicationDate').asc())))
            .filter(f.col('rank') == 1)
            .select('id', 'publicationDate')
        )

        # 4. Broadcast for efficiency and join back to main evidence
        dated_evidence_lut = f.broadcast(dated_evidence.orderBy(f.col('id').asc()))

        self.evidence_df = self.evidence_df.join(dated_evidence_lut, on='id', how='left_outer')

        return self

    def resolve_evidence_date(self: Evidence) -> Evidence:
        """Generate evidenceDate for each evidence based on a number of columns.

        Evidence date column has to be presented even if none of the eligible columns are present in the dataset.
        """
        # Get which columns do we have:
        dating_columns = [f.col(col) for col in self.EVIDENCE_DATE_COLUMNS if col in self.evidence_df.columns]

        # If no columns are there, we just return:
        if len(dating_columns) == 0:
            dating_columns = [f.lit(None).cast(t.StringType())]

        # Assign date:
        self.evidence_df = self.evidence_df.withColumn('evidenceDate', f.array_min(f.array(dating_columns)))
        return self

    def resolve_direction_of_effect(self: Evidence, mechanism_of_action: DataFrame) -> Evidence:
        raise NotImplementedError

    def save_evidence(self: Evidence, output_path: str, save_valid: bool = True) -> None:
        if save_valid:
            self.evidence_df.filter(f.size(self.QC_COLUMN) == 0).write.mode('overwrite').parquet(output_path)
        else:
            self.evidence_df.filter(f.size(self.QC_COLUMN) > 0).write.mode('overwrite').parquet(output_path)

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


def prepare_target_lut(target_index: DataFrame) -> DataFrame:
    """Prepare target index for target validation.

    Args:
        target_index (DataFrame): raw, unprocessed target index.

    Returns:
        DataFrame: dataframe with column `targetId` and `targetFromSourceId`.
    """
    return (
        target_index.select(
            f.col('id').alias('targetId'),
            'biotype',
            f.array_distinct(
                f.flatten(
                    f.array(
                        f.array(f.col('id')),
                        f.transform(f.col('proteinIds'), lambda protein: protein.id),
                        f.array(f.col('approvedSymbol')),
                    )
                )
            ).alias('targetFromSourceIds'),
        )
        .select('targetId', 'biotype', f.explode('targetFromSourceIds').alias('targetFromSourceId'))
        .orderBy(f.col('targetFromSourceId').asc())
        .repartition(f.col('targetFromSourceId'))
    )


def prepare_disease_lut(disease_index: DataFrame) -> DataFrame:
    """Prepare disease index for disease validation.

    Args:
        disease_index (DataFrame):raw, unprocessed disease index.

    Returns:
        DataFrame: dataframe with column `diseaseId` and `diseaseFromSourceMappedId`.
    """
    return (
        disease_index.select(
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


def linear_rescaling(
    value: float, in_range_min: float, in_range_max: float, out_range_min: float, out_range_max: float
) -> float:
    """Linearly rescaling a value across ranges.

    Args:
        value (float): value to rescale
        in_range_min (float): minimum of the starting range
        in_range_max (float): maximum of the starting range
        out_range_min (float): minimum of the resulting range
        out_range_max (float): maximum of the resulting range

    Returns:
        float: re-scaled value
    """
    delta1 = in_range_max - in_range_min
    delta2 = out_range_max - out_range_min

    if delta1 != 0.0:
        score = (delta2 * (value - in_range_min) / delta1) + out_range_min
    elif delta1 == 0.0 and delta2 == 0.0:
        score = value
    else:  # delta1 == 0.0 and delta2 != 0.0
        score = out_range_min

    # clamp to [out_range_min, out_range_max]
    return min(max(score, out_range_min), out_range_max)


def pvalue_linear_rescaling(
    pvalue: float,
    in_range_min: float = 1.0,
    in_range_max: float = 1e-10,
    out_range_min: float = 0.0,
    out_range_max: float = 1.0,
) -> float:
    """Convert p-value to log10 space and apply linear_rescaling.

    Args:
        pvalue (float): p-value to rescale
        in_range_min (float): minimum of the starting p-value range
        in_range_max (float): maximum of the starting p-value range
        out_range_min (float): minimum of the resulting range
        out_range_max (float): maximum of the resulting range

    Returns:
        float: re-scaled value
    """
    pvalue_log = log10(pvalue)
    in_min_log = log10(in_range_min)
    in_max_log = log10(in_range_max)

    return linear_rescaling(pvalue_log, in_min_log, in_max_log, out_range_min, out_range_max)


def generate_evidence(
    evidence_path,
    evidence_format,
    score_expression,
    unique_fields,
    invalid_biotypes,
    disease_path,
    target_path,
    literature_path,
    mechanism_of_action_path,
    output_success_path,
    otuput_failed_path,
) -> None:
    session = Session()

    # Registering UDFs in the spark session:
    session.spark.udf.register('linear_rescale', linear_rescaling)
    session.spark.udf.register('pvalue_linear_score', pvalue_linear_rescaling)

    # Reading input datasets:
    evidence_df = session.load_data(evidence_path, evidence_format)
    disease_lut = prepare_disease_lut(session.load_data(disease_path, 'parquet'))
    target_lut = prepare_target_lut(session.load_data(target_path, 'parquet'))
    # mechanism_of_action_df = session.load_data(mechanism_of_action_path, 'parquet')
    literature_mapping_lut = session.load_data(literature_path, 'json')

    print(target_lut.filter(f.col('biotype').isin(invalid_biotypes)).count())

    print(target_lut.filter(f.col('biotype').isin(invalid_biotypes)).select('targetId').distinct().count())

    # # Processing evidence:
    # processed_evidence = (
    #     Evidence(evidence_df=evidence_df, UNIQUE_FIELDS=unique_fields)
    #     # Hash long variant identifiers:
    #     .hash_long_variant_identifiers()
    #     # Validate diseases:
    #     .validate_diseases(disease_lut)
    #     # Validate targets:
    #     .validate_target(target_lut, invalid_biotypes)
    #     # Assing variant identifiers and flag duplicates:
    #     .assign_evidence_identifier()
    #     .validate_uniqueness()
    #     # Calculate evidence score - flag unscored:
    #     .calculate_evidence_score(score_expression)
    #     # Map literature to publication date:
    #     .resolve_publication_date(literature_mapping_lut)
    #     # Assing evidence date:
    #     .resolve_evidence_date()
    #     # # Resolve direction of effect:
    # )

    # processed_evidence.evidence_df.show(1, False, True)
    # processed_evidence.evidence_df.groupBy('qualityControls').count().show(truncate=False)

    # processed_evidence.save_evidence(output_success_path, True)
    # processed_evidence.save_evidence(otuput_failed_path, False)


if __name__ == '__main__':
    # # Datasource specific parameters GWAS_EVIDENCE:
    # evidence_path: str = '/Users/dsuveges/repositories/pts/gwas_evidence'
    # evidence_format: str = 'parquet'
    # score_expression: str = 'resourceScore'
    # unique_fields: list[str] = ['studyLocusId']
    # invalid_biotypes: list[str] = []
    # output_success_path: str = '/Users/dsuveges/repositories/pts/gwas_evidence_passed'
    # otuput_failed_path: str = '/Users/dsuveges/repositories/pts/gwas_evidence_failed'

    # Datasource specific parameters Expression atlas:
    evidence_path: str = '/Users/dsuveges/project_data/25.09/input/evidence/atlas.json.bz2'
    evidence_format: str = 'json'
    score_expression: str = 'array_min(array(1.0, pvalue_linear_score(resourceScore) * (abs(log2FoldChangeValue) / 10) * (log2FoldChangePercentileRank / 100)))'
    unique_fields: list[str] = ['contrast', 'studyId']
    invalid_biotypes: list[str] = [
        'IG_C_pseudogene',
        'IG_J_pseudogene',
        'IG_pseudogene',
        'IG_V_pseudogene',
        'polymorphic_pseudogene',
        'processed_pseudogene',
        'pseudogene',
        'rRNA',
        'rRNA_pseudogene',
        'snoRNA',
        'snRNA',
        'transcribed_processed_pseudogene',
        'transcribed_unitary_pseudogene',
        'transcribed_unprocessed_pseudogene',
        'TR_J_pseudogene',
        'TR_V_pseudogene',
        'unitary_pseudogene',
        'unprocessed_pseudogene',
    ]
    output_success_path: str = '/Users/dsuveges/repositories/pts/atlas_passed'
    otuput_failed_path: str = '/Users/dsuveges/repositories/pts/atlas_failed'

    # Shared evidence configuration:
    disease_path: str = '/Users/dsuveges/project_data/25.09/output/disease'
    target_path: str = '/Users/dsuveges/project_data/25.09/output/target'
    literature_path: str = '/Users/dsuveges/repositories/pts/literature_export'
    mechanism_of_action_path: str = ''

    # Process evidence:
    generate_evidence(
        evidence_path,
        evidence_format,
        score_expression,
        unique_fields,
        invalid_biotypes,
        disease_path,
        target_path,
        literature_path,
        mechanism_of_action_path,
        output_success_path,
        otuput_failed_path,
    )
