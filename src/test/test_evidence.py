"""Tests evidence logic."""

from __future__ import annotations

import pytest
from numpy import isin
from pyspark.sql import SparkSession
from pyspark.sql import functions as f

from pts.pyspark.evidence import Evidence, EvidenceFlags


class TestEvidence:
    """Testing suite for the Evidence dataset."""

    EVIDENCE_DATASET = [
        ('t1', 'd1', 0.3),
        ('t1', 'd1', 2.0),
        ('t3', 'd1', -0.1),
        ('t2', 'd1', 12.0),
        ('t2', 'd1', None),
        ('missing_target', 'd1', 0.3),
    ]

    TARGET_LUT_DATA = [
        ('t1', 't1', 'bt1'),
        ('t1', 't2', 'bt1'),
        ('t4', 't3', 'bt2'),
        ('t4', 't4', 'bt2'),
        ('t5', 't5', 'bt3'),
    ]

    INVALID_BIOTYPES = ['bt2', 'bt3']

    @pytest.fixture(autouse=True)
    def _setup(self: TestEvidence, spark: SparkSession) -> None:
        """Setting up input datasets."""
        self.evidence = Evidence(
            spark.createDataFrame(
                self.EVIDENCE_DATASET,
                'targetFromSourceId STRING, diseaseFromSourceMappedId STRING, resourceScore FLOAT',
            ).withColumns({'datasourceId': f.lit('ds1'), 'datatypeId': f.lit('dt1')})
        )

        # Generate target look up table:
        self.target_lut = spark.createDataFrame(
            self.TARGET_LUT_DATA, 'targetId STRING, targetFromSourceId STRING, biotype STRING'
        )

    @pytest.fixture(autouse=True)
    def scored_evidence(self: TestEvidence) -> Evidence:
        """Calculate score for evidence."""
        return self.evidence.calculate_evidence_score('resourceScore / 2')

    # @pytest.fixture(autouse=True)
    # def id_evidence(self: TestEvidence) -> Evidence:
    #     """Generating identifier for evidence."""
    #     return (
    #         self.evidence.validate_diseases(self.disease_lut)
    #         .validate_target(self.target_lut)
    #         .assign_evidence_identifier([])
    #     )

    @pytest.fixture(autouse=True)
    def target_validated(self: TestEvidence) -> Evidence:
        """Validate targets for evidence."""
        return self.evidence.validate_target(self.target_lut, self.INVALID_BIOTYPES)

    # Test constructor:
    def test__init__type(self: TestEvidence) -> None:
        """Test if the evidence object is the right type."""
        assert isinstance(self.evidence, Evidence)

    def test__init__size(self: TestEvidence) -> None:
        """Test if the evidence object has the right number of rows."""
        assert self.evidence.evidence_df.count() == len(self.EVIDENCE_DATASET)

    def test__init__qc_column(self: TestEvidence) -> None:
        """Test if the evidence object has the right number of rows."""
        assert Evidence.QC_COLUMN in self.evidence.evidence_df.columns

    # Test scoring:
    def test_score__type(self: TestEvidence, scored_evidence: Evidence) -> None:
        """Testing if the scoring results in the right data type."""
        assert isinstance(scored_evidence, Evidence)

    def test_score__correct_score(self: TestEvidence, scored_evidence: Evidence) -> None:
        """Testing if the scoring results in the expected scores."""
        assert scored_evidence.evidence_df.filter(f.col('score') * 2 != f.col('resourceScore')).count() == 0

    def test_score__flagged(self: TestEvidence, scored_evidence: Evidence) -> None:
        """Testing if the scoring flags the right evidence."""
        evidence_df = scored_evidence.evidence_df

        flagged_count = evidence_df.filter(f.array_contains(Evidence.QC_COLUMN, EvidenceFlags.NO_VALID_SCORE)).count()
        bad_scores_count = evidence_df.filter(
            f.col('score').isNull() | (f.col('score') <= 0) | (f.col('score') > 1)
        ).count()

        assert flagged_count == bad_scores_count

    def test_score__valid(self: TestEvidence, scored_evidence: Evidence) -> None:
        """Testing if the scoring leaves the right evidence unflagged."""
        evidence_df = scored_evidence.evidence_df

        unflagged_count = evidence_df.filter(f.size(Evidence.QC_COLUMN) == 0).count()
        good_scores_count = evidence_df.filter((f.col('score') > 0) & (f.col('score') <= 1)).count()

        assert unflagged_count == good_scores_count

    # Test target validation:
    def test_target_validation__type(self: TestEvidence, target_validated: Evidence) -> None:
        """Testing if the validation returns the right type."""
        assert isinstance(target_validated, Evidence)

    def test_target_validation__new_column(self: TestEvidence, target_validated: Evidence) -> None:
        """Testing if the validation adds target id column."""
        assert 'targetId' in target_validated.evidence_df.columns

    def test_target_validation__flagged_missing_target(self: TestEvidence, target_validated: Evidence) -> None:
        """Testing if the validation flags evidence where no target is present."""
        problematic_rows = target_validated.evidence_df.withColumn(
            'problematic',
            (f.col('targetId').isNull() & ~f.array_contains(Evidence.QC_COLUMN, EvidenceFlags.INVALID_TARGET))
            | (f.col('targetId').isNotNull() & f.array_contains(Evidence.QC_COLUMN, EvidenceFlags.INVALID_TARGET)),
        ).filter(f.col('problematic'))

        assert problematic_rows.count() == 0

    def test_target_validation__flagged_invalid_biotype(self: TestEvidence, target_validated: Evidence) -> None:
        """Testing if the validation flags evidence where no target is present."""
        invalid_targets = list({t for (t, _, b) in self.TARGET_LUT_DATA if b in self.INVALID_BIOTYPES})

        # Making sure all these genes are flagged, but nothing else.
        problematic_rows = target_validated.evidence_df.withColumn(
            'problematic',
            (
                f.col('targetId').isin(invalid_targets)
                & ~f.array_contains(Evidence.QC_COLUMN, EvidenceFlags.INVALID_BIOTYPE)
            )
            | (
                ~f.col('targetId').isin(invalid_targets)
                & f.array_contains(Evidence.QC_COLUMN, EvidenceFlags.INVALID_BIOTYPE)
            ),
        ).filter(f.col('problematic'))

        assert problematic_rows.count() == 0
