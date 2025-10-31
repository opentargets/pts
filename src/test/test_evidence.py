"""Tests evidence logic."""

from __future__ import annotations

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as f

from pts.pyspark.evidence import Evidence


class TestEvidence:
    """Testing suite for the Evidence dataset."""

    EVIDENCE_DATASET = [
        ('t1', 'd1', 0.3),
        ('t3', 'd1', -0.1),
        ('t2', 'd1', 12.0),
    ]

    @pytest.fixture(autouse=True)
    def _setup(self: TestEvidence, spark: SparkSession) -> None:
        """Setting up input datasets."""
        self.evidence = Evidence(
            spark.createDataFrame(
                self.EVIDENCE_DATASET,
                'targetFromSourceId STRING, diseaseFromSourceMappedId STRING, resourceScore FLOAT',
            ).withColumns({'datasourceId': f.lit('ds1'), 'datatypeId': f.lit('dt1')})
        )

    def test_evidence__init__type(self: TestEvidence) -> None:
        """Test if the evidence object is the right type."""
        assert isinstance(self.evidence, Evidence)

    def test_evidence__init__size(self: TestEvidence) -> None:
        """Test if the evidence object has the right number of rows."""
        assert self.evidence.evidence_df.count() == len(self.EVIDENCE_DATASET)

    def test_evidence__init__qc_column(self: TestEvidence) -> None:
        """Test if the evidence object has the right number of rows."""
        assert Evidence.QC_COLUMN in self.evidence.evidence_df.columns

    def test_evidence_score_flagging(self: TestEvidence) -> None:
        """Testing if scoring works as expected."""
        self.evidence.calculate_evidence_score('resourceScore / 2')
        self.evidence.evidence_df.show()

    def test_evidence_score_flagging(self: TestEvidence) -> None:
        """Testing if scoring works as expected."""
        self.evidence.calculate_evidence_score('resourceScore / 2')
        self.evidence.evidence_df.show()
