"""Tests for literature_publication_match step."""


class TestEpmcReadPath:
    """Test the EPMC jsonl glob path builder."""

    def test_with_date_prefix(self):
        from pts.pyspark.literature_publication_match import _epmc_read_path

        assert (
            _epmc_read_path('gs://otar025-epmc/ml02', 'fulltext', '2026_03')
            == 'gs://otar025-epmc/ml02/fulltext/2026_03*/**/*.jsonl'
        )

    def test_without_date_prefix(self):
        from pts.pyspark.literature_publication_match import _epmc_read_path

        assert (
            _epmc_read_path('gs://otar025-epmc/ml02', 'abstract', None)
            == 'gs://otar025-epmc/ml02/abstract/**/*.jsonl'
        )

    def test_strips_trailing_slash(self):
        from pts.pyspark.literature_publication_match import _epmc_read_path

        assert (
            _epmc_read_path('gs://otar025-epmc/ml02/', 'fulltext', None)
            == 'gs://otar025-epmc/ml02/fulltext/**/*.jsonl'
        )

    def test_empty_date_prefix_treated_as_none(self):
        from pts.pyspark.literature_publication_match import _epmc_read_path

        assert (
            _epmc_read_path('gs://otar025-epmc/ml02', 'abstract', '')
            == 'gs://otar025-epmc/ml02/abstract/**/*.jsonl'
        )


class TestMaybeRepartition:
    """Test the optional repartition helper."""

    def test_repartitions_when_count_given(self, spark):
        from pts.pyspark.literature_publication_match import _maybe_repartition

        df = spark.range(100)
        result = _maybe_repartition(df, 4)
        assert result.rdd.getNumPartitions() == 4

    def test_returns_df_unchanged_when_none(self, spark):
        from pts.pyspark.literature_publication_match import _maybe_repartition

        df = spark.range(100)
        result = _maybe_repartition(df, None)
        assert result is df

    def test_returns_df_unchanged_when_zero(self, spark):
        from pts.pyspark.literature_publication_match import _maybe_repartition

        df = spark.range(100)
        result = _maybe_repartition(df, 0)
        assert result is df
