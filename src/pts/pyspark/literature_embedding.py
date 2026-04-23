"""Literature embedding model training via Word2Vec.

Ported from Embedding.scala in platform-etl-backend.
Trains a Word2Vec model on entity co-occurrence patterns from literature
matches. Entities are grouped by publication and section, then permuted
to create training sentences.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pyspark.ml.feature import Word2Vec
from pyspark.sql import DataFrame
from pyspark.sql import functions as f
from pyspark.sql.window import Window

from pts.pyspark.common.session import Session

# Section importance ranks (same as literature_entity_lut)
_SECTION_RANKS = [
    {'section': 'title', 'rank': 1, 'weight': 1.0},
    {'section': 'abstract', 'rank': 1, 'weight': 0.8},
    {'section': 'concl', 'rank': 1, 'weight': 0.7},
    {'section': 'results', 'rank': 2, 'weight': 0.6},
    {'section': 'discuss', 'rank': 2, 'weight': 0.5},
    {'section': 'methods', 'rank': 3, 'weight': 0.3},
    {'section': 'other', 'rank': 4, 'weight': 0.1},
]

# Word2Vec model configuration (from Scala reference.conf)
_W2V_WINDOW_SIZE = 10
_W2V_NUM_PARTITIONS = 16
_W2V_MAX_ITER = 3
_W2V_MIN_COUNT = 1
_W2V_STEP_SIZE = 0.02


def _filter_matches(matches: DataFrame) -> DataFrame:
    """Filter matches to mapped entities of types DS, GP, CD."""
    valid_types = ['DS', 'GP', 'CD']
    return matches.filter(
        (f.col('isMapped') == True)  # noqa: E712
        & f.col('type').isin(valid_types)
    )


def _regroup_matches(matches: DataFrame) -> DataFrame:
    """Regroup matches by publication and ranked section for training.

    Groups matched entities by (pmid, section rank), collects keyword sets,
    then creates training permutations by combining per-section keyword lists
    with an overall keyword list.
    """
    spark = matches.sparkSession

    section_rank_table = f.broadcast(spark.createDataFrame(_SECTION_RANKS).orderBy(f.col('rank').asc()))

    w_per_section = Window.partitionBy('pmid', 'rank')

    return (
        matches
        .join(section_rank_table, on='section')
        .withColumn('keys', f.collect_set(f.col('keywordId')).over(w_per_section))
        .dropDuplicates(['pmid', 'rank'])
        .groupBy('pmid')
        .agg(f.collect_list(f.col('keys')).alias('keys'))
        .withColumn('overall', f.flatten(f.col('keys')))
        .withColumn('all', f.concat(f.col('keys'), f.array(f.col('overall'))))
        .withColumn('terms', f.explode(f.col('all')))
        .select('pmid', 'terms')
    )


def _train_word2vec(df: DataFrame) -> Any:
    """Train Word2Vec model on the training DataFrame."""
    w2v = (
        Word2Vec()
        .setWindowSize(_W2V_WINDOW_SIZE)
        .setNumPartitions(_W2V_NUM_PARTITIONS)
        .setMaxIter(_W2V_MAX_ITER)
        .setMinCount(_W2V_MIN_COUNT)
        .setStepSize(_W2V_STEP_SIZE)
        .setInputCol('terms')
        .setOutputCol('prediction')
    )
    return w2v.fit(df)


def literature_embedding(
    source: dict[str, str] | str,
    destination: dict[str, str] | str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Train Word2Vec embedding model on literature entity co-occurrences."""
    spark = Session(app_name='literature_embedding', properties=properties).spark

    logger.info('Reading literature matches')
    matches = spark.read.parquet(source['matches'])

    logger.info('Filtering and regrouping matches')
    filtered = _filter_matches(matches)
    training = _regroup_matches(filtered)
    training.persist()

    logger.info('Training Word2Vec model')
    model = _train_word2vec(training)

    dest = destination['model'] if isinstance(destination, dict) else destination
    logger.info(f'Saving Word2Vec model to {dest}')
    model.save(dest)
