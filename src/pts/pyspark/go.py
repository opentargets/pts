from __future__ import annotations

from loguru import logger
from pyspark.sql import DataFrame, Row, SparkSession

from pts.pyspark.common.session import Session
from pts.schemas.go import go_schema

try:
    import obonet
except Exception:
    obonet = None


def _rows_from_obo(graph) -> list[Row]:
    rows: list[Row] = []
    obsolete_count = 0
    for go_id, data in graph.nodes(data=True):
        if not go_id.startswith('GO:'):
            continue
        label = data.get('name')
        namespace = data.get('namespace')
        alt_ids = data.get('alt_id') or []
        # robust parse: obonet stores 'is_obsolete' as 'true' (str) when present
        obs_value = data.get('is_obsolete', False)
        if isinstance(obs_value, str):
            is_obsolete = obs_value.lower() == 'true'
        else:
            is_obsolete = bool(obs_value)
        if is_obsolete:
            obsolete_count += 1
            if obsolete_count <= 3:
                logger.debug(
                    f'obsolete term found: {go_id}, raw value: {obs_value!r}, type: {type(obs_value).__name__}'
                )
        is_a = data.get('is_a') or []
        # relationships may come as strings 'rel TARGET' or tuples ('rel', 'TARGET')
        relationships = data.get('relationship') or []
        part_of: list[str] = []
        regulates: list[str] = []
        negatively_regulates: list[str] = []
        positively_regulates: list[str] = []

        for rel in relationships:
            if isinstance(rel, tuple) and len(rel) >= 2:
                key, target = rel[0], rel[1]
            elif isinstance(rel, str) and ' ' in rel:
                key, target = rel.split(' ', 1)
            else:
                continue

            if key == 'part_of':
                part_of.append(target)
            elif key == 'regulates':
                regulates.append(target)
            elif key == 'negatively_regulates':
                negatively_regulates.append(target)
            elif key == 'positively_regulates':
                positively_regulates.append(target)

        rows.append(
            Row(
                id=go_id,
                label=label,
                namespace=namespace,
                alt_ids=alt_ids,
                is_a=is_a,
                part_of=part_of,
                regulates=regulates,
                negatively_regulates=negatively_regulates,
                positively_regulates=positively_regulates,
                obsolete=is_obsolete,
            )
        )
    logger.info(f'parsed {len(rows)} GO terms, {obsolete_count} obsolete')
    return rows


def _parse_go_obo(path: str, spark: SparkSession) -> DataFrame:
    if obonet is None:
        raise RuntimeError('Missing dependency obonet. Add obonet and networkx to project dependencies.')
    logger.debug(f'parsing GO OBO from: {path}')

    # Use smart_open for gs:// URLs to stream from GCS to driver memory
    if path.startswith('gs://'):
        from smart_open import open as smart_open

        logger.debug('detected GCS path, using smart_open for streaming')
        with smart_open(path, 'r') as fh:
            graph = obonet.read_obo(fh, ignore_obsolete=False)
    else:
        graph = obonet.read_obo(path, ignore_obsolete=False)

    rows = _rows_from_obo(graph)
    return spark.createDataFrame(rows, schema=go_schema)


def go(
    source: str | dict[str, str],
    destination: str | dict[str, str],
    properties: dict[str, str] | None,
) -> None:
    # Unify IO parameters
    src_path = source['go'] if isinstance(source, dict) else source
    dst_path = destination['go'] if isinstance(destination, dict) else destination

    session = Session(app_name='go', properties=properties)
    spark = session.spark
    logger.info('Executing Gene Ontology step (PySpark).')

    go_df = _parse_go_obo(src_path, spark)

    logger.debug('writing Gene Ontology outputs')
    go_df.write.mode('overwrite').parquet(dst_path)

    logger.info('Gene Ontology step complete')
    session.stop()
