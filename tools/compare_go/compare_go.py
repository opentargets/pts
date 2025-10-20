from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger
from pyspark.sql.functions import col

from pts.pyspark.common.session import Session


def read_go_df(spark, path: str, drop_obsolete: bool):
    df = spark.read.load(path, format='parquet')
    if drop_obsolete and 'obsolete' in df.columns:
        df = df.filter(~col('obsolete'))
    return df


def download_gcs_folder_to_local(gcs_uri: str, local_dir: Path) -> str:
    """Download a folder or file from GCS to a local directory using Spark.

    Returns the local path string suitable for subsequent Spark reads.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    spark = Session(app_name='download_gcs').spark
    df = spark.read.load(gcs_uri, format='parquet')
    local_path = local_dir.as_posix()
    df.write.mode('overwrite').parquet(local_path)
    logger.info(f'downloaded {gcs_uri} to {local_path}')
    spark.stop()
    return local_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare GO outputs: row counts and unique id counts.')
    parser.add_argument('--ref', required=True, help='Reference path (gs:// or local)')
    parser.add_argument('--cand', required=True, help='Candidate path (local Parquet path)')
    parser.add_argument('--drop-obsolete', action='store_true', help='Filter out rows where obsolete == true')
    parser.add_argument(
        '--download-ref-local', action='store_true', help='Download gs:// ref to local folder before comparing'
    )
    parser.add_argument(
        '--local-dir', default='work/local_go_ref', help='Local folder to store downloaded ref if enabled'
    )
    parser.add_argument(
        '--report-out', default=None, help='Optional path to write JSON report (e.g., work/reports/go_compare.json)'
    )
    args = parser.parse_args()

    ref_path = args.ref
    if args.download_ref_local and ref_path.startswith('gs://'):
        ref_path = download_gcs_folder_to_local(ref_path, Path(args.local_dir))

    session = Session(app_name='compare_go')
    spark = session.spark

    logger.info(f'Loading reference from: {ref_path}')
    ref = read_go_df(spark, ref_path, args.drop_obsolete)
    logger.info(f'Loading candidate from: {args.cand}')
    cand = read_go_df(spark, args.cand, args.drop_obsolete)

    ref_rows = ref.count()
    ref_unique = ref.select('id').distinct().count()
    cand_rows = cand.count()
    cand_unique = cand.select('id').distinct().count()

    # Find difference in IDs
    ref_ids = ref.select('id').distinct()
    cand_ids = cand.select('id').distinct()
    only_in_ref = ref_ids.join(cand_ids, 'id', 'left_anti').select('id').rdd.flatMap(lambda x: x).collect()
    only_in_cand = cand_ids.join(ref_ids, 'id', 'left_anti').select('id').rdd.flatMap(lambda x: x).collect()

    label = '(excluding obsolete)' if args.drop_obsolete else '(including obsolete)'
    logger.info(f'Reference rows {label}: {ref_rows}')
    logger.info(f'Reference unique ids {label}: {ref_unique}')
    logger.info(f'Candidate rows {label}: {cand_rows}')
    logger.info(f'Candidate unique ids {label}: {cand_unique}')
    logger.info(f'Delta rows: {cand_rows - ref_rows}')
    logger.info(f'Delta unique ids: {cand_unique - ref_unique}')
    logger.info(f'IDs only in reference: {len(only_in_ref)}')
    logger.info(f'IDs only in candidate: {len(only_in_cand)}')

    if args.report_out:
        report = {
            'drop_obsolete': bool(args.drop_obsolete),
            'reference': {
                'path': ref_path,
                'rows': ref_rows,
                'unique_ids': ref_unique,
            },
            'candidate': {
                'path': args.cand,
                'rows': cand_rows,
                'unique_ids': cand_unique,
            },
            'delta': {
                'rows': cand_rows - ref_rows,
                'unique_ids': cand_unique - ref_unique,
            },
            'differences': {
                'only_in_reference': sorted(only_in_ref),
                'only_in_candidate': sorted(only_in_cand),
                'only_in_reference_count': len(only_in_ref),
                'only_in_candidate_count': len(only_in_cand),
            },
        }
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open('w', encoding='utf-8') as fh:
            json.dump(report, fh, indent=2)
        logger.info(f'Report written to: {report_path}')

    session.stop()


if __name__ == '__main__':
    main()
