from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import polars as pl
from loguru import logger


def _resolve_parquet_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob('*.parquet'))
        if files:
            return files
        # fallback to recursive search if needed
        return sorted(path.rglob('*.parquet'))
    raise FileNotFoundError(f'Path not found: {path}')


def _load_df(parquet_path: Path) -> pl.DataFrame:
    files = _resolve_parquet_files(parquet_path)
    if not files:
        raise FileNotFoundError(f'No parquet files found under: {parquet_path}')
    # polars can read a list of files and concatenate them
    return pl.read_parquet([str(f) for f in files])


def _schema_as_dict(df: pl.DataFrame) -> list[dict[str, str]]:
    return [
        {
            'name': name,
            'dtype': str(dtype),
        }
        for name, dtype in df.schema.items()
    ]


def _has_nested(df: pl.DataFrame) -> bool:
    return any('List' in str(dtype) or 'Struct' in str(dtype) for dtype in df.schema.values())


def main() -> None:
    parser = argparse.ArgumentParser(description='Inspect GO parquet output, print schema and sample rows.')
    parser.add_argument('-p', '--path', default='work/output/go/go.parquet', help='Path to parquet folder or file')
    parser.add_argument('-n', '--limit', type=int, default=5, help='Number of sample rows to display/save')
    parser.add_argument('--schema-out', type=str, default=None, help='Optional path to write schema JSON')
    parser.add_argument('--sample-out', type=str, default=None, help='Optional path to write sample (csv/json)')
    args = parser.parse_args()

    parquet_path = Path(args.path)
    df = _load_df(parquet_path)

    # Print high-level info
    logger.info(f'Rows: {df.height}')
    logger.info(f'Columns: {df.width}')

    # Print schema
    logger.info('Schema:')
    for name, dtype in df.schema.items():
        logger.info(f'- {name}: {dtype}')

    # Print sample rows
    logger.info(f'Sample (first {args.limit} rows):')
    head_df = df.head(args.limit)
    for row in head_df.to_dicts():
        logger.info(json.dumps(row, default=str))

    # Optionally write schema
    if args.schema_out:
        schema_path = Path(args.schema_out)
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        with schema_path.open('w', encoding='utf-8') as fh:
            json.dump(_schema_as_dict(df), fh, indent=2)
        logger.info(f'Schema written to: {schema_path}')

    # Optionally write sample
    if args.sample_out:
        sample_path = Path(args.sample_out)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        head_df = df.head(args.limit)
        ext = sample_path.suffix.lower()
        if ext == '.csv':
            # Robust CSV writer that JSON-encodes nested values
            rows = head_df.to_dicts()
            fieldnames = list(rows[0].keys()) if rows else []
            with sample_path.open('w', newline='', encoding='utf-8') as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                if fieldnames:
                    writer.writeheader()
                for row in rows:
                    safe_row: dict[str, str] = {}
                    for k, v in row.items():
                        if isinstance(v, (list, dict)):
                            safe_row[k] = json.dumps(v, ensure_ascii=False)
                        elif isinstance(v, (str, int, float, bool)) or v is None:
                            safe_row[k] = '' if v is None else str(v)
                        else:
                            # fallback for any other types
                            safe_row[k] = json.dumps(v, default=str, ensure_ascii=False)
                    writer.writerow(safe_row)
        elif ext in ('.json', '.jsonl', '.ndjson'):
            # write line-delimited JSON for readability
            head_df.write_ndjson(sample_path)
        else:
            # default to ndjson if unknown extension (supports nested data)
            head_df.write_ndjson(sample_path)
        logger.info(f'Sample written to: {sample_path}')


if __name__ == '__main__':
    main()
