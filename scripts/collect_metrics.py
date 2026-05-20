#!/usr/bin/env python3
"""Run metric collection on all datasets configured in metrics.yaml.

Usage:
    uv run python scripts/collect_metrics.py [config] [--release TAG] [--run ID]

Defaults:
    config   = metrics.yaml (in project root)
    release  = value from config scratchpad
    run      = value from config scratchpad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description='Collect metrics from output datasets')
    parser.add_argument('config', nargs='?', default='metrics.yaml', help='Path to metrics.yaml')
    parser.add_argument('--release', help='Release tag (overrides scratchpad)')
    parser.add_argument('--run', dest='run_id', help='Run identifier (overrides scratchpad)')
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f'Config not found: {config_path}', file=sys.stderr)
        sys.exit(1)

    with config_path.open() as f:
        config = yaml.safe_load(f)

    scratchpad = config.get('scratchpad', {})
    release = args.release or str(scratchpad.get('release', ''))
    run_id = args.run_id or str(scratchpad.get('run', ''))
    data_root = Path(config['data_root'])
    metrics_root = Path(config['metrics_root'])

    # import here so errors are reported after arg parsing
    from pts.metrics.runner import MetricRunner
    from pts.tasks.transform import _load_metric

    print(f'release={release}  run={run_id}')
    print(f'data_root={data_root}')
    print(f'metrics_root={metrics_root}')
    print()

    runner = MetricRunner()
    errors: list[tuple[str, Exception]] = []

    for dataset_name, ds_config in config.get('datasets', {}).items():
        raw_metrics = ds_config.get('metrics', [])
        if not raw_metrics:
            continue
        try:
            metrics = [_load_metric(m) for m in raw_metrics]
        except Exception as e:
            print(f'[SKIP] {dataset_name}: bad metric config — {e}')
            errors.append((dataset_name, e))
            continue

        dataset_path = data_root / dataset_name
        if not dataset_path.exists():
            print(f'[SKIP] {dataset_name}: path not found ({dataset_path})')
            continue

        print(f'[RUN]  {dataset_name} ({len(metrics)} metrics) ...', end=' ', flush=True)
        try:
            runner.run(
                metrics=metrics,
                dataset_path=dataset_path,
                metrics_root=metrics_root,
                dataset_name=dataset_name,
                release=release,
                run=run_id,
            )
            written = sorted((metrics_root / dataset_name).glob('*.json'))
            print(f'OK  → {len(written)} JSON files')
        except Exception as e:
            print(f'ERROR')
            print(f'       {type(e).__name__}: {e}')
            errors.append((dataset_name, e))

    print()
    if errors:
        print(f'Finished with {len(errors)} error(s):')
        for name, err in errors:
            print(f'  {name}: {err}')
        sys.exit(1)
    else:
        print('All metrics collected successfully.')


if __name__ == '__main__':
    main()
