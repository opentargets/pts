"""Metric type registry: MetricType enum and load_metric dispatcher."""
from __future__ import annotations

from enum import StrEnum
from importlib import import_module
from typing import Any

from pts.metrics.base import Metric
from pts.metrics.count import CountMetric, DistinctCountMetric
from pts.metrics.grouped import GroupedCountExplodeMetric, GroupedCountMetric, GroupedSumMetric


class MetricType(StrEnum):
    """Supported metric kinds that can be instantiated from config."""

    count = 'count'
    distinct_count = 'distinct_count'
    grouped_count = 'grouped_count'
    grouped_count_explode = 'grouped_count_explode'
    grouped_sum = 'grouped_sum'
    custom = 'custom'


_IMPLEMENTERS: dict[MetricType, type[Metric]] = {
    MetricType.count: CountMetric,
    MetricType.distinct_count: DistinctCountMetric,
    MetricType.grouped_count: GroupedCountMetric,
    MetricType.grouped_count_explode: GroupedCountExplodeMetric,
    MetricType.grouped_sum: GroupedSumMetric,
}


_BUILTIN_CLASSES: frozenset[type[Metric]] = frozenset(_IMPLEMENTERS.values())


def load_metric(cfg: dict[str, Any]) -> Metric:
    """Instantiate a Metric from a config dict using MetricType as the dispatcher key."""
    cfg = dict(cfg)
    raw_type = cfg.pop('type', None)

    try:
        metric_type = MetricType(raw_type)
    except ValueError:
        raise ValueError(f"unknown metric type '{raw_type}'")

    if metric_type is MetricType.custom:
        class_path = cfg.pop('class')
        module_path, cls_name = class_path.rsplit('.', 1)
        cls = getattr(import_module(module_path), cls_name)
        return cls(**cfg)

    return _IMPLEMENTERS[metric_type](**cfg)


def metric_to_dict(m: Metric) -> dict[str, Any]:
    """Serialize a Metric back to a loadable config dict (inverse of load_metric).

    Needed so CollectMetricsSpec/TransformSpec.model_dump() round-trips correctly
    through otter's double-parse in TaskRegistry.build().
    """
    data = m.model_dump()
    if type(m) in _BUILTIN_CLASSES:
        return data  # built-in: 'type' literal field already present in data
    # custom metric: inject type and class path
    cls = type(m)
    data['type'] = 'custom'
    data['class'] = f'{cls.__module__}.{cls.__qualname__}'
    return data
