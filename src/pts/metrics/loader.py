"""Metric type registry: MetricType enum and load_metric dispatcher.

Registering a new built-in metric type
----------------------------------------
Add a member to :class:`MetricType` whose name matches the ``type:
Literal['my_type']`` discriminator on the :class:`~pts.metrics.base.Metric`
subclass and whose value is the implementer class::

    class MetricType(Enum):
        ...
        my_type = MyMetric

No separate mapping dict is needed — the enum value IS the implementer.

For one-off metrics that do not need registration, use ``type: custom`` in the
YAML config with a ``class`` dotted-path instead — see
:mod:`pts.metrics.base` for details.
"""
from __future__ import annotations

from enum import Enum
from importlib import import_module
from typing import Any

from pts.metrics.base import Metric
from pts.metrics.count import CountMetric, DistinctCountMetric
from pts.metrics.grouped import GroupedCountExplodeMetric, GroupedCountMetric, GroupedSumMetric


class MetricType(Enum):
    """Supported metric kinds recognised by :func:`load_metric`.

    Each member's **name** matches the ``type`` field used in YAML config;
    its **value** is the implementer :class:`~pts.metrics.base.Metric` class
    (or ``None`` for ``custom``, which uses a dotted class path instead).
    """

    count = CountMetric
    distinct_count = DistinctCountMetric
    grouped_count = GroupedCountMetric
    grouped_count_explode = GroupedCountExplodeMetric
    grouped_sum = GroupedSumMetric
    custom = None


_BUILTIN_CLASSES: frozenset[type[Metric]] = frozenset(
    m.value for m in MetricType if m.value is not None
)


def load_metric(cfg: dict[str, Any]) -> Metric:
    """Instantiate a Metric from a config dict using MetricType as the dispatcher key."""
    cfg = dict(cfg)
    raw_type = cfg.pop('type', None)

    try:
        metric_type = MetricType[raw_type]
    except KeyError:
        raise ValueError(f"unknown metric type '{raw_type}'")

    match metric_type:
        case MetricType.custom:
            class_path = cfg.pop('class')
            module_path, cls_name = class_path.rsplit('.', 1)
            cls = getattr(import_module(module_path), cls_name)
            return cls(**cfg)
        case _:
            return metric_type.value(**cfg)


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
