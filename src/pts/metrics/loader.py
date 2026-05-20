"""Metric type registry: MetricType enum with built-in constructor.

Registering a new metric type
--------------------------------
Add a member to :class:`MetricType` whose name matches the ``type:
Literal['my_type']`` discriminator on the :class:`~pts.metrics.base.Metric`
subclass and whose value is the implementer class::

    class MetricType(Enum):
        ...
        my_type = MyMetric

No separate mapping dict is needed — the enum value IS the implementer.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pts.metrics.base import Metric
from pts.metrics.count import CountMetric, DistinctCountMetric
from pts.metrics.custom.l2g import L2GSignificantGeneMetric
from pts.metrics.grouped import GroupedCountExplodeMetric, GroupedCountMetric, GroupedSumMetric


class MetricType(Enum):
    """Supported metric kinds recognised by :meth:`MetricType.load`.

    Each member's **name** matches the ``type`` field used in YAML config;
    its **value** is the implementer :class:`~pts.metrics.base.Metric` class.
    """

    count = CountMetric
    distinct_count = DistinctCountMetric
    grouped_count = GroupedCountMetric
    grouped_count_explode = GroupedCountExplodeMetric
    grouped_sum = GroupedSumMetric
    l2g_significant_gene = L2GSignificantGeneMetric

    @classmethod
    def load(cls, cfg: dict[str, Any]) -> Metric:
        """Instantiate a :class:`~pts.metrics.base.Metric` from a config dict.

        The ``type`` key selects the :class:`MetricType` member by name;
        remaining keys are forwarded to the implementer class as keyword
        arguments.
        """
        cfg = dict(cfg)
        raw_type = cfg.pop('type', None)
        if raw_type is None:
            raise ValueError("metric entry is missing required 'type' field")
        try:
            return cls[raw_type].value(**cfg)
        except KeyError:
            raise ValueError(f"unknown metric type '{raw_type}'")
