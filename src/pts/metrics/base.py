"""Abstract base classes for the pts.metrics framework.

Adding a new built-in metric type
-----------------------------------
1. **Subclass** :class:`Metric` and :class:`MetricResult` in a module under
   ``pts/metrics/``, placing the metric class first::

       class MyMetric(Metric):
           type: Literal['my_type'] = 'my_type'
           some_param: str

           @property
           def required_columns(self) -> list[str]:
               return ['col_a', 'col_b']  # None = all columns, [] = row count only

           def compute(self, df: pl.DataFrame) -> MyResult:
               ...
               return MyResult(name=self.name, value=...)

       class MyResult(MetricResult):
           metric_type: Literal['my_type'] = 'my_type'
           value: int

2. **Register** in :mod:`pts.metrics.loader` — add ``my_type = 'my_type'`` to
   :class:`~pts.metrics.loader.MetricType` and ``MetricType.my_type: MyMetric``
   to ``_IMPLEMENTERS``.

3. **Export** from :mod:`pts.metrics` (``__init__.py``) if the type is part of
   the public API.

For one-off, dataset-specific metrics use ``type: custom`` in config and reference
the class by dotted path — no loader registration needed::

    metrics:
      - type: custom
        class: pts.metrics.custom.my_module.MyMetric
        name: my_metric_name
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

import polars as pl
from loguru import logger
from pydantic import BaseModel

_ENVELOPE_FIELDS = frozenset({'name', 'metric_type'})


class EmptyDatasetError(ValueError):
    """Raised when a metric is run against an empty DataFrame."""


class Metric(BaseModel, ABC):
    """Abstract base for all metric definitions.

    Subclasses declare their configuration parameters as Pydantic fields and
    implement :meth:`compute`. Callers (i.e. :class:`MetricRunner`) invoke
    :meth:`run`, which wraps :meth:`compute` with debug logging.

    Each built-in subclass must carry a ``type: Literal['<kind>'] = '<kind>'``
    discriminator field so that :func:`~pts.metrics.loader.load_metric` can
    reconstruct the correct class from a config dict.
    """

    name: str
    """Unique metric name used to identify the result record in JSONL output."""

    @property
    def required_columns(self) -> list[str] | None:
        """Columns needed by compute(). None = all columns; [] = row count only."""
        return None

    def run(self, df: pl.DataFrame) -> MetricResult:
        """Invoke :meth:`compute` with debug logging; entry point for :class:`MetricRunner`.

        Raises :class:`EmptyDatasetError` if ``df`` has no rows.
        """
        if df.is_empty():
            raise EmptyDatasetError(f"metric '{self.name}': input DataFrame is empty")
        logger.debug('metric {} | {} rows', self.name, df.height)
        result = self.compute(df)
        logger.debug('metric {} | done ({})', self.name, result.metric_type)
        return result

    @abstractmethod
    def compute(self, df: pl.DataFrame) -> MetricResult:
        """Compute the metric from a Polars DataFrame."""


class MetricResult(BaseModel, ABC):
    """Abstract base for metric computation results.

    Subclasses add metric-specific fields (e.g. ``value``, ``groups``).
    Envelope fields (``release``, ``run``, ``dataset``, ``source``,
    ``destination``) are not stored here; they are injected by
    :class:`MetricRunner` via :meth:`to_unified_record`.

    Each subclass must carry a ``metric_type: Literal['<kind>'] = '<kind>'``
    discriminator field that matches the ``type`` value of its companion
    :class:`Metric` subclass.
    """

    name: str
    """Metric name, copied from the originating :class:`Metric` definition."""
    metric_type: str
    """Kind discriminator; set as a ``Literal`` on each subclass."""

    def to_unified_record(
        self,
        *,
        release: str,
        run: str,
        dataset: str,
        source: str,
        destination: str,
    ) -> UnifiedMetricRecord:
        """Produce a :class:`UnifiedMetricRecord` from this result and envelope data.

        Metric-specific fields (everything except ``name`` and ``metric_type``)
        are JSON-serialised into the ``result`` field.
        """
        payload = {k: v for k, v in self.model_dump().items() if k not in _ENVELOPE_FIELDS}
        return UnifiedMetricRecord(
            name=self.name,
            metric_type=self.metric_type,
            release=release,
            run=run,
            dataset=dataset,
            source=source,
            destination=destination,
            result=json.dumps(payload),
        )


class UnifiedMetricRecord(BaseModel):
    """Flat JSONL record produced by MetricRunner for each metric computation."""

    name: str
    """Metric name as declared in config."""
    metric_type: str
    """Metric kind identifier (e.g. ``'count'``, ``'grouped_count'``)."""
    release: str
    """Pipeline release string (e.g. ``'26.06-pub'``)."""
    run: str
    """Pipeline run identifier."""
    dataset: str
    """Name of the source dataset directory."""
    source: str
    """Absolute path to the input parquet directory."""
    destination: str
    """Absolute path to the output JSONL file."""
    result: str
    """JSON-serialised metric-specific payload (value, groups, etc.), excluding envelope fields."""
