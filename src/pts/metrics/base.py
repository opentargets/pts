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

2. **Register** in :mod:`pts.metrics.loader` — add ``my_type = MyMetric`` to
   :class:`~pts.metrics.loader.MetricType`. The member name must match the
   ``type`` literal; the value is the implementer class directly.

3. **Export** from :mod:`pts.metrics` (``__init__.py``) if the type is part of
   the public API.
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
    implement :meth:`compute`. Callers invoke :meth:`run`, which wraps
    :meth:`compute` with debug logging and optional pre-filtering.

    Each subclass must carry a ``type: Literal['<kind>'] = '<kind>'``
    discriminator field so that :meth:`~pts.metrics.loader.MetricType.load`
    can reconstruct the correct class from a config dict.
    """

    name: str
    """Unique metric name used to identify the result row in Parquet output."""
    filter_expr: str | None = None
    """Optional SQL WHERE-clause expression applied to the dataset before :meth:`compute` is called.

    When set, all dataset columns are loaded regardless of :attr:`required_columns`
    so the expression can reference any column.
    """

    @property
    def required_columns(self) -> list[str] | None:
        """Columns needed by compute(). None = all columns; [] = row count only."""
        return None

    def run(self, df: pl.DataFrame) -> MetricResult:
        """Invoke :meth:`compute` with debug logging.

        Applies :attr:`filter_expr` (if set) before calling :meth:`compute`.
        Raises :class:`EmptyDatasetError` if ``df`` has no rows.

        >>> from pts.metrics.count import CountMetric, DistinctCountMetric
        >>> import polars as pl
        >>> CountMetric(name='x').run(pl.DataFrame({'a': pl.Series([], dtype=pl.Int64)}))
        Traceback (most recent call last):
            ...
        pts.metrics.base.EmptyDatasetError: metric 'x': input DataFrame is empty
        >>> CountMetric(name='n', filter_expr='score >= 0.5').run(pl.DataFrame({'score': [0.8, 0.3, 0.9]})).value
        2
        >>> m = DistinctCountMetric(name='g', columns=['id'], filter_expr='score >= 0.5')
        >>> m.run(pl.DataFrame({'id': ['A', 'A', 'B'], 'score': [0.9, 0.3, 0.8]})).value
        2
        """
        if df.is_empty():
            raise EmptyDatasetError(f"metric '{self.name}': input DataFrame is empty")
        if self.filter_expr is not None:
            df = df.filter(pl.sql_expr(self.filter_expr))
            logger.debug('metric {} | {} rows after filter', self.name, df.height)
        else:
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
    ``destination``) are not stored here; they are injected via
    :meth:`to_unified_record`.

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
        filter_expr: str | None = None,
    ) -> UnifiedMetricRecord:
        """Produce a :class:`UnifiedMetricRecord` from this result and envelope data.

        Metric-specific fields (everything except ``name`` and ``metric_type``)
        are JSON-serialised into the ``result`` field.

        >>> from pts.metrics.count import CountMetric
        >>> import polars as pl
        >>> r = CountMetric(name='n').run(pl.DataFrame({'x': [1]}))
        >>> r.to_unified_record(release='r', run='1', dataset='d', source='/s', destination='/d').filter_expr is None
        True
        >>> r.to_unified_record(
        ...     release='r', run='1', dataset='d', source='/s', destination='/d', filter_expr='x>0',
        ... ).filter_expr
        'x>0'
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
            filter_expr=filter_expr,
            result=json.dumps(payload),
        )


class UnifiedMetricRecord(BaseModel):
    """Flat record produced for each metric computation; written as a Parquet row."""

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
    """Absolute path to the output Parquet file."""
    filter_expr: str | None = None
    """SQL WHERE-clause expression used to pre-filter the dataset, or ``None`` if no filter was applied."""
    result: str
    """JSON-serialised metric-specific payload (value, groups, etc.), excluding envelope fields."""
