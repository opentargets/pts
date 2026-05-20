"""Abstract base classes for the pts.metrics framework."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

import polars as pl
from loguru import logger
from pydantic import BaseModel

_ENVELOPE_FIELDS = frozenset({'name', 'metric_type'})


class UnifiedMetricRecord(BaseModel):
    """Flat JSONL record produced by MetricRunner for each metric computation.

    Attributes:
        name: Metric name as declared in config.
        metric_type: Metric kind identifier (e.g. ``'count'``, ``'grouped_count'``).
        release: Pipeline release string (e.g. ``'26.06-pub'``).
        run: Pipeline run identifier.
        dataset: Name of the source dataset directory.
        source: Absolute path to the input parquet directory.
        destination: Absolute path to the output JSONL file.
        result: JSON-serialised metric-specific payload (value, groups, etc.),
            excluding the envelope fields above.
    """

    name: str
    metric_type: str
    release: str
    run: str
    dataset: str
    source: str
    destination: str
    result: str


class MetricResult(BaseModel, ABC):
    """Abstract base for metric computation results.

    Subclasses add metric-specific fields (e.g. ``value``, ``groups``).
    Envelope fields (``release``, ``run``, ``dataset``, ``source``,
    ``destination``) are not stored here; they are injected by
    :class:`MetricRunner` via :meth:`to_unified_record`.

    Attributes:
        name: Metric name, copied from the originating :class:`Metric` definition.
        metric_type: Kind discriminator; set as a ``Literal`` on each subclass.
    """

    name: str
    metric_type: str

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


class Metric(BaseModel, ABC):
    """Abstract base for all metric definitions.

    Subclasses declare their configuration parameters as Pydantic fields and
    implement :meth:`compute`. Callers (i.e. :class:`MetricRunner`) invoke
    :meth:`run`, which adds debug logging around :meth:`compute`.

    Attributes:
        name: Unique metric name used to identify the result record in JSONL output.
    """

    name: str

    @property
    def required_columns(self) -> list[str] | None:
        """Columns needed by compute(). None = all columns; [] = row count only."""
        return None

    def run(self, df: pl.DataFrame) -> MetricResult:
        """Invoke :meth:`compute` with debug logging; entry point for :class:`MetricRunner`."""
        logger.debug('metric {} | {} rows', self.name, df.height)
        result = self.compute(df)
        logger.debug('metric {} | done ({})', self.name, result.metric_type)
        return result

    @abstractmethod
    def compute(self, df: pl.DataFrame) -> MetricResult:
        """Compute the metric from a Polars DataFrame."""
