"""Task that applies a transformation function to input files."""

from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any, Self

from loguru import logger
from otter.manifest.model import Artifact
from otter.storage.util import make_absolute
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.fs import check_destination
from pydantic import field_validator

from pts.metrics.base import Metric
from pts.metrics.count import CountMetric, DistinctCountMetric
from pts.metrics.distribution import DistributionMetric
from pts.metrics.grouped import GroupedCountMetric, GroupedSumMetric
from pts.metrics.runner import MetricRunner

TRANSFORMER_PACKAGE = 'pts.transformers'

_STANDARD_METRIC_TYPES: dict[str, type[Metric]] = {
    'count': CountMetric,
    'distinct_count': DistinctCountMetric,
    'distribution': DistributionMetric,
    'grouped_count': GroupedCountMetric,
    'grouped_sum': GroupedSumMetric,
}


def _load_metric(cfg: dict[str, Any]) -> Metric:
    cfg = dict(cfg)
    metric_type = cfg.pop('type', None)
    if metric_type in _STANDARD_METRIC_TYPES:
        return _STANDARD_METRIC_TYPES[metric_type](**cfg)
    elif metric_type == 'custom':
        class_path = cfg.pop('class')
        module_path, cls_name = class_path.rsplit('.', 1)
        module = import_module(module_path)
        cls = getattr(module, cls_name)
        return cls(**cfg)
    else:
        raise ValueError(f"unknown metric type '{metric_type}'")

path_or_paths = str | dict[str, str]
transformer_type = Callable[[str | dict[str, str], str | dict[str, str], dict[str, Any] | None], None]


class TransformSpec(Spec):
    """Configuration for the Transform task."""

    transformer: str
    """A string with the name of a transformer function.

        The function should be available in the package `pts.transformers`.

        It takes four arguments:
            * source: a string or a dict of source paths
            * destination: a string or a dict of destination paths
            * settings: a dict of settings to pass to the transformer
            * config: the config object

        The function should read the data from the source path, transform it,
        and write the result to the destination path. Both paths will be local.
    """
    source: path_or_paths
    """A string or a dictionary with the source paths."""
    destination: path_or_paths
    """A string or a dictionary with the destination paths."""
    settings: dict[str, Any] | None = None
    """A dictionary with settings to pass to the transformer."""
    metrics: list[Metric] = []
    """Metric definitions to compute after the transform step completes."""

    @field_validator('metrics', mode='before')
    @classmethod
    def _parse_metrics(cls, v: list[dict[str, Any]]) -> list[Metric]:
        return [_load_metric(cfg) for cfg in v]


class Transform(Task):
    """Task that applies a transformation function to input files."""

    def __init__(self, spec: TransformSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: TransformSpec
        self.transformer = self.load_transformer(spec.transformer)

    def _prepare_dirs(self, paths: str | dict[str, str]) -> None:
        if isinstance(paths, dict):
            for path in paths.values():
                check_destination(path, delete=True)
        else:
            check_destination(paths, delete=True)

    @staticmethod
    def load_transformer(transformer_name: str) -> transformer_type:
        try:
            module = import_module(f'{TRANSFORMER_PACKAGE}.{transformer_name}')
            transformer: transformer_type = getattr(
                module,
                transformer_name,
            )
            if not callable(transformer):
                raise TypeError(f'{transformer_name} is not a callable')
            return transformer
        except ImportError:
            raise ModuleNotFoundError(f'{transformer_name} not found in {TRANSFORMER_PACKAGE}')

    @report
    def run(self) -> Self:
        self.srcs = make_absolute(self.spec.source, self.context.config)
        self.dsts = make_absolute(self.spec.destination, self.context.config)

        logger.debug(f'running transformer {self.spec.transformer}')
        logger.debug(f'with source {self.srcs} and destination {self.dsts}')

        # prepare the destination directories if running locally
        if not self.context.config.release_uri:
            self._prepare_dirs(self.dsts)

        self.transformer(
            self.srcs,
            self.dsts,
            self.spec.settings or {},
            self.context.config,
        )

        srcs = list(self.srcs.values()) if isinstance(self.srcs, dict) else self.srcs
        dsts = list(self.dsts.values()) if isinstance(self.dsts, dict) else self.dsts
        self.artifacts = [Artifact(source=srcs, destination=dsts)]

        if self.spec.metrics:
            destination = self.dsts if isinstance(self.dsts, str) else next(iter(self.dsts.values()))
            dst_path = Path(destination)
            dataset_path = dst_path.parent if dst_path.suffix else dst_path
            dataset_name = dataset_path.name
            release = self.context.scratchpad.sentinel_dict.get('release', '')
            run = self.context.scratchpad.sentinel_dict.get('run', '')
            MetricRunner().run(
                metrics=self.spec.metrics,
                dataset_path=dataset_path,
                metrics_root=dataset_path.parent.parent / 'metrics',
                dataset_name=dataset_name,
                release=release,
                run=run,
            )

        return self
