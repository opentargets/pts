"""Custom metric: disease grouped by therapeutic area."""
from __future__ import annotations

import polars as pl

from pts.metrics.base import Metric
from pts.metrics.grouped import GroupedCountResult, GroupRow


class DiseaseByTherapeuticAreaMetric(Metric):
    """Counts diseases per therapeutic area by exploding the therapeuticAreas array."""

    def compute(self, df: pl.DataFrame) -> GroupedCountResult:
        """Compute per-therapeutic-area disease counts."""
        # 1. Filter out rows where therapeuticAreas is null
        filtered = df.drop_nulls(subset=['therapeuticAreas'])

        if filtered.is_empty():
            groups: list[GroupRow] = []
        else:
            # 2. Explode the therapeuticAreas list column
            exploded = filtered.explode('therapeuticAreas')

            # 3. Group by therapeuticAreas, count rows per group
            # 4. Sort descending by count
            agg = (
                exploded.group_by('therapeuticAreas')
                .agg(pl.len().alias('count'))
                .sort('count', descending=True)
            )

            # 5. Build GroupRow list
            groups = [
                GroupRow(key={'therapeuticAreas': row['therapeuticAreas']}, count=row['count'])
                for row in agg.iter_rows(named=True)
            ]

        return GroupedCountResult(
            name=self.name,
            release='',
            run='',
            group_by=['therapeuticAreas'],
            groups=groups,
        )
