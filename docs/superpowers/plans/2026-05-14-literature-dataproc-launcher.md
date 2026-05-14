# Literature Pipeline — Dataproc Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the `vh-add-literature` branch's literature pipeline on Dataproc via per-stage launcher scripts, with two adjacent step-pairs collapsed into single jobs.

**Architecture:** Two new "collapsed" PySpark modules (`literature_publication_match`, `literature_cooccurrence_evidence`) chain adjacent steps in one job; five bash launcher scripts under `scripts/literature/` (sharing a `_common.sh` library) create Dataproc clusters and submit the jobs in dependency order, all writing under `gs://ot-team/dochoa/literature_runs/<RUN_ID>/`.

**Tech Stack:** Python 3.11, PySpark, the `literature` (`ot-literature`) and `ontoma` libraries, Otter task framework, `gcloud dataproc`, bash.

**Reference:** Design doc at `docs/superpowers/specs/2026-05-14-literature-dataproc-launcher-design.md`. Template launcher at `scripts/launch_association_benchmark.sh`.

---

## File Structure

**New Python modules:**
- `src/pts/pyspark/literature_publication_match.py` — collapsed publication+match step. Pure helpers `_epmc_read_path`, `_maybe_repartition`, `_read_publications`; entry point `literature_publication_match`.
- `src/pts/pyspark/literature_cooccurrence_evidence.py` — collapsed cooccurrence+evidence step. Pure helper `_adapt_cooccurrence_for_evidence`; entry point `literature_cooccurrence_evidence`.

**New tests:**
- `test/test_literature_publication_match.py` — tests `_epmc_read_path`, `_maybe_repartition`.
- `test/test_literature_cooccurrence_evidence.py` — tests `_adapt_cooccurrence_for_evidence`.

**New bash scripts (all under `scripts/literature/`):**
- `_common.sh` — sourced library: GCP config, runner/init-script generation, upload, `ensure_cluster`, `submit_job`, `print_monitoring`, `run_stage` orchestrator.
- `01_launch_ontoma_lut.sh` — stage 01, `pts`-like cluster.
- `02_launch_publication_match.sh` — stage 02, `pts_openfda`-like + spark-nlp cluster.
- `03a_launch_entity_lut.sh` — stage 03a, shared stage-3 cluster.
- `03b_launch_embedding_vector.sh` — stage 03b, shared stage-3 cluster.
- `03c_launch_cooccurrence_evidence.sh` — stage 03c, shared stage-3 cluster.

**Modified:**
- `config.yaml` — fix the `literature` step group's path inconsistency; add `literature_publication_match` and `literature_cooccurrence_evidence` step entries.

---

## Background the engineer needs

- **PySpark step modules** live in `src/pts/pyspark/`. Each exposes a function named after its file, with signature `(source, destination, settings, properties)`. Otter's `Pyspark` task (`src/pts/tasks/pyspark.py`) imports `pts.pyspark.<name>` and calls `<name>`. `source`/`destination` are `dict[str,str]`; relative paths are resolved against `release_uri`, absolute (`gs://...`) paths are passed through.
- **The `literature` library** (`literature` package, installed from `ot-literature`). Key classes:
  - `EPMCPublication` (`literature.datasource.epmc.publication`) — has `defined_schema` (a `StructType`) and `from_source(session, epmc_path, lut)`. `from_source` hardcodes the read glob `{epmc_path}/{kind}/**/*.jsonl`, which is why we cannot date-filter through it. Its building-block static methods `_annotate_fulltexts_with_pmid(fulltext, lut)`, `_merge_abstracts_with_fulltexts(abstract, fulltext)`, `_get_most_recent_publications(df)` *are* reusable and we call them directly.
  - `PublicationIdLUT` (`literature.datasource.epmc.publication_id_lut`) — `from_csv(session, csv_path)` returns a parsed LUT DataFrame. It only touches `session.spark`, so the PTS `Session` works as the `session` argument (duck-typed — `literature_publication.py` already does this).
  - `Publication` (`literature.dataset.publication`) — a dataclass wrapping a DataFrame; `Publication(df)` validates `df` against `publication.json`. Has `.extract_matches()` → `Match`.
  - `Match` (`literature.dataset.match`) — has `.map_labels(session, label_lut_path, label_col_name, type_col_name)` → `MatchMapped`. `map_labels` internally constructs `OnToma(...)` and calls `map_entities`, which runs the spark-nlp normalisation pipeline — **this is why stage 02 needs spark-nlp.**
  - `MatchMapped` (`literature.dataset.match_mapped`) — has `.disambiguate(trusted_sources)` → `MatchMapped`, and `.generate_target_disease_cooccurrences()` → `Cooccurrence`.
- **The PTS `Session`** (`pts.pyspark.common.session.Session`) — `Session(app_name=..., properties=...)`; `.spark` is the raw `SparkSession`; `.load_data(path=...)` reads parquet by default.
- **Known schema mismatch (handled in Task 4):** the `Cooccurrence` schema (`literature/schemas/cooccurrence.json`) names columns `mappedId1`, `mappedId2`, `evidenceScore`. But `pts.pyspark.evidence_epmc._compute_evidence` consumes `keywordId1`, `keywordId2`, `evidence_score`. The collapsed `literature_cooccurrence_evidence` module bridges this with a rename helper. (The pre-existing non-collapsed `evidence_epmc` step has the same latent mismatch; fixing that is out of scope.)
- **Tests** use the session-scoped `spark` fixture from `test/conftest.py`. Existing literature-step tests (`test/test_literature_entity_lut.py`, `test/test_evidence_epmc.py`) test *pure helper functions* and do not test the `Session`-creating entry points — this plan follows that convention.
- **Commit style:** Conventional Commits, no `Co-Authored-By` lines (per repo/global config).

---

## Phase 1 — Collapsed PySpark modules + config

### Task 1: `_epmc_read_path` helper

**Files:**
- Create: `src/pts/pyspark/literature_publication_match.py`
- Test: `test/test_literature_publication_match.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_literature_publication_match.py`:

```python
"""Tests for literature_publication_match step."""


class TestEpmcReadPath:
    """Test the EPMC jsonl glob path builder."""

    def test_with_date_prefix(self):
        from pts.pyspark.literature_publication_match import _epmc_read_path

        assert (
            _epmc_read_path('gs://otar025-epmc/ml02', 'fulltext', '2026_03')
            == 'gs://otar025-epmc/ml02/fulltext/2026_03*/**/*.jsonl'
        )

    def test_without_date_prefix(self):
        from pts.pyspark.literature_publication_match import _epmc_read_path

        assert (
            _epmc_read_path('gs://otar025-epmc/ml02', 'abstract', None)
            == 'gs://otar025-epmc/ml02/abstract/**/*.jsonl'
        )

    def test_strips_trailing_slash(self):
        from pts.pyspark.literature_publication_match import _epmc_read_path

        assert (
            _epmc_read_path('gs://otar025-epmc/ml02/', 'fulltext', None)
            == 'gs://otar025-epmc/ml02/fulltext/**/*.jsonl'
        )

    def test_empty_date_prefix_treated_as_none(self):
        from pts.pyspark.literature_publication_match import _epmc_read_path

        assert (
            _epmc_read_path('gs://otar025-epmc/ml02', 'abstract', '')
            == 'gs://otar025-epmc/ml02/abstract/**/*.jsonl'
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_literature_publication_match.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pts.pyspark.literature_publication_match'`

- [ ] **Step 3: Write minimal implementation**

Create `src/pts/pyspark/literature_publication_match.py`:

```python
"""Collapsed literature_publication + literature_match step.

Reads EPMC publications, builds the publication dataset in memory (no
intermediate publication parquet is written), extracts and maps matches, then
writes only the valid and failed match datasets.
"""

from __future__ import annotations


def _epmc_read_path(epmc_path: str, kind: str, date_prefix: str | None) -> str:
    """Build the EPMC jsonl glob path for a publication kind.

    Mirrors ``EPMCPublication._read_in_with_schema`` but allows an optional
    day-folder prefix so a single month can be selected. The library hardcodes
    its own glob, hence this reimplementation.

    Args:
        epmc_path: Base EPMC path, e.g. ``gs://otar025-epmc/ml02``.
        kind: Publication kind, ``abstract`` or ``fulltext``.
        date_prefix: Optional day-folder prefix, e.g. ``2026_03``. Falsy values
            (None, empty string) select all dates.

    Returns:
        A glob path string suitable for ``spark.read.json``.
    """
    base = epmc_path.rstrip('/')
    if date_prefix:
        return f'{base}/{kind}/{date_prefix}*/**/*.jsonl'
    return f'{base}/{kind}/**/*.jsonl'
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_literature_publication_match.py -v`
Expected: PASS — 4 passed

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/pts/pyspark/literature_publication_match.py test/test_literature_publication_match.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/pts/pyspark/literature_publication_match.py test/test_literature_publication_match.py
git commit -m "feat(literature): add _epmc_read_path glob builder for publication_match"
```

---

### Task 2: `_maybe_repartition` helper

**Files:**
- Modify: `src/pts/pyspark/literature_publication_match.py`
- Test: `test/test_literature_publication_match.py`

- [ ] **Step 1: Write the failing test**

Append to `test/test_literature_publication_match.py`:

```python
class TestMaybeRepartition:
    """Test the optional repartition helper."""

    def test_repartitions_when_count_given(self, spark):
        from pts.pyspark.literature_publication_match import _maybe_repartition

        df = spark.range(100)
        result = _maybe_repartition(df, 4)
        assert result.rdd.getNumPartitions() == 4

    def test_returns_df_unchanged_when_none(self, spark):
        from pts.pyspark.literature_publication_match import _maybe_repartition

        df = spark.range(100)
        result = _maybe_repartition(df, None)
        assert result is df

    def test_returns_df_unchanged_when_zero(self, spark):
        from pts.pyspark.literature_publication_match import _maybe_repartition

        df = spark.range(100)
        result = _maybe_repartition(df, 0)
        assert result is df
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_literature_publication_match.py::TestMaybeRepartition -v`
Expected: FAIL — `ImportError: cannot import name '_maybe_repartition'`

- [ ] **Step 3: Write minimal implementation**

In `src/pts/pyspark/literature_publication_match.py`, add the `TYPE_CHECKING` import block directly under the module docstring's `from __future__ import annotations` line:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame
```

Then add this function after `_epmc_read_path`:

```python
def _maybe_repartition(df: DataFrame, repartition: int | None) -> DataFrame:
    """Repartition a DataFrame to a fixed partition count when configured.

    The raw EPMC input is scattered across many small day-folder files; an
    explicit repartition right after the read keeps Spark task counts sane.

    Args:
        df: DataFrame to repartition.
        repartition: Target partition count. Falsy values leave ``df`` unchanged.

    Returns:
        The repartitioned DataFrame, or ``df`` unchanged when ``repartition`` is falsy.
    """
    if repartition:
        return df.repartition(repartition)
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_literature_publication_match.py -v`
Expected: PASS — 7 passed

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/pts/pyspark/literature_publication_match.py test/test_literature_publication_match.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/pts/pyspark/literature_publication_match.py test/test_literature_publication_match.py
git commit -m "feat(literature): add _maybe_repartition helper for publication_match"
```

---

### Task 3: `_read_publications` + `literature_publication_match` entry point

**Files:**
- Modify: `src/pts/pyspark/literature_publication_match.py`

No unit test: this code creates a `Session` and calls library readers — following the repo convention (`evidence_epmc.py`'s entry point is likewise untested). It is validated by the Dataproc run in Phase 2. Verification here is import + lint.

- [ ] **Step 1: Add imports**

In `src/pts/pyspark/literature_publication_match.py`, replace the existing import block (the `from __future__` line plus the `TYPE_CHECKING` block) with:

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from literature.datasource.epmc.publication import EPMCPublication
from literature.datasource.epmc.publication_id_lut import PublicationIdLUT
from literature.dataset.publication import Publication
from loguru import logger
from pyspark.sql import functions as f

from pts.pyspark.common.session import Session

if TYPE_CHECKING:
    from pyspark.sql import DataFrame
```

- [ ] **Step 2: Add `_read_publications`**

Add after `_maybe_repartition`:

```python
def _read_publications(
    session: Session,
    epmc_path: str,
    pub_id_lut: DataFrame,
    date_prefix: str | None,
    repartition: int | None,
) -> Publication:
    """Read EPMC publications into a ``Publication`` dataset, in memory.

    Replicates ``EPMCPublication.from_source`` but with a parameterisable
    date-folder glob and an optional post-read repartition. No intermediate
    publication parquet is written. The library's underscore-prefixed building
    blocks are reused intentionally — the only part that must be reimplemented
    is the read, because the library hardcodes its glob.

    Args:
        session: PTS Spark session wrapper.
        epmc_path: Base EPMC path, e.g. ``gs://otar025-epmc/ml02``.
        pub_id_lut: Parsed publication id lookup table DataFrame.
        date_prefix: Optional day-folder prefix, e.g. ``2026_03``.
        repartition: Optional partition count applied right after the read.

    Returns:
        A ``Publication`` dataset built entirely in memory.
    """
    spark = session.spark
    schema = EPMCPublication.defined_schema

    def _read_kind(kind: str) -> DataFrame:
        df = (
            spark.read.schema(schema)
            .json(_epmc_read_path(epmc_path, kind, date_prefix))
            .withColumn('kind', f.lit(kind))
            .withColumn('traceSource', f.input_file_name())
        )
        return _maybe_repartition(df, repartition)

    fulltexts = _read_kind('fulltext')
    processed_fulltexts = EPMCPublication._annotate_fulltexts_with_pmid(fulltexts, pub_id_lut)

    abstracts = _read_kind('abstract')
    all_publications = EPMCPublication._merge_abstracts_with_fulltexts(abstracts, processed_fulltexts)

    most_recent = EPMCPublication._get_most_recent_publications(all_publications)
    return Publication(
        _df=most_recent.repartition(f.col('pmid')),
        _schema=Publication.get_schema(),
    )
```

- [ ] **Step 3: Add the entry point**

Add at the end of the file:

```python
def literature_publication_match(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Collapsed literature_publication + literature_match step.

    Reads EPMC publications (optionally restricted to a single month via
    ``settings.date_prefix``), builds the publication dataset in memory, extracts
    and maps matches, disambiguates them, and writes only the valid and failed
    match datasets. The publication parquet is never written.

    Args:
        source: ``pub_id_lut``, ``epmc_publication``, ``ontoma_disease_target_drug_label_lut``.
        destination: ``match_valid``, ``match_failed``.
        settings: optional ``date_prefix`` (str) and ``repartition`` (int).
        properties: Spark properties forwarded to the session.
    """
    spark = Session(app_name='literature', properties=properties)

    date_prefix = settings.get('date_prefix')
    repartition = settings.get('repartition')

    logger.info(f'load publication id lut from: {source["pub_id_lut"]}')
    pub_id_lut = PublicationIdLUT.from_csv(spark, source['pub_id_lut']).persist()

    logger.info(
        f'read EPMC publications from: {source["epmc_publication"]} '
        f'(date_prefix={date_prefix}, repartition={repartition})'
    )
    publication = _read_publications(
        spark,
        source['epmc_publication'],
        pub_id_lut,
        date_prefix,
        repartition,
    )

    logger.info('extract matches and map labels')
    match_mapped = (
        publication
        .extract_matches()
        .map_labels(
            session=spark,
            label_lut_path=source['ontoma_disease_target_drug_label_lut'],
            label_col_name='label',
            type_col_name='type',
        )
    )
    # consumed by match_disambiguated and the isMapped==False filter
    match_mapped.df.persist()

    logger.info('disambiguate')
    match_disambiguated = match_mapped.disambiguate(
        trusted_sources=[
            'name',
            'ot_curation',
            'eva_clinvar',
            'clinvar_xrefs',
            'approved_name',
            'approved_symbol',
        ]
    )
    # consumed by the isValid==True and isValid==False filters
    match_disambiguated.df.persist()

    match_valid = match_disambiguated.df.filter(f.col('isValid'))

    # rows that fail mapping are emitted via the isMapped==False branch, so
    # guard the disambiguation branch with isMapped==True to keep the union disjoint
    match_failed = (
        match_mapped.df
        .filter(~f.col('isMapped'))
        .unionByName(
            match_disambiguated.df
            .filter(f.col('isMapped'))
            .filter(~f.col('isValid')),
            allowMissingColumns=True,
        )
    )

    logger.info(f'write valid matches to {destination["match_valid"]}')
    match_valid.write.mode('overwrite').parquet(destination['match_valid'])

    logger.info(f'write failed matches to {destination["match_failed"]}')
    match_failed.write.mode('overwrite').parquet(destination['match_failed'])

    match_mapped.df.unpersist()
    match_disambiguated.df.unpersist()
```

- [ ] **Step 4: Verify it imports**

Run: `uv run python -c "from pts.pyspark.literature_publication_match import literature_publication_match, _read_publications"`
Expected: no output, exit 0

- [ ] **Step 5: Run existing tests + lint**

Run: `uv run pytest test/test_literature_publication_match.py -v`
Expected: PASS — 7 passed

Run: `uv run ruff check src/pts/pyspark/literature_publication_match.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/pts/pyspark/literature_publication_match.py
git commit -m "feat(literature): add literature_publication_match collapsed step"
```

---

### Task 4: `_adapt_cooccurrence_for_evidence` helper

**Files:**
- Create: `src/pts/pyspark/literature_cooccurrence_evidence.py`
- Test: `test/test_literature_cooccurrence_evidence.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_literature_cooccurrence_evidence.py`:

```python
"""Tests for literature_cooccurrence_evidence step."""

from pyspark.sql import Row


class TestAdaptCooccurrenceForEvidence:
    """Test the Cooccurrence -> _compute_evidence column adapter."""

    def test_renames_columns(self, spark):
        from pts.pyspark.literature_cooccurrence_evidence import (
            _adapt_cooccurrence_for_evidence,
        )

        df = spark.createDataFrame([
            Row(mappedId1='ENSG001', mappedId2='EFO_0000311', evidenceScore=0.8, section='abstract'),
        ])
        result = _adapt_cooccurrence_for_evidence(df)
        assert 'keywordId1' in result.columns
        assert 'keywordId2' in result.columns
        assert 'evidence_score' in result.columns

    def test_drops_old_names(self, spark):
        from pts.pyspark.literature_cooccurrence_evidence import (
            _adapt_cooccurrence_for_evidence,
        )

        df = spark.createDataFrame([
            Row(mappedId1='ENSG001', mappedId2='EFO_0000311', evidenceScore=0.8, section='abstract'),
        ])
        result = _adapt_cooccurrence_for_evidence(df)
        assert 'mappedId1' not in result.columns
        assert 'mappedId2' not in result.columns
        assert 'evidenceScore' not in result.columns

    def test_preserves_other_columns_and_values(self, spark):
        from pts.pyspark.literature_cooccurrence_evidence import (
            _adapt_cooccurrence_for_evidence,
        )

        df = spark.createDataFrame([
            Row(mappedId1='ENSG001', mappedId2='EFO_0000311', evidenceScore=0.8, section='abstract'),
        ])
        result = _adapt_cooccurrence_for_evidence(df)
        assert 'section' in result.columns
        row = result.collect()[0]
        assert row['keywordId1'] == 'ENSG001'
        assert row['evidence_score'] == 0.8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_literature_cooccurrence_evidence.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pts.pyspark.literature_cooccurrence_evidence'`

- [ ] **Step 3: Write minimal implementation**

Create `src/pts/pyspark/literature_cooccurrence_evidence.py`:

```python
"""Collapsed literature_cooccurrence + evidence_epmc step.

Generates target-disease cooccurrences from matches, writes the intermediate
cooccurrence parquet, then re-reads it and computes EPMC evidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


def _adapt_cooccurrence_for_evidence(cooccurrences: DataFrame) -> DataFrame:
    """Rename ``Cooccurrence``-schema columns to the names ``_compute_evidence`` expects.

    The ``literature`` library's ``Cooccurrence`` dataset names the mapped entity
    ids ``mappedId1``/``mappedId2`` and the score ``evidenceScore``, but
    ``pts.pyspark.evidence_epmc._compute_evidence`` consumes
    ``keywordId1``/``keywordId2``/``evidence_score``. This bridges the two.

    Args:
        cooccurrences: A DataFrame with the ``Cooccurrence`` schema.

    Returns:
        The same DataFrame with the three columns renamed.
    """
    return (
        cooccurrences
        .withColumnRenamed('mappedId1', 'keywordId1')
        .withColumnRenamed('mappedId2', 'keywordId2')
        .withColumnRenamed('evidenceScore', 'evidence_score')
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_literature_cooccurrence_evidence.py -v`
Expected: PASS — 3 passed

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/pts/pyspark/literature_cooccurrence_evidence.py test/test_literature_cooccurrence_evidence.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/pts/pyspark/literature_cooccurrence_evidence.py test/test_literature_cooccurrence_evidence.py
git commit -m "feat(literature): add _adapt_cooccurrence_for_evidence column bridge"
```

---

### Task 5: `literature_cooccurrence_evidence` entry point

**Files:**
- Modify: `src/pts/pyspark/literature_cooccurrence_evidence.py`

No unit test: `Session`-creating entry point, validated by the Dataproc run (repo convention). Verification = import + lint.

- [ ] **Step 1: Add imports**

In `src/pts/pyspark/literature_cooccurrence_evidence.py`, replace the import block (the `from __future__` line plus the `TYPE_CHECKING` block) with:

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from literature.dataset.match_mapped import MatchMapped
from loguru import logger

from pts.pyspark import evidence_epmc
from pts.pyspark.common.session import Session

if TYPE_CHECKING:
    from pyspark.sql import DataFrame
```

Note: `evidence_epmc` is imported as a module (not `from ... import _compute_evidence`) so that reusing its `_compute_evidence` helper is plain attribute access — this avoids a private-name-import lint warning.

- [ ] **Step 2: Add the entry point**

Add at the end of the file:

```python
def literature_cooccurrence_evidence(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Collapsed literature_cooccurrence + evidence_epmc step.

    Generates target-disease cooccurrences from the match dataset and writes the
    intermediate cooccurrence parquet. Then re-reads that parquet (a clean
    lineage cut on a memory-tight cluster), bridges its column names, and
    computes EPMC evidence.

    Args:
        source: ``match``.
        destination: ``cooccurrence``, ``evidence``.
        settings: unused.
        properties: Spark properties forwarded to the session.
    """
    spark = Session(app_name='literature', properties=properties)

    logger.info(f'load matches from: {source["match"]}')
    match = spark.load_data(path=source['match'])

    logger.info(f'write cooccurrences to {destination["cooccurrence"]}')
    cooccurrence = MatchMapped(match).generate_target_disease_cooccurrences().df
    cooccurrence.write.mode('overwrite').parquet(destination['cooccurrence'])

    logger.info('re-read cooccurrences and compute EPMC evidence')
    cooccurrence_reread = spark.spark.read.parquet(destination['cooccurrence'])
    evidence = evidence_epmc._compute_evidence(_adapt_cooccurrence_for_evidence(cooccurrence_reread))

    logger.info(f'write EPMC evidence to {destination["evidence"]}')
    evidence.write.mode('overwrite').parquet(destination['evidence'])
```

- [ ] **Step 3: Verify it imports**

Run: `uv run python -c "from pts.pyspark.literature_cooccurrence_evidence import literature_cooccurrence_evidence"`
Expected: no output, exit 0

- [ ] **Step 4: Run existing tests + lint**

Run: `uv run pytest test/test_literature_cooccurrence_evidence.py -v`
Expected: PASS — 3 passed

Run: `uv run ruff check src/pts/pyspark/literature_cooccurrence_evidence.py`
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/pts/pyspark/literature_cooccurrence_evidence.py
git commit -m "feat(literature): add literature_cooccurrence_evidence collapsed step"
```

---

### Task 6: Fix the `literature` step group path inconsistency in `config.yaml`

**Files:**
- Modify: `config.yaml` (lines ~1767-1792)

The `literature` step group references the match and cooccurrence datasets with underscored paths that don't match what their producers write. Canonical form is the slashed path.

- [ ] **Step 1: Fix the `literature_entity_lut` source path**

In `config.yaml`, replace:

```yaml
    - name: pyspark literature_entity_lut
      pyspark: literature_entity_lut
      source:
        matches: intermediate/literature_match
```

with:

```yaml
    - name: pyspark literature_entity_lut
      pyspark: literature_entity_lut
      source:
        matches: intermediate/literature/match
```

- [ ] **Step 2: Fix the `literature_embedding` source path**

In `config.yaml`, replace:

```yaml
    - name: pyspark literature_embedding
      pyspark: literature_embedding
      source:
        matches: intermediate/literature_match
```

with:

```yaml
    - name: pyspark literature_embedding
      pyspark: literature_embedding
      source:
        matches: intermediate/literature/match
```

- [ ] **Step 3: Fix the `evidence_epmc` source path**

In `config.yaml`, replace:

```yaml
    - name: pyspark evidence_epmc
      pyspark: evidence_epmc
      source:
        cooccurrences: intermediate/literature_cooccurrence
```

with:

```yaml
    - name: pyspark evidence_epmc
      pyspark: evidence_epmc
      source:
        cooccurrences: intermediate/literature/cooccurrence
```

- [ ] **Step 4: Verify the config still parses**

Run: `uv run python -c "import yaml; yaml.safe_load(open('config.yaml'))"`
Expected: no output, exit 0

- [ ] **Step 5: Verify there are no underscored paths left**

Run: `grep -nE 'intermediate/literature_(match|cooccurrence)' config.yaml || echo "none remaining"`
Expected: `none remaining`

- [ ] **Step 6: Commit**

```bash
git add config.yaml
git commit -m "fix(config): use canonical slashed paths in literature step group"
```

---

### Task 7: Add the two collapsed step entries to `config.yaml`

**Files:**
- Modify: `config.yaml` (insert after the `literature_cooccurrence` step, line ~1826)

- [ ] **Step 1: Insert the two new step groups**

In `config.yaml`, find this block (end of the `literature_cooccurrence` step):

```yaml
  literature_cooccurrence:
    - name: pyspark literature cooccurrence
      pyspark: literature_cooccurrence
      source:
        match: intermediate/literature/match
      destination:
        cooccurrence: intermediate/literature/cooccurrence
  ##################################################################################################
```

Replace it with (i.e. append the two new step groups after it):

```yaml
  literature_cooccurrence:
    - name: pyspark literature cooccurrence
      pyspark: literature_cooccurrence
      source:
        match: intermediate/literature/match
      destination:
        cooccurrence: intermediate/literature/cooccurrence
  ##################################################################################################

  #: LITERATURE_PUBLICATION_MATCH STEP (collapsed publication + match) :###########################
  literature_publication_match:
    - name: pyspark literature publication match
      pyspark: literature_publication_match
      source:
        pub_id_lut: input/literature/PMID_PMCID_DOI.csv.gz
        epmc_publication: gs://otar025-epmc/ml02
        ontoma_disease_target_drug_label_lut: intermediate/ontoma/disease_target_drug_label_lookup_table.parquet
      destination:
        match_valid: intermediate/literature/match
        match_failed: excluded/literature/match
  ##################################################################################################

  #: LITERATURE_COOCCURRENCE_EVIDENCE STEP (collapsed cooccurrence + evidence) :####################
  literature_cooccurrence_evidence:
    - name: pyspark literature cooccurrence evidence
      pyspark: literature_cooccurrence_evidence
      source:
        match: intermediate/literature/match
      destination:
        cooccurrence: intermediate/literature/cooccurrence
        evidence: intermediate/evidence/literature_epmc
  ##################################################################################################
```

Note: the canonical `config.yaml` entries deliberately omit `settings` — the full-data run uses no `date_prefix`. The launcher's generated config (Task 10) adds `date_prefix`/`repartition` for the test run.

- [ ] **Step 2: Verify the config still parses and the steps are present**

Run: `uv run python -c "import yaml; c = yaml.safe_load(open('config.yaml')); assert 'literature_publication_match' in c['steps']; assert 'literature_cooccurrence_evidence' in c['steps']; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add config.yaml
git commit -m "feat(config): add collapsed literature step entries"
```

---

## Phase 2 — Dataproc launcher scripts

All scripts go under `scripts/literature/`. They are validated by `bash -n` (syntax) here; end-to-end validation is the actual Dataproc run, which the operator performs after this plan.

### Task 8: `scripts/literature/_common.sh` shared library

**Files:**
- Create: `scripts/literature/_common.sh`

- [ ] **Step 1: Write the file**

Create `scripts/literature/_common.sh`:

```bash
#!/usr/bin/env bash
# Shared helpers for the literature Dataproc launcher scripts.
#
# Each stage script sources this file, then defines:
#   - variables:  RUN_ID, PTS_REF, STEP, CLUSTER_NAME
#   - gen_config()      writes the per-stage PTS config to ${WORK_DIR}/config.yaml
#   - create_cluster()  creates the stage's Dataproc cluster
# and finally calls:  run_stage
#
# Outputs for every stage live under:
#   gs://ot-team/dochoa/literature_runs/<RUN_ID>/

set -euo pipefail

# ── Shared GCP configuration ────────────────────────────────────────────────
PROJECT="open-targets-eu-dev"
REGION="europe-west1"
ZONE="europe-west1-d"
SERVICE_ACCOUNT="up-airflow-dev@open-targets-eu-dev.iam.gserviceaccount.com"
AUTOSCALING_POLICY="otg-etl-25-secondary"

# ── Shared inputs / outputs ─────────────────────────────────────────────────
INPUT_BASE="gs://open-targets-pipeline-runs/ds/26.03-test5"
OUTPUT_BASE="gs://ot-team/dochoa/literature_runs"
EPMC_BASE="gs://otar025-epmc/ml02"
SPARK_NLP_PACKAGE="com.johnsnowlabs.nlp:spark-nlp_2.12:6.1.3"

# Default PTS git ref installed on the cluster; override via the stage script's
# second positional argument.
PTS_REF="${PTS_REF:-vh-add-literature}"

# ── Generate the Dataproc runner script (mirrors orchestration's) ───────────
gen_runner() {
  cat > "${WORK_DIR}/dataproc_pts_run.py" <<'PYEOF'
#!/opt/conda/default/bin/python
import sys
from pts.core import main

if __name__ == "__main__":
    if sys.argv[0].endswith("-script.pyw"):
        sys.argv[0] = sys.argv[0][:-11]
    elif sys.argv[0].endswith(".exe"):
        sys.argv[0] = sys.argv[0][:-4]
    sys.exit(main())
PYEOF
}

# ── Generate the custom init script (skips the openai-token secret fetch) ───
# The standard init action at
# gs://opentargets-pipelines/up/pts/install_dependencies_on_cluster.sh reads a
# Secret Manager secret 'openai-token' that is not provisioned here, which fails
# cluster creation. The literature steps do not need OpenAI.
gen_init_script() {
  cat > "${WORK_DIR}/install_dependencies_on_cluster.sh" <<'SHEOF'
#!/usr/bin/env bash
set -euo pipefail
set -x

PTS_REF=$(/usr/share/google/get_metadata_value attributes/PTS_REF)
readonly PTS_REF
readonly REPO_URI="https://github.com/opentargets/pts"
DATAPROC_CLUSTER_NAME=$(/usr/share/google/get_metadata_value attributes/dataproc-cluster-name)
readonly DATAPROC_CLUSTER_NAME
echo "export DATAPROC_CLUSTER_NAME=${DATAPROC_CLUSTER_NAME}" >> /etc/profile.d/custom_env_vars.sh

err() { echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*" >&2; exit 1; }
run_with_retry() {
    local -r cmd=("$@")
    for ((i = 0; i < 3; i++)); do
        if "${cmd[@]}"; then return 0; fi
        sleep 5
    done
    err "Failed to run command: ${cmd[*]}"
}

if [[ -z "${PTS_REF}" ]]; then
    echo "ERROR: Must specify PTS_REF metadata key"; exit 1
fi
command -v pip >/dev/null || { apt update && apt install python-pip -y; }
pip install uv
pip uninstall -y pts || true
run_with_retry uv pip install pandas scipy numpy pyarrow fsspec --system --no-break-system-packages --upgrade
run_with_retry uv pip install --no-break-system-packages --system "pts @ git+${REPO_URI}.git@${PTS_REF}"
SHEOF
}

# ── Upload runner + config + init script to GCS ─────────────────────────────
upload_artifacts() {
  echo "Uploading runner, config, and init script to GCS..."
  gcloud storage cp "${WORK_DIR}/dataproc_pts_run.py" "${RUNNER_GCS}" --quiet
  gcloud storage cp "${WORK_DIR}/config.yaml" "${CONFIG_GCS}" --quiet
  gcloud storage cp "${WORK_DIR}/install_dependencies_on_cluster.sh" "${INIT_SCRIPT_GCS}" --quiet
}

# ── Create the cluster only if it does not already exist ────────────────────
# create_cluster() is provided by the stage script.
ensure_cluster() {
  if gcloud dataproc clusters describe "${CLUSTER_NAME}" \
       --project="${PROJECT}" --region="${REGION}" >/dev/null 2>&1; then
    echo "Cluster ${CLUSTER_NAME} already exists -- reusing it."
  else
    echo "Creating Dataproc cluster ${CLUSTER_NAME}..."
    create_cluster
  fi
}

# ── Submit the PySpark job (async) ──────────────────────────────────────────
submit_job() {
  echo "Submitting job (step=${STEP}, config=${CONFIG_GCS})..."
  gcloud dataproc jobs submit pyspark "${RUNNER_GCS}" \
    --project="${PROJECT}" \
    --region="${REGION}" \
    --cluster="${CLUSTER_NAME}" \
    --id="${JOB_ID}" \
    --files="${CONFIG_GCS}" \
    --async \
    -- \
    -s "${STEP}" \
    -c config.yaml \
    -r "${RELEASE_URI}"
}

# ── Print monitoring + teardown info ────────────────────────────────────────
print_monitoring() {
  cat <<EOF

========================================================================
Job submitted (async).
  Run ID      : ${RUN_ID}
  Step        : ${STEP}
  Cluster     : ${CLUSTER_NAME}
  Release URI : ${RELEASE_URI}

Monitor:
  gcloud dataproc jobs wait ${JOB_ID} --project=${PROJECT} --region=${REGION}

Console:
  https://console.cloud.google.com/dataproc/jobs/${JOB_ID}?project=${PROJECT}&region=${REGION}

Spark UI (after the job starts):
  gcloud dataproc clusters describe ${CLUSTER_NAME} --project=${PROJECT} --region=${REGION} --format='value(config.endpointConfig.httpPorts)'

When done with this cluster, tear down:
  gcloud dataproc clusters delete ${CLUSTER_NAME} --project=${PROJECT} --region=${REGION} --quiet
========================================================================
EOF
}

# ── Orchestrate one stage end-to-end ────────────────────────────────────────
# Expects the stage script to have set RUN_ID, PTS_REF, STEP, CLUSTER_NAME and
# to have defined gen_config() and create_cluster().
run_stage() {
  RELEASE_URI="${OUTPUT_BASE}/${RUN_ID}"
  WORK_PREFIX="${RELEASE_URI}/etc/${STEP}"
  RUNNER_GCS="${WORK_PREFIX}/bin/dataproc_pts_run.py"
  CONFIG_GCS="${WORK_PREFIX}/config/config.yaml"
  INIT_SCRIPT_GCS="${WORK_PREFIX}/bin/install_dependencies_on_cluster.sh"
  JOB_ID="pts-${STEP}-${RUN_ID}-$(date +%s)"

  echo "========================================================================"
  echo "Run ID      : ${RUN_ID}"
  echo "PTS ref     : ${PTS_REF}"
  echo "Step        : ${STEP}"
  echo "Cluster     : ${CLUSTER_NAME}"
  echo "Release URI : ${RELEASE_URI}"
  echo "========================================================================"

  WORK_DIR=$(mktemp -d)
  trap 'rm -rf "${WORK_DIR}"' EXIT
  echo "Local workspace: ${WORK_DIR}"

  gen_runner
  gen_init_script
  gen_config

  echo
  echo "── Generated config: ──"
  cat "${WORK_DIR}/config.yaml"
  echo

  upload_artifacts
  ensure_cluster
  submit_job
  print_monitoring
}
```

- [ ] **Step 2: Verify bash syntax**

Run: `bash -n scripts/literature/_common.sh`
Expected: no output, exit 0

- [ ] **Step 3: Commit**

```bash
git add scripts/literature/_common.sh
git commit -m "feat(literature): add shared launcher library _common.sh"
```

---

### Task 9: `scripts/literature/01_launch_ontoma_lut.sh`

**Files:**
- Create: `scripts/literature/01_launch_ontoma_lut.sh`

- [ ] **Step 1: Write the file**

Create `scripts/literature/01_launch_ontoma_lut.sh`:

```bash
#!/usr/bin/env bash
# Stage 01 -- literature_ontoma_lut_generation on a pts-like Dataproc cluster.
#
# Produces the disease/target/drug label lookup table consumed by stage 02.
#
# Usage:
#   ./scripts/literature/01_launch_ontoma_lut.sh [run-id] [pts-ref]
# Defaults:
#   run-id  : run-001
#   pts-ref : vh-add-literature

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/literature/_common.sh
source "${SCRIPT_DIR}/_common.sh"

RUN_ID="${1:-run-001}"
PTS_REF="${2:-vh-add-literature}"
STEP="literature_ontoma_lut_generation"
CLUSTER_NAME="pts-lit-ontoma-${RUN_ID}"

gen_config() {
  cat > "${WORK_DIR}/config.yaml" <<EOF
work_path: /mnt/disks/work
log_level: INFO
pool_size: 8
release_uri: ${RELEASE_URI}
scratchpad: {}

steps:
  ${STEP}:
    - name: pyspark literature ontoma lut generation
      pyspark: literature_ontoma_lut_generation
      source:
        disease_index: ${INPUT_BASE}/output/disease/disease.parquet
        ot_disease_curation: ${INPUT_BASE}/input/ontoma/ot_disease_curation.tsv
        eva_clinvar: ${INPUT_BASE}/input/ontoma/eva_clinvar.txt
        clinvar_xrefs: ${INPUT_BASE}/input/ontoma/clinvar_xrefs.txt
        target_index: ${INPUT_BASE}/output/target
        drug_index: ${INPUT_BASE}/output/drug_molecule
      destination:
        disease_target_drug_label_lut: intermediate/ontoma/disease_target_drug_label_lookup_table.parquet
EOF
}

# pts-like topology: image 2.3, single-node n1-standard-32, full PTS Spark/YARN
# property block including spark-nlp (OnToma's normalisation pipeline needs it).
create_cluster() {
  gcloud dataproc clusters create "${CLUSTER_NAME}" \
    --project="${PROJECT}" \
    --region="${REGION}" \
    --zone="${ZONE}" \
    --image-version=2.3 \
    --single-node \
    --master-machine-type=n1-standard-32 \
    --master-boot-disk-size=128GB \
    --service-account="${SERVICE_ACCOUNT}" \
    --metadata="PTS_REF=${PTS_REF}" \
    --initialization-actions="${INIT_SCRIPT_GCS}" \
    --max-idle=3600s \
    --public-ip-address \
    --properties="^#^yarn:yarn.scheduler.maximum-allocation-mb=25296#yarn:yarn.nodemanager.resource.memory-mb=25296#yarn:yarn.scheduler.maximum-allocation-vcores=10#yarn:yarn.nodemanager.resource.cpu-vcores=10#spark:spark.executor.memory=14894m#spark:spark.executor.cores=5#spark:spark.memory.fraction=0.6#spark:spark.driver.memory=14894m#spark:spark.driver.cores=5#spark:spark.executor.memoryOverheadFactor=0.1#spark:spark.dynamicAllocation.enabled=true#spark:spark.dynamicAllocation.maxExecutors=6#spark:spark.sql.adaptive.enabled=true#spark:spark.memory.storageFraction=0.5#spark:spark.serializer=org.apache.spark.serializer.KryoSerializer#spark:spark.executor.extraJavaOptions=-XX:+UseG1GC#spark:spark.jars.packages=${SPARK_NLP_PACKAGE}"
}

run_stage
```

- [ ] **Step 2: Verify bash syntax**

Run: `bash -n scripts/literature/01_launch_ontoma_lut.sh`
Expected: no output, exit 0

- [ ] **Step 3: Make executable**

Run: `chmod +x scripts/literature/01_launch_ontoma_lut.sh`

- [ ] **Step 4: Commit**

```bash
git add scripts/literature/01_launch_ontoma_lut.sh
git commit -m "feat(literature): add stage 01 ontoma_lut launcher"
```

---

### Task 10: `scripts/literature/02_launch_publication_match.sh`

**Files:**
- Create: `scripts/literature/02_launch_publication_match.sh`

- [ ] **Step 1: Write the file**

Create `scripts/literature/02_launch_publication_match.sh`:

```bash
#!/usr/bin/env bash
# Stage 02 -- literature_publication_match (collapsed publication + match) on a
# pts_openfda-like Dataproc cluster with spark-nlp.
#
# Reads the ontoma LUT from stage 01. Restricts EPMC input to March 2026
# (date_prefix 2026_03) for the first test run -- drop the date_prefix setting
# to run the full dataset.
#
# Usage:
#   ./scripts/literature/02_launch_publication_match.sh [run-id] [pts-ref]
# Defaults:
#   run-id  : run-001
#   pts-ref : vh-add-literature

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/literature/_common.sh
source "${SCRIPT_DIR}/_common.sh"

RUN_ID="${1:-run-001}"
PTS_REF="${2:-vh-add-literature}"
STEP="literature_publication_match"
CLUSTER_NAME="pts-lit-pubmatch-${RUN_ID}"

# Test-run tunables -- edit between runs after watching the Spark UI.
DATE_PREFIX="2026_03"
REPARTITION="1000"

gen_config() {
  cat > "${WORK_DIR}/config.yaml" <<EOF
work_path: /mnt/disks/work
log_level: INFO
pool_size: 8
release_uri: ${RELEASE_URI}
scratchpad: {}

steps:
  ${STEP}:
    - name: pyspark literature publication match
      pyspark: literature_publication_match
      source:
        pub_id_lut: ${INPUT_BASE}/input/literature/PMID_PMCID_DOI.csv.gz
        epmc_publication: ${EPMC_BASE}
        ontoma_disease_target_drug_label_lut: ${RELEASE_URI}/intermediate/ontoma/disease_target_drug_label_lookup_table.parquet
      destination:
        match_valid: intermediate/literature/match
        match_failed: excluded/literature/match
      settings:
        date_prefix: '${DATE_PREFIX}'
        repartition: ${REPARTITION}
EOF
}

# pts_openfda-like topology: image 2.2, 1 master + 2 workers n1-standard-8,
# autoscaling -- PLUS spark-nlp, required by literature_match's map_labels
# (OnToma.map_entities runs the spark-nlp normalisation pipeline).
create_cluster() {
  gcloud dataproc clusters create "${CLUSTER_NAME}" \
    --project="${PROJECT}" \
    --region="${REGION}" \
    --zone="${ZONE}" \
    --image-version=2.2 \
    --num-masters=1 \
    --num-workers=2 \
    --master-machine-type=n1-standard-8 \
    --master-boot-disk-size=512GB \
    --worker-machine-type=n1-standard-8 \
    --worker-boot-disk-size=128GB \
    --autoscaling-policy="${AUTOSCALING_POLICY}" \
    --service-account="${SERVICE_ACCOUNT}" \
    --metadata="PTS_REF=${PTS_REF}" \
    --initialization-actions="${INIT_SCRIPT_GCS}" \
    --max-idle=3600s \
    --public-ip-address \
    --properties="spark:spark.jars.packages=${SPARK_NLP_PACKAGE}"
}

run_stage
```

- [ ] **Step 2: Verify bash syntax**

Run: `bash -n scripts/literature/02_launch_publication_match.sh`
Expected: no output, exit 0

- [ ] **Step 3: Make executable**

Run: `chmod +x scripts/literature/02_launch_publication_match.sh`

- [ ] **Step 4: Commit**

```bash
git add scripts/literature/02_launch_publication_match.sh
git commit -m "feat(literature): add stage 02 publication_match launcher"
```

---

### Task 11: `scripts/literature/03a_launch_entity_lut.sh`

**Files:**
- Create: `scripts/literature/03a_launch_entity_lut.sh`

- [ ] **Step 1: Write the file**

Create `scripts/literature/03a_launch_entity_lut.sh`:

```bash
#!/usr/bin/env bash
# Stage 03a -- literature_entity_lut on the shared stage-3 Dataproc cluster.
#
# Reads the valid matches from stage 02. The three 03* scripts share one
# cluster (pts-lit-stage3-<run-id>); whichever runs first creates it.
#
# Usage:
#   ./scripts/literature/03a_launch_entity_lut.sh [run-id] [pts-ref]
# Defaults:
#   run-id  : run-001
#   pts-ref : vh-add-literature

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/literature/_common.sh
source "${SCRIPT_DIR}/_common.sh"

RUN_ID="${1:-run-001}"
PTS_REF="${2:-vh-add-literature}"
STEP="literature_entity_lut"
CLUSTER_NAME="pts-lit-stage3-${RUN_ID}"

gen_config() {
  cat > "${WORK_DIR}/config.yaml" <<EOF
work_path: /mnt/disks/work
log_level: INFO
pool_size: 8
release_uri: ${RELEASE_URI}
scratchpad: {}

steps:
  ${STEP}:
    - name: pyspark literature entity lut
      pyspark: literature_entity_lut
      source:
        matches: ${RELEASE_URI}/intermediate/literature/match
      destination:
        literature_entity_lut: output/literature_entity_lut
EOF
}

# pts_openfda-like topology, no spark-nlp -- shared by all three 03* stages.
create_cluster() {
  gcloud dataproc clusters create "${CLUSTER_NAME}" \
    --project="${PROJECT}" \
    --region="${REGION}" \
    --zone="${ZONE}" \
    --image-version=2.2 \
    --num-masters=1 \
    --num-workers=2 \
    --master-machine-type=n1-standard-8 \
    --master-boot-disk-size=512GB \
    --worker-machine-type=n1-standard-8 \
    --worker-boot-disk-size=128GB \
    --autoscaling-policy="${AUTOSCALING_POLICY}" \
    --service-account="${SERVICE_ACCOUNT}" \
    --metadata="PTS_REF=${PTS_REF}" \
    --initialization-actions="${INIT_SCRIPT_GCS}" \
    --max-idle=3600s \
    --public-ip-address
}

run_stage
```

- [ ] **Step 2: Verify bash syntax**

Run: `bash -n scripts/literature/03a_launch_entity_lut.sh`
Expected: no output, exit 0

- [ ] **Step 3: Make executable**

Run: `chmod +x scripts/literature/03a_launch_entity_lut.sh`

- [ ] **Step 4: Commit**

```bash
git add scripts/literature/03a_launch_entity_lut.sh
git commit -m "feat(literature): add stage 03a entity_lut launcher"
```

---

### Task 12: `scripts/literature/03b_launch_embedding_vector.sh`

**Files:**
- Create: `scripts/literature/03b_launch_embedding_vector.sh`

- [ ] **Step 1: Write the file**

Create `scripts/literature/03b_launch_embedding_vector.sh`:

```bash
#!/usr/bin/env bash
# Stage 03b -- literature_embedding then literature_vector on the shared
# stage-3 Dataproc cluster. The two tasks run in one Otter step; vector
# `requires` embedding, so they run sequentially.
#
# Reads the valid matches from stage 02. The three 03* scripts share one
# cluster (pts-lit-stage3-<run-id>); whichever runs first creates it.
#
# Usage:
#   ./scripts/literature/03b_launch_embedding_vector.sh [run-id] [pts-ref]
# Defaults:
#   run-id  : run-001
#   pts-ref : vh-add-literature

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/literature/_common.sh
source "${SCRIPT_DIR}/_common.sh"

RUN_ID="${1:-run-001}"
PTS_REF="${2:-vh-add-literature}"
STEP="literature_embedding_vector"
CLUSTER_NAME="pts-lit-stage3-${RUN_ID}"

gen_config() {
  cat > "${WORK_DIR}/config.yaml" <<EOF
work_path: /mnt/disks/work
log_level: INFO
pool_size: 8
release_uri: ${RELEASE_URI}
scratchpad: {}

steps:
  ${STEP}:
    - name: pyspark literature embedding
      pyspark: literature_embedding
      source:
        matches: ${RELEASE_URI}/intermediate/literature/match
      destination:
        model: etc/model/w2v_model
      properties:
        spark.sql.shuffle.partitions: '800'
    - name: pyspark literature vector
      pyspark: literature_vector
      requires:
        - pyspark literature embedding
      source:
        model: etc/model/w2v_model
      destination:
        vectors: output/literature_vector
EOF
}

# pts_openfda-like topology, no spark-nlp -- shared by all three 03* stages.
create_cluster() {
  gcloud dataproc clusters create "${CLUSTER_NAME}" \
    --project="${PROJECT}" \
    --region="${REGION}" \
    --zone="${ZONE}" \
    --image-version=2.2 \
    --num-masters=1 \
    --num-workers=2 \
    --master-machine-type=n1-standard-8 \
    --master-boot-disk-size=512GB \
    --worker-machine-type=n1-standard-8 \
    --worker-boot-disk-size=128GB \
    --autoscaling-policy="${AUTOSCALING_POLICY}" \
    --service-account="${SERVICE_ACCOUNT}" \
    --metadata="PTS_REF=${PTS_REF}" \
    --initialization-actions="${INIT_SCRIPT_GCS}" \
    --max-idle=3600s \
    --public-ip-address
}

run_stage
```

- [ ] **Step 2: Verify bash syntax**

Run: `bash -n scripts/literature/03b_launch_embedding_vector.sh`
Expected: no output, exit 0

- [ ] **Step 3: Make executable**

Run: `chmod +x scripts/literature/03b_launch_embedding_vector.sh`

- [ ] **Step 4: Commit**

```bash
git add scripts/literature/03b_launch_embedding_vector.sh
git commit -m "feat(literature): add stage 03b embedding_vector launcher"
```

---

### Task 13: `scripts/literature/03c_launch_cooccurrence_evidence.sh`

**Files:**
- Create: `scripts/literature/03c_launch_cooccurrence_evidence.sh`

- [ ] **Step 1: Write the file**

Create `scripts/literature/03c_launch_cooccurrence_evidence.sh`:

```bash
#!/usr/bin/env bash
# Stage 03c -- literature_cooccurrence_evidence (collapsed cooccurrence +
# evidence) on the shared stage-3 Dataproc cluster.
#
# Reads the valid matches from stage 02; writes both the intermediate
# cooccurrence parquet and the EPMC evidence. The three 03* scripts share one
# cluster (pts-lit-stage3-<run-id>); whichever runs first creates it.
#
# Usage:
#   ./scripts/literature/03c_launch_cooccurrence_evidence.sh [run-id] [pts-ref]
# Defaults:
#   run-id  : run-001
#   pts-ref : vh-add-literature

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/literature/_common.sh
source "${SCRIPT_DIR}/_common.sh"

RUN_ID="${1:-run-001}"
PTS_REF="${2:-vh-add-literature}"
STEP="literature_cooccurrence_evidence"
CLUSTER_NAME="pts-lit-stage3-${RUN_ID}"

gen_config() {
  cat > "${WORK_DIR}/config.yaml" <<EOF
work_path: /mnt/disks/work
log_level: INFO
pool_size: 8
release_uri: ${RELEASE_URI}
scratchpad: {}

steps:
  ${STEP}:
    - name: pyspark literature cooccurrence evidence
      pyspark: literature_cooccurrence_evidence
      source:
        match: ${RELEASE_URI}/intermediate/literature/match
      destination:
        cooccurrence: intermediate/literature/cooccurrence
        evidence: intermediate/evidence/literature_epmc
EOF
}

# pts_openfda-like topology, no spark-nlp -- shared by all three 03* stages.
create_cluster() {
  gcloud dataproc clusters create "${CLUSTER_NAME}" \
    --project="${PROJECT}" \
    --region="${REGION}" \
    --zone="${ZONE}" \
    --image-version=2.2 \
    --num-masters=1 \
    --num-workers=2 \
    --master-machine-type=n1-standard-8 \
    --master-boot-disk-size=512GB \
    --worker-machine-type=n1-standard-8 \
    --worker-boot-disk-size=128GB \
    --autoscaling-policy="${AUTOSCALING_POLICY}" \
    --service-account="${SERVICE_ACCOUNT}" \
    --metadata="PTS_REF=${PTS_REF}" \
    --initialization-actions="${INIT_SCRIPT_GCS}" \
    --max-idle=3600s \
    --public-ip-address
}

run_stage
```

- [ ] **Step 2: Verify bash syntax**

Run: `bash -n scripts/literature/03c_launch_cooccurrence_evidence.sh`
Expected: no output, exit 0

- [ ] **Step 3: Make executable**

Run: `chmod +x scripts/literature/03c_launch_cooccurrence_evidence.sh`

- [ ] **Step 4: Commit**

```bash
git add scripts/literature/03c_launch_cooccurrence_evidence.sh
git commit -m "feat(literature): add stage 03c cooccurrence_evidence launcher"
```

---

## Phase 3 — Final verification

### Task 14: Full test suite + lint

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest test --doctest-modules src/pts`
Expected: PASS — all tests pass, including the new `test_literature_publication_match.py` (7) and `test_literature_cooccurrence_evidence.py` (3).

- [ ] **Step 2: Lint the whole tree**

Run: `uv run ruff check`
Expected: `All checks passed!`

- [ ] **Step 3: Verify all launcher scripts parse**

Run: `for s in scripts/literature/*.sh; do bash -n "$s" && echo "ok: $s"; done`
Expected: `ok:` line for each of the 6 scripts.

- [ ] **Step 4: If anything fails, fix and re-run**

Do not proceed past a failing suite. Fix the offending task's code, re-run Steps 1-3.

---

## Operator runbook (post-implementation, not a code task)

Once merged, run the test pipeline in dependency order from the repo root:

```bash
./scripts/literature/01_launch_ontoma_lut.sh run-001
# wait for the stage 01 job to succeed (gcloud dataproc jobs wait ...)
./scripts/literature/02_launch_publication_match.sh run-001
# wait for the stage 02 job to succeed
# then the three stage-3 scripts, in parallel if desired:
./scripts/literature/03a_launch_entity_lut.sh run-001
./scripts/literature/03b_launch_embedding_vector.sh run-001
./scripts/literature/03c_launch_cooccurrence_evidence.sh run-001
```

Each script prints its own monitor and teardown commands. Outputs land under
`gs://ot-team/dochoa/literature_runs/run-001/`. Tune cluster topology and the
stage-02 `REPARTITION` value between runs after inspecting the Spark UI.

---

## Self-Review Notes

- **Spec coverage:** collapsed modules (Tasks 1-5), config path fix (Task 6), config step entries (Task 7), `_common.sh` (Task 8), the five stage scripts incl. spark-nlp on stage 02 and shared stage-3 cluster (Tasks 9-13), output location `gs://ot-team/dochoa/literature_runs/` (Task 8 `OUTPUT_BASE`), date_prefix wiring (Task 10), repartition (Tasks 2-3, 10). All design sections map to a task.
- **Discovered during planning:** the `Cooccurrence` schema (`mappedId1`/`mappedId2`/`evidenceScore`) does not match what `evidence_epmc._compute_evidence` consumes (`keywordId1`/`keywordId2`/`evidence_score`). Task 4's `_adapt_cooccurrence_for_evidence` bridges this inside the collapsed module; the pre-existing non-collapsed `evidence_epmc` step has the same latent mismatch but fixing that is out of scope.
- **TDD note:** the two `Session`-creating entry points (Tasks 3, 5) have no unit tests, matching the repo convention (`evidence_epmc.py`'s entry point is untested; only pure helpers are). They are import-checked here and validated by the Dataproc run.
- **Type consistency:** `_epmc_read_path(epmc_path, kind, date_prefix)`, `_maybe_repartition(df, repartition)`, `_read_publications(session, epmc_path, pub_id_lut, date_prefix, repartition)`, `_adapt_cooccurrence_for_evidence(cooccurrences)` — signatures consistent across all references.
