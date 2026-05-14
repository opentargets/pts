# Literature Pipeline — Dataproc Launcher Design

## Summary

Run the literature pipeline steps from the `vh-add-literature` branch on Dataproc
via per-stage launcher scripts, modeled on `scripts/launch_association_benchmark.sh`.

The work has two parts:

1. **Code changes** — two new "collapsed" PySpark modules so adjacent steps can run
   as a single job:
   - `literature_publication_match` — collapses `literature_publication` +
     `literature_match`, *without* writing the intermediate publication parquet.
   - `literature_cooccurrence_evidence` — collapses `literature_cooccurrence` +
     `evidence_epmc`, *writing* the intermediate cooccurrence parquet.
2. **Launcher scripts** — `scripts/literature/`, one script per pipeline stage, that
   create a Dataproc cluster, generate a per-stage config, submit the job, and print
   monitoring/teardown commands.

This is a **first test run**: input is restricted to March 2026 EPMC documents, and
upstream dependencies (disease/target/drug indexes) are reused from an existing
release rather than recomputed.

## Dependency graph

```
literature_ontoma_lut_generation
        │
        ▼
literature_publication ──► literature_match     [collapsed: literature_publication_match]
        │
        ├─► literature_entity_lut
        ├─► literature_embedding ──► literature_vector
        └─► literature_cooccurrence ──► evidence_epmc   [collapsed: literature_cooccurrence_evidence]
```

After `literature_match`, the pipeline fans out into three independent streams that
can run in parallel.

## Decisions (from brainstorming)

| Topic | Decision |
|-------|----------|
| Collapse `publication + match` | New combined PySpark module; no intermediate publication parquet written. |
| Collapse `cooccurrence + evidence` | New combined PySpark module; intermediate cooccurrence parquet **is** written. |
| Upstream deps (disease/target/drug) | Reuse from `gs://open-targets-pipeline-runs/ds/26.03-test5/`; not recomputed. |
| March-2026 input filter | Optional `settings.date_prefix` in the combined module. Set `'2026_03'` for the test run; omit for full data later. |
| Launcher shape | Separate per-stage scripts, invoked manually so cluster sizing can be tuned between stages. |
| Module structure | Standalone combined modules (Approach A); existing single-step modules left untouched. |
| Output location | `gs://ot-team/dochoa/literature_runs/<RUN_ID>/...` |

## Code changes

### `src/pts/pyspark/literature_publication_match.py` (new)

Collapses `literature_publication` + `literature_match` into one job. The publication
DataFrame is held in memory and fed straight into the match logic — the publication
parquet is never written.

**Inputs (`source`):**
- `pub_id_lut` — `input/literature/PMID_PMCID_DOI.csv.gz`
- `epmc_publication` — `gs://otar025-epmc/ml02`
- `ontoma_disease_target_drug_label_lut` — output of stage 01

**Outputs (`destination`):**
- `match_valid` — `intermediate/literature/match`
- `match_failed` — `excluded/literature/match`

**Settings:**
- `date_prefix` (optional) — e.g. `'2026_03'`. When set, the EPMC read path becomes
  `{epmc_path}/{kind}/{date_prefix}*/**/*.jsonl`; when omitted, `{epmc_path}/{kind}/**/*.jsonl`
  (all dates).
- `repartition` (optional, int) — partition count applied immediately after the
  scattered jsonl read, since the input is spread across many small day-folders.

**Logic:**
1. Load `pub_id_lut` via `PublicationIdLUT.from_csv(...)`.
2. Read EPMC `fulltext` and `abstract` jsonl with the (optionally date-globbed) path —
   a ~6-line reimplementation of `EPMCPublication._read_in_with_schema` so the glob can
   be parameterised (the library hardcodes it).
3. Apply `.repartition(repartition)` if configured.
4. Reuse the library's `EPMCPublication` static methods
   (`_annotate_fulltexts_with_pmid`, `_merge_abstracts_with_fulltexts`,
   `_get_most_recent_publications`) to build the publication DataFrame in memory.
5. Feed into `Publication(df).extract_matches().map_labels(...).disambiguate(...)`,
   mirroring `literature_match.py`.
6. Write `match_valid` and `match_failed` only.

### `src/pts/pyspark/literature_cooccurrence_evidence.py` (new)

Collapses `literature_cooccurrence` + `evidence_epmc` into one job. Unlike the
publication+match collapse, the intermediate cooccurrence parquet **is** written.

**Inputs (`source`):**
- `match` — output of stage 02 (`intermediate/literature/match`)

**Outputs (`destination`):**
- `cooccurrence` — `intermediate/literature/cooccurrence`
- `evidence` — `intermediate/evidence/literature_epmc`

**Logic:**
1. Load `match`.
2. Compute cooccurrences via `MatchMapped(match).generate_target_disease_cooccurrences()`;
   write to `cooccurrence`.
3. Re-read the `cooccurrence` parquet (clean lineage cut on a memory-tight cluster),
   compute evidence via the existing `_compute_evidence` helper from `evidence_epmc.py`
   (imported, not duplicated), write to `evidence`.

### `config.yaml`

Add two step entries — `literature_publication_match` and
`literature_cooccurrence_evidence` — as the canonical source/destination wiring, so the
collapsed steps are also runnable locally via `uv run pts --step`. The launcher scripts
generate their own per-stage configs inline (matching the association template) but with
values kept consistent with these entries.

The existing single-step entries (`literature_publication`, `literature_match`,
`literature_cooccurrence`, `literature`) are left untouched.

## Launcher scripts — `scripts/literature/`

| Script | Job(s) | Cluster |
|--------|--------|---------|
| `_common.sh` | Shared helpers: `ensure_cluster` (create only if absent), runner-script + custom init-script generation, config upload, monitoring output. Sourced by every stage script. | — |
| `01_launch_ontoma_lut.sh` | `literature_ontoma_lut_generation` | own `pts`-like cluster |
| `02_launch_publication_match.sh` | `literature_publication_match` (`date_prefix: 2026_03`) | own `pts_openfda`-like cluster |
| `03a_launch_entity_lut.sh` | `literature_entity_lut` | shared stage-3 cluster |
| `03b_launch_embedding_vector.sh` | `literature_embedding` → `literature_vector` (sequential) | shared stage-3 cluster |
| `03c_launch_cooccurrence_evidence.sh` | `literature_cooccurrence_evidence` | shared stage-3 cluster |

Each stage script:
1. Generates the Dataproc runner (`dataproc_pts_run.py`, mirrors orchestration's).
2. Generates a per-stage PTS config inline (single step, absolute GCS input paths,
   relative outputs resolved against `release_uri`).
3. Generates the custom init script that installs the PTS dependency from
   `PTS_REF` and skips the `openai-token` Secret Manager fetch (the workaround already
   used in `launch_association_benchmark.sh`).
4. Uploads runner + config + init script to GCS under the run's `etc/` prefix.
5. Ensures its cluster exists (`ensure_cluster` — creates it if absent, reuses it
   otherwise).
6. Submits the job `--async`.
7. Prints monitoring and teardown commands.

**Shared stage-3 cluster:** the three `03*` scripts all target one `pts_openfda`-like
cluster (a single fixed name per `RUN_ID`). Whichever script runs first creates it; the
others reuse it. They are mutually independent and may be launched in parallel —
Dataproc runs the concurrent jobs on the shared cluster.

**Dependency enforcement:** there is no DAG engine. Stages are run in order by the
operator (01 → 02 → 03*). All stages share a `RUN_ID`; each stage writes under
`gs://ot-team/dochoa/literature_runs/<RUN_ID>/...` and later stages read the prior
stage's output by `RUN_ID`.

`--max-idle=3600s` plus the printed teardown commands handle cluster cleanup.

## Cluster topologies

Starting points taken from the orchestration repo `clusters.yaml`. Stage scripts expose
worker count, machine type, and a `CLUSTER_PROPS` override as easily-edited variables,
so topology can be tuned after observing the Spark UI.

### `pts`-like (stage 01)
- Image 2.3, single-node (1 master, 0 workers), `n1-standard-32`, 128GB disk.
- Full PTS YARN/Spark property block, **including** `spark.jars.packages:
  com.johnsnowlabs.nlp:spark-nlp_2.12:6.1.3` (OnToma LUT generation may need it).

### `pts_openfda`-like (stage 02, and the shared stage-3 cluster)
- Image 2.2, 1 master + 2 workers, worker `n1-standard-8`, master 512GB / worker 128GB.
- Autoscaling policy `otg-etl-25-secondary`.
- No special Spark properties by default. These are the topologies to tune after
  watching partitioning behaviour on the first run.
- Stage 02 and the shared stage-3 cluster are separate clusters with the same starting
  topology; they can be sized independently.
- The `literature_embedding` job (run by `03b`) carries through
  `spark.sql.shuffle.partitions: '800'` as a per-job Spark property at submit time
  (from the existing `literature_embedding` config entry) rather than as a
  cluster-level property, so it does not affect the other stage-3 jobs.

## Data flow

### Inputs (verified present)
- `gs://open-targets-pipeline-runs/ds/26.03-test5/output/disease` (`disease.parquet`)
- `gs://open-targets-pipeline-runs/ds/26.03-test5/output/target`
- `gs://open-targets-pipeline-runs/ds/26.03-test5/output/drug_molecule`
- `gs://open-targets-pipeline-runs/ds/26.03-test5/input/ontoma/{ot_disease_curation.tsv,eva_clinvar.txt,clinvar_xrefs.txt}`
- `gs://open-targets-pipeline-runs/ds/26.03-test5/input/literature/PMID_PMCID_DOI.csv.gz`
- `gs://otar025-epmc/ml02` — EPMC publications (`abstract/` and `fulltext/`, with
  `YYYY_MM_DD` day-folders). Test run reads `2026_03*` only.

### Outputs
All under `gs://ot-team/dochoa/literature_runs/<RUN_ID>/`:

| Stage | Output paths |
|-------|--------------|
| 01 | `intermediate/ontoma/disease_target_drug_label_lookup_table.parquet` |
| 02 | `intermediate/literature/match`, `excluded/literature/match` |
| 03a | `output/literature_entity_lut` |
| 03b | `etc/model/w2v_model`, `output/literature_vector` |
| 03c | `intermediate/literature/cooccurrence`, `intermediate/evidence/literature_epmc` |

## Open notes / risks

- **spark-nlp on stages 02/03:** assumed not required — the migrated logic
  (`extract_matches`, `map_labels`, `disambiguate`, cooccurrence/evidence) is pure
  Spark SQL; the library only pulls spark-nlp for the non-migrated grounding steps.
  Fallback: add the `spark.jars.packages` property to the affected stage script (kept
  one edit away).
- **Cooccurrence re-read vs. cache:** stage 03c re-reads the cooccurrence parquet after
  writing it, for a clean lineage cut on a memory-constrained cluster. Can switch to
  `.cache()` if I/O turns out to dominate.
- **Repartition tuning:** the `repartition` value for stage 02 is a first guess to be
  refined after observing input partitioning on the first run.
- **`pts_openfda` topology is a starting point**, not a validated size — the whole point
  of separate per-stage scripts is to resize between runs.
- **Existing config path inconsistency:** the `literature` step group reads matches from
  `intermediate/literature_match`, while `literature_match` writes to
  `intermediate/literature/match`. The launcher sidesteps this entirely — it generates
  per-stage configs with absolute GCS paths, pointing stages 03a/03b/03c at stage 02's
  actual output. Reconciling the in-repo config is out of scope here.

## Files to create / modify

**New:**
- `src/pts/pyspark/literature_publication_match.py`
- `src/pts/pyspark/literature_cooccurrence_evidence.py`
- `scripts/literature/_common.sh`
- `scripts/literature/01_launch_ontoma_lut.sh`
- `scripts/literature/02_launch_publication_match.sh`
- `scripts/literature/03a_launch_entity_lut.sh`
- `scripts/literature/03b_launch_embedding_vector.sh`
- `scripts/literature/03c_launch_cooccurrence_evidence.sh`
- `test/test_literature_publication_match.py`
- `test/test_literature_cooccurrence_evidence.py`

**Modified:**
- `config.yaml` — add `literature_publication_match` and
  `literature_cooccurrence_evidence` step entries.

## Testing

Unit tests for the new modules' helper logic, using the existing `spark` fixture from
`test/conftest.py`:
- `test_literature_publication_match.py` — date-prefix glob construction, repartition
  wiring, the in-memory publication→match chaining.
- `test_literature_cooccurrence_evidence.py` — cooccurrence write + evidence
  re-read/compute chaining.

The launcher scripts are validated by running the actual test pipeline on Dataproc
(stage 01 → 02 → 03a/b/c), not by unit tests.
