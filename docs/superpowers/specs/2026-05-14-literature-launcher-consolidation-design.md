# Literature Launcher Consolidation Design

## Summary

Replace the six files under `scripts/literature/` (`_common.sh` plus the five
per-stage launchers) with a single self-contained `scripts/literature/launch_literature.sh`
that creates **one** Dataproc cluster and runs the whole literature pipeline
against it as **six separate `pts` step-jobs**, orchestrated by the script in
dependency order.

This supersedes the multi-cluster launcher architecture in
`2026-05-14-literature-dataproc-launcher-design.md`. The collapsed PySpark
modules (`literature_publication_match`, `literature_cooccurrence_evidence`),
the `config.yaml` step entries, and the two bug fixes already on the branch are
unchanged.

## Motivation

The per-stage launcher split (5 scripts, 3 cluster topologies) existed so each
stage could be run and monitored in isolation while validating that the PTS
steps behave correctly. That validation is done — a full `run-001` completed
end to end, surfacing and fixing two real bugs (the EPMC glob and the
`keywordId`/`mappedId` schema mismatch). The split has served its purpose; the
operational complexity it carries is no longer worth it.

## Decisions (from brainstorming)

| Topic | Decision |
|-------|----------|
| Consolidation level | One cluster, one launcher script — but **keep six separate step-jobs**, not one Otter `requires` DAG. |
| Orchestration | The bash launcher submits jobs and waits between dependent stages. No Otter `requires` edges. |
| Cluster topology | `pts_openfda`-like + autoscaling + spark-nlp (single shared cluster). |
| Launcher shape | One script runs the full sequence end to end. |
| `literature_embedding` / `literature_vector` | Two separate step-jobs (not a step-group with an internal `requires`). |
| Failure behaviour | Explicit per-job exit-code checks; on failure, print the failed job's logs URL and stop, leaving the cluster up. |
| Cluster teardown | Not auto-deleted on success; `--max-idle=3600s` + a printed teardown command. |

## Architecture

### Single launcher script — `scripts/literature/launch_literature.sh`

Self-contained (no shared library — a `_common.sh` made sense for five scripts,
not for one). Usage unchanged from the old per-stage scripts:

```
./scripts/literature/launch_literature.sh [run-id] [pts-ref]
```

Defaults: `run-id` = `run-001`, `pts-ref` = `vh-add-literature`.

Responsibilities, in order:

1. Generate the Dataproc runner (`dataproc_pts_run.py`) and the custom init
   script (installs PTS from `git+…@PTS_REF`, skips the `openai-token` secret
   fetch) — same content as the current launchers.
2. Generate **one** `config.yaml` with six independent `steps:` entries (see
   below).
3. Upload runner + config + init script to
   `gs://ot-team/dochoa/literature_runs/<run-id>/etc/`.
4. Create one cluster (see Cluster section).
5. Submit the six step-jobs in dependency order, waiting between dependent
   stages (see Orchestration).
6. Print monitoring + teardown commands.

### Orchestration sequence

```
1. submit literature_ontoma_lut_generation      → wait DONE
2. submit literature_publication_match          → wait DONE
3. submit literature_entity_lut,
          literature_embedding,
          literature_cooccurrence_evidence       (async — run in parallel)
4. wait literature_embedding DONE → submit literature_vector
5. wait literature_entity_lut + literature_vector + literature_cooccurrence_evidence
6. print teardown command
```

Each job is `gcloud dataproc jobs submit pyspark "${RUNNER_GCS}" --cluster=<one
cluster> --files=config.yaml --async -- -s <step> -c config.yaml -r <release_uri>`,
followed (for the blocking points) by `gcloud dataproc jobs wait <job-id>`.

- The launcher captures each submitted job id.
- After each `gcloud dataproc jobs wait`, it checks the **exit code explicitly**
  (not via bare `set -e` on the pipeline) so it can print a useful message:
  the failed step name, the job console URL, and the `gcloud dataproc jobs wait`
  command to re-inspect.
- On failure the launcher stops and exits non-zero, **leaving the cluster up**.
  Sibling parallel jobs already submitted are left to finish (or are cleaned up
  by `--max-idle`). The operator can inspect logs / the Spark UI, fix, and
  re-run (or manually submit individual steps to the still-running cluster).
- `literature_vector` depends only on `literature_embedding`; `literature_entity_lut`
  and `literature_cooccurrence_evidence` keep running in parallel while
  embedding → vector proceeds.

### The generated `config.yaml` — six step entries

One config file, six independent step groups (not a single `requires` DAG):

```yaml
work_path: /mnt/disks/work
log_level: INFO
pool_size: 8
release_uri: gs://ot-team/dochoa/literature_runs/<run-id>
scratchpad: {}

steps:
  literature_ontoma_lut_generation:
    - name: pyspark literature ontoma lut generation
      pyspark: literature_ontoma_lut_generation
      source:
        disease_index: gs://open-targets-pipeline-runs/ds/26.03-test5/output/disease/disease.parquet
        ot_disease_curation: gs://open-targets-pipeline-runs/ds/26.03-test5/input/ontoma/ot_disease_curation.tsv
        eva_clinvar: gs://open-targets-pipeline-runs/ds/26.03-test5/input/ontoma/eva_clinvar.txt
        clinvar_xrefs: gs://open-targets-pipeline-runs/ds/26.03-test5/input/ontoma/clinvar_xrefs.txt
        target_index: gs://open-targets-pipeline-runs/ds/26.03-test5/output/target
        drug_index: gs://open-targets-pipeline-runs/ds/26.03-test5/output/drug_molecule
      destination:
        disease_target_drug_label_lut: intermediate/ontoma/disease_target_drug_label_lookup_table.parquet

  literature_publication_match:
    - name: pyspark literature publication match
      pyspark: literature_publication_match
      source:
        pub_id_lut: gs://open-targets-pipeline-runs/ds/26.03-test5/input/literature/PMID_PMCID_DOI.csv.gz
        epmc_publication: gs://otar025-epmc/ml02
        ontoma_disease_target_drug_label_lut: intermediate/ontoma/disease_target_drug_label_lookup_table.parquet
      destination:
        match_valid: intermediate/literature/match
        match_failed: excluded/literature/match
      settings:
        date_prefix: '2026_03'
        repartition: 1000

  literature_entity_lut:
    - name: pyspark literature entity lut
      pyspark: literature_entity_lut
      source:
        matches: intermediate/literature/match
      destination:
        literature_entity_lut: output/literature_entity_lut

  literature_embedding:
    - name: pyspark literature embedding
      pyspark: literature_embedding
      source:
        matches: intermediate/literature/match
      destination:
        model: etc/model/w2v_model
      properties:
        spark.sql.shuffle.partitions: '800'

  literature_vector:
    - name: pyspark literature vector
      pyspark: literature_vector
      source:
        model: etc/model/w2v_model
      destination:
        vectors: output/literature_vector

  literature_cooccurrence_evidence:
    - name: pyspark literature cooccurrence evidence
      pyspark: literature_cooccurrence_evidence
      source:
        match: intermediate/literature/match
      destination:
        cooccurrence: intermediate/literature/cooccurrence
        evidence: intermediate/evidence/literature_epmc
```

**Path handling:** all six jobs run against the same `-r release_uri`, so
intra-pipeline handoffs use **relative paths** (`intermediate/ontoma/…`,
`intermediate/literature/match`, `etc/model/w2v_model`) — each job resolves them
against `release_uri` via Otter's `make_absolute`. Only the external inputs
(disease / target / drug from the `26.03-test5` release, `pub_id_lut`,
`epmc_publication`) are absolute `gs://…`.

This differs from the per-stage launchers, which used absolute
`${RELEASE_URI}/…` paths for cross-stage handoffs because each stage was a
separate run with its own config.

### Cluster

One cluster, `pts-literature-<run-id>`:

- Image 2.2, 1 master + 2 workers `n1-standard-8`, master 512GB / worker 128GB.
- Autoscaling policy `otg-etl-25-secondary` — absorbs the heavier
  `ontoma_lut` / `publication_match` load by adding secondary workers, scales
  back down for the lighter stage-3 jobs.
- `spark.jars.packages=com.johnsnowlabs.nlp:spark-nlp_2.12:6.1.3` — required by
  `literature_ontoma_lut_generation` and `literature_publication_match` (OnToma's
  spark-nlp normalisation pipeline). Harmless for the stage-3 jobs.
- Init action: the generated custom `install_dependencies_on_cluster.sh`.
- `--max-idle=3600s`, `--public-ip-address`.
- Not auto-deleted on success — the printed teardown command plus `--max-idle`
  handle cleanup, while leaving the Spark UI reachable for inspection.

Run-data context informing the topology: on the validated `run-001`,
`literature_ontoma_lut_generation` took ~19 min (then on a single-node
`n1-standard-32`), `literature_publication_match` ~11.5 min and all of stage 3
< 4 min (then on 1 master + 2 workers `n1-standard-8`). Autoscaling on the
shared cluster is expected to cover the heavier stages; this is the first thing
to tune from the next run's data.

## Files to create / modify / delete

**New:**
- `scripts/literature/launch_literature.sh`

**Deleted:**
- `scripts/literature/_common.sh`
- `scripts/literature/01_launch_ontoma_lut.sh`
- `scripts/literature/02_launch_publication_match.sh`
- `scripts/literature/03a_launch_entity_lut.sh`
- `scripts/literature/03b_launch_embedding_vector.sh`
- `scripts/literature/03c_launch_cooccurrence_evidence.sh`

**Unchanged:**
- `config.yaml` — the launcher generates its own inline config (the established
  pattern, same as `scripts/launch_association_benchmark.sh`). The canonical
  per-step entries stay valid for granular local `pts --step` runs.
- `src/pts/pyspark/literature_publication_match.py`,
  `src/pts/pyspark/literature_cooccurrence_evidence.py`, and the two bug-fix
  commits — unrelated to this consolidation.

## Validation

`launch_literature.sh` is validated by `bash -n` (syntax) and by an actual
end-to-end `run-001` on Dataproc — the same way the per-stage launchers were
validated. Success criteria: all six step-jobs reach `DONE`, and the expected
outputs exist under `gs://ot-team/dochoa/literature_runs/<run-id>/` with
`_SUCCESS` markers:

| Step | Output(s) |
|------|-----------|
| `literature_ontoma_lut_generation` | `intermediate/ontoma/disease_target_drug_label_lookup_table.parquet` |
| `literature_publication_match` | `intermediate/literature/match`, `excluded/literature/match` |
| `literature_entity_lut` | `output/literature_entity_lut` |
| `literature_embedding` | `etc/model/w2v_model` |
| `literature_vector` | `output/literature_vector` |
| `literature_cooccurrence_evidence` | `intermediate/literature/cooccurrence`, `intermediate/evidence/literature_epmc` |

## Open notes

- **Failure of a parallel sibling:** if `literature_embedding` fails, the
  launcher stops before submitting `literature_vector`, but
  `literature_entity_lut` and `literature_cooccurrence_evidence` may still be
  running. They are left to finish; the cluster stays up. This is intentional —
  each step is an independent job.
- **Re-running:** a re-run recreates the cluster from scratch (so it always
  installs the current `PTS_REF` from GitHub). To re-run a single failed step
  against a still-running cluster, the operator submits it manually with
  `gcloud dataproc jobs submit` — the launcher does not need a per-step mode.
- **`pool_size`** in the generated config is irrelevant here (each job runs a
  single-task step group); kept at 8 only for consistency with other configs.
