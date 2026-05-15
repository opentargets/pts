# Literature Launcher Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the six files in `scripts/literature/` with a single self-contained `launch_literature.sh` that creates one Dataproc cluster and runs the whole literature pipeline against it as six separate `pts` step-jobs, orchestrated by the script in dependency order.

**Architecture:** One bash script. Creates one cluster (`pts_openfda`-like + autoscaling + spark-nlp). Generates one `config.yaml` with six independent `steps:` entries. Submits each step as a separate `gcloud dataproc jobs submit pyspark --async` against the shared cluster; uses `gcloud dataproc jobs wait` between dependent stages; fans out the three independent stage-3 jobs in parallel; waits for the parallel set before declaring success. No Otter `requires`.

**Tech Stack:** Bash, `gcloud dataproc`, the existing PTS PySpark step modules (`literature_ontoma_lut_generation`, `literature_publication_match`, `literature_entity_lut`, `literature_embedding`, `literature_vector`, `literature_cooccurrence_evidence`) — none of which change.

**Reference:** Design doc at `docs/superpowers/specs/2026-05-14-literature-launcher-consolidation-design.md`. The current per-stage launchers in `scripts/literature/` (especially `_common.sh` and `02_launch_publication_match.sh`) are the closest pattern to copy/inline from.

---

## File Structure

**New:**
- `scripts/literature/launch_literature.sh` — the single consolidated launcher. Self-contained: generates the Dataproc runner + custom init script + unified `config.yaml`, uploads them, creates the cluster, submits and waits on six step-jobs in dependency order, prints monitoring/teardown.

**Deleted (after the new script is in place and verified):**
- `scripts/literature/_common.sh`
- `scripts/literature/01_launch_ontoma_lut.sh`
- `scripts/literature/02_launch_publication_match.sh`
- `scripts/literature/03a_launch_entity_lut.sh`
- `scripts/literature/03b_launch_embedding_vector.sh`
- `scripts/literature/03c_launch_cooccurrence_evidence.sh`

**Unchanged (do NOT modify):**
- `config.yaml` — the launcher generates its own inline config; the canonical per-step entries remain valid for granular local runs.
- `src/pts/pyspark/literature_*.py` and the two bug-fix commits already on the branch.

---

## Background the engineer needs

- **PTS step contract.** The PTS Dataproc entry point is `pts.core.main`, run as a pyspark job with args `-s <step-name> -c <config-basename> -r <release_uri>`. Each `-s <step>` runs one step group from the config. The runner mirrors orchestration's `dataproc_pts_run.py` (already used by every launcher in this repo).
- **Init script workaround.** The standard init action at `gs://opentargets-pipelines/up/pts/install_dependencies_on_cluster.sh` reads a `openai-token` Secret Manager secret that isn't provisioned in this project and fails cluster creation. The existing launchers ship a custom init script that installs PTS from `git+https://github.com/opentargets/pts.git@${PTS_REF}` and **skips** the secret fetch. Copy that script verbatim.
- **Path handling within one `release_uri`.** Otter's `make_absolute` resolves relative source/destination paths against `release_uri`. All six jobs in this script share one `release_uri`, so intra-pipeline handoffs (e.g. `intermediate/literature/match`) can use relative paths — each job resolves them to the same GCS location. Only external inputs use absolute `gs://…` paths.
- **`gcloud dataproc jobs wait` and exit codes.** `gcloud dataproc jobs wait` blocks until terminal state and exits non-zero when the job failed. The launcher must check that exit code **explicitly** (e.g. `if ! gcloud … ; then …`) so it can print a useful failure message before exiting. **Important:** earlier in this branch we found that piping `gcloud … wait` into `| tail` masks the exit code (the pipe exit code is the last command's). Do not pipe the wait through `tail` / `head` — let stdout go where it goes.
- **Job id uniqueness.** `gcloud dataproc jobs submit --id=<id>` rejects duplicate ids. We construct ids like `pts-<step>-<run-id>-<ts>` where `<ts>` is captured **once** at the start of the script, so the parallel fan-out submits don't collide on the same `date +%s` second.
- **Commit style.** Conventional Commits, no `Co-Authored-By` lines or Claude/Anthropic references (per repo/global config).
- **Working directory.** This work happens in the worktree `/Users/ochoa/Projects/pts/.claude/worktrees/vh-add-literature` on branch `vh-add-literature`. Run all commands from there.

---

## Task 1: Add the consolidated `launch_literature.sh`

**Files:**
- Create: `scripts/literature/launch_literature.sh`

No unit test — this is a bash launcher. Verification is `bash -n` (syntax) plus, post-merge, a real Dataproc run.

- [ ] **Step 1: Write the file**

Create `scripts/literature/launch_literature.sh` with EXACTLY this content:

```bash
#!/usr/bin/env bash
# Run the whole literature pipeline on a single Dataproc cluster.
#
# Creates one cluster, then submits each literature step as its own pyspark
# job and waits between dependent stages. Three stage-3 jobs (entity_lut,
# embedding, cooccurrence_evidence) run in parallel on the same cluster;
# vector is submitted after embedding succeeds.
#
# On failure of any step the launcher stops with a useful message and
# leaves the cluster up (sibling parallel jobs are left to finish; --max-idle
# eventually reclaims it).
#
# Usage:
#   ./scripts/literature/launch_literature.sh [run-id] [pts-ref]
# Defaults:
#   run-id  : run-001
#   pts-ref : vh-add-literature

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

# ── Stage-2 test-run tunables ───────────────────────────────────────────────
# Drop DATE_PREFIX to read all EPMC days; bump REPARTITION after watching the
# stage-2 Spark UI.
DATE_PREFIX="2026_03"
REPARTITION="1000"

# ── Args ────────────────────────────────────────────────────────────────────
RUN_ID="${1:-run-001}"
PTS_REF="${2:-vh-add-literature}"

CLUSTER_NAME="pts-literature-${RUN_ID}"
RELEASE_URI="${OUTPUT_BASE}/${RUN_ID}"
WORK_PREFIX="${RELEASE_URI}/etc"
RUNNER_GCS="${WORK_PREFIX}/bin/dataproc_pts_run.py"
CONFIG_GCS="${WORK_PREFIX}/config/config.yaml"
INIT_SCRIPT_GCS="${WORK_PREFIX}/bin/install_dependencies_on_cluster.sh"

# Captured once so the parallel fan-out submits don't collide on the same
# second when building job ids.
TS="$(date +%s)"

cat <<EOF
========================================================================
Run ID      : ${RUN_ID}
PTS ref     : ${PTS_REF}
Cluster     : ${CLUSTER_NAME}
Release URI : ${RELEASE_URI}
========================================================================
EOF

# ── Workspace ───────────────────────────────────────────────────────────────
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
echo "Local workspace: ${WORK_DIR}"

# ── Generate the Dataproc runner script (mirrors orchestration's) ───────────
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

# ── Generate the custom init script (skips the openai-token secret fetch) ───
# The standard init at gs://opentargets-pipelines/up/pts/install_dependencies_on_cluster.sh
# reads a Secret Manager secret 'openai-token' that isn't provisioned here,
# which fails cluster creation. The literature steps don't need OpenAI.
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

# ── Generate the unified PTS config (six independent step entries) ──────────
# Intra-pipeline paths are relative (resolved against release_uri per job);
# external inputs (release bucket, EPMC) are absolute.
cat > "${WORK_DIR}/config.yaml" <<EOF
work_path: /mnt/disks/work
log_level: INFO
pool_size: 8
release_uri: ${RELEASE_URI}
scratchpad: {}

steps:
  literature_ontoma_lut_generation:
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

  literature_publication_match:
    - name: pyspark literature publication match
      pyspark: literature_publication_match
      source:
        pub_id_lut: ${INPUT_BASE}/input/literature/PMID_PMCID_DOI.csv.gz
        epmc_publication: ${EPMC_BASE}
        ontoma_disease_target_drug_label_lut: intermediate/ontoma/disease_target_drug_label_lookup_table.parquet
      destination:
        match_valid: intermediate/literature/match
        match_failed: excluded/literature/match
      settings:
        date_prefix: '${DATE_PREFIX}'
        repartition: ${REPARTITION}

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
EOF

# ── Upload runner + config + init script to GCS ─────────────────────────────
echo "Uploading runner, config, and init script to GCS..."
gcloud storage cp "${WORK_DIR}/dataproc_pts_run.py" "${RUNNER_GCS}" --quiet
gcloud storage cp "${WORK_DIR}/config.yaml" "${CONFIG_GCS}" --quiet
gcloud storage cp "${WORK_DIR}/install_dependencies_on_cluster.sh" "${INIT_SCRIPT_GCS}" --quiet

# ── Create the shared cluster ───────────────────────────────────────────────
# pts_openfda-like topology + autoscaling + spark-nlp (needed by ontoma_lut
# and publication_match; harmless for the stage-3 jobs).
echo "Creating Dataproc cluster ${CLUSTER_NAME}..."
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

# ── Helpers ─────────────────────────────────────────────────────────────────
# Submit a single step against the shared cluster (async). Informational lines
# go to stderr so the captured stdout is just the job id.
submit_step() {
  local step="$1"
  local job_id="pts-${step}-${RUN_ID}-${TS}"
  echo >&2
  echo "Submitting step: ${step} (job ${job_id})..." >&2
  gcloud dataproc jobs submit pyspark "${RUNNER_GCS}" \
    --project="${PROJECT}" \
    --region="${REGION}" \
    --cluster="${CLUSTER_NAME}" \
    --id="${job_id}" \
    --files="${CONFIG_GCS}" \
    --async \
    -- \
    -s "${step}" \
    -c config.yaml \
    -r "${RELEASE_URI}" >/dev/null
  echo "${job_id}"
}

# Wait for one job and check exit code explicitly. On failure, print a useful
# message (console URL + re-inspect command + teardown command) and exit
# non-zero, leaving the cluster up.
wait_job() {
  local step="$1"
  local job_id="$2"
  echo
  echo "Waiting for ${step} (job ${job_id})..."
  if ! gcloud dataproc jobs wait "${job_id}" \
       --project="${PROJECT}" --region="${REGION}" >/dev/null 2>&1; then
    cat >&2 <<FAILEOF

========================================================================
FAILED: ${step}
  Job ID  : ${job_id}
  Console : https://console.cloud.google.com/dataproc/jobs/${job_id}?project=${PROJECT}&region=${REGION}
  Re-inspect:
    gcloud dataproc jobs wait ${job_id} --project=${PROJECT} --region=${REGION}

Cluster ${CLUSTER_NAME} is left running for inspection.
Tear it down when you're finished with:
  gcloud dataproc clusters delete ${CLUSTER_NAME} --project=${PROJECT} --region=${REGION} --quiet
========================================================================
FAILEOF
    exit 1
  fi
  echo "${step} DONE"
}

# ── Orchestration sequence ──────────────────────────────────────────────────
JOB_ONTOMA="$(submit_step literature_ontoma_lut_generation)"
wait_job literature_ontoma_lut_generation "${JOB_ONTOMA}"

JOB_PUBMATCH="$(submit_step literature_publication_match)"
wait_job literature_publication_match "${JOB_PUBMATCH}"

# Fan out: three stage-3 jobs run in parallel on the cluster.
JOB_ENTITY="$(submit_step literature_entity_lut)"
JOB_EMBED="$(submit_step literature_embedding)"
JOB_COOC="$(submit_step literature_cooccurrence_evidence)"

# Vector depends only on embedding; submit it as soon as embedding is done,
# while entity_lut and cooccurrence_evidence keep running in parallel.
wait_job literature_embedding "${JOB_EMBED}"
JOB_VECTOR="$(submit_step literature_vector)"

# Drain the remaining three.
wait_job literature_entity_lut "${JOB_ENTITY}"
wait_job literature_vector "${JOB_VECTOR}"
wait_job literature_cooccurrence_evidence "${JOB_COOC}"

cat <<EOF

========================================================================
All literature steps completed.
  Run ID      : ${RUN_ID}
  Release URI : ${RELEASE_URI}
  Cluster     : ${CLUSTER_NAME}

Outputs:
  ${RELEASE_URI}/intermediate/ontoma/disease_target_drug_label_lookup_table.parquet
  ${RELEASE_URI}/intermediate/literature/match
  ${RELEASE_URI}/excluded/literature/match
  ${RELEASE_URI}/output/literature_entity_lut
  ${RELEASE_URI}/etc/model/w2v_model
  ${RELEASE_URI}/output/literature_vector
  ${RELEASE_URI}/intermediate/literature/cooccurrence
  ${RELEASE_URI}/intermediate/evidence/literature_epmc

Spark UI (while cluster is up):
  gcloud dataproc clusters describe ${CLUSTER_NAME} --project=${PROJECT} --region=${REGION} --format='value(config.endpointConfig.httpPorts)'

Tear down (or wait for --max-idle to handle it):
  gcloud dataproc clusters delete ${CLUSTER_NAME} --project=${PROJECT} --region=${REGION} --quiet
========================================================================
EOF
```

- [ ] **Step 2: Verify bash syntax**

Run: `cd /Users/ochoa/Projects/pts/.claude/worktrees/vh-add-literature && bash -n scripts/literature/launch_literature.sh`
Expected: no output, exit 0.

- [ ] **Step 3: Make executable**

Run: `cd /Users/ochoa/Projects/pts/.claude/worktrees/vh-add-literature && chmod +x scripts/literature/launch_literature.sh`

- [ ] **Step 4: Confirm directory state**

Run: `cd /Users/ochoa/Projects/pts/.claude/worktrees/vh-add-literature && ls scripts/literature/`
Expected: the old 6 files PLUS `launch_literature.sh` are listed. (The deletion happens in Task 2.)

- [ ] **Step 5: Commit**

```bash
cd /Users/ochoa/Projects/pts/.claude/worktrees/vh-add-literature
git add scripts/literature/launch_literature.sh
git commit -m "feat(literature): add consolidated single-cluster launcher"
```

---

## Task 2: Delete the per-stage launcher files

**Files (all deleted):**
- `scripts/literature/_common.sh`
- `scripts/literature/01_launch_ontoma_lut.sh`
- `scripts/literature/02_launch_publication_match.sh`
- `scripts/literature/03a_launch_entity_lut.sh`
- `scripts/literature/03b_launch_embedding_vector.sh`
- `scripts/literature/03c_launch_cooccurrence_evidence.sh`

- [ ] **Step 1: Remove the six files via git**

Run:

```bash
cd /Users/ochoa/Projects/pts/.claude/worktrees/vh-add-literature
git rm scripts/literature/_common.sh \
       scripts/literature/01_launch_ontoma_lut.sh \
       scripts/literature/02_launch_publication_match.sh \
       scripts/literature/03a_launch_entity_lut.sh \
       scripts/literature/03b_launch_embedding_vector.sh \
       scripts/literature/03c_launch_cooccurrence_evidence.sh
```

Expected: `git rm` reports `rm 'scripts/literature/_common.sh'` (and similarly for each), all six removed from index and working tree.

- [ ] **Step 2: Confirm only the consolidated launcher remains**

Run: `cd /Users/ochoa/Projects/pts/.claude/worktrees/vh-add-literature && ls scripts/literature/`
Expected: a single entry, `launch_literature.sh`.

- [ ] **Step 3: Confirm `launch_literature.sh` still parses (sanity)**

Run: `cd /Users/ochoa/Projects/pts/.claude/worktrees/vh-add-literature && bash -n scripts/literature/launch_literature.sh`
Expected: no output, exit 0. (`launch_literature.sh` is self-contained and does not source `_common.sh`, so removing `_common.sh` cannot break it — this is a defensive check.)

- [ ] **Step 4: Commit**

```bash
cd /Users/ochoa/Projects/pts/.claude/worktrees/vh-add-literature
git commit -m "chore(literature): remove per-stage launcher scripts superseded by launch_literature.sh"
```

---

## Task 3: Final verification

**Files:** none (verification only).

- [ ] **Step 1: Full test suite**

Run: `cd /Users/ochoa/Projects/pts/.claude/worktrees/vh-add-literature && uv run pytest test --doctest-modules src/pts`
Expected: all tests pass. (No Python changed in this plan; this is a regression sanity check.)

- [ ] **Step 2: Lint the tree**

Run: `cd /Users/ochoa/Projects/pts/.claude/worktrees/vh-add-literature && uv run ruff check`
Expected: `All checks passed!`

- [ ] **Step 3: Verify the launcher parses and is executable**

Run: `cd /Users/ochoa/Projects/pts/.claude/worktrees/vh-add-literature && bash -n scripts/literature/launch_literature.sh`
Expected: no output, exit 0.

Run: `cd /Users/ochoa/Projects/pts/.claude/worktrees/vh-add-literature && test -x scripts/literature/launch_literature.sh && echo "executable" || echo "NOT-EXECUTABLE"`
Expected: `executable`.

- [ ] **Step 4: Verify the directory now has exactly one file**

Run: `cd /Users/ochoa/Projects/pts/.claude/worktrees/vh-add-literature && ls scripts/literature/`
Expected: only `launch_literature.sh` is listed.

- [ ] **Step 5: If anything fails, stop**

Do not declare completion past a failing check. Fix the offending task's content and re-run Steps 1–4.

---

## Operator runbook (post-implementation, not a code task)

Once merged, run the whole literature pipeline with a single command:

```bash
./scripts/literature/launch_literature.sh run-001
```

The script blocks until the pipeline completes (or stops at the first failed step, leaving the cluster up for inspection). Outputs land under `gs://ot-team/dochoa/literature_runs/run-001/`. To tune the test-run scope, edit the `DATE_PREFIX` / `REPARTITION` variables near the top of the script.

---

## Self-Review Notes

- **Spec coverage:** single self-contained `launch_literature.sh` (Task 1); 6-step config with relative intra-pipeline paths (Task 1, generated config block); `pts_openfda` + autoscaling + spark-nlp cluster (Task 1, `clusters create`); bash-orchestrated dependency sequence with parallel fan-out (Task 1, orchestration section); explicit per-job exit-code checks with failure message and cluster-left-running (Task 1, `wait_job` helper); `--max-idle` + printed teardown, not auto-deleted on success (Task 1, final echo and `clusters create` flag); deletion of the six per-stage files (Task 2); test suite + lint + syntax/exec checks (Task 3). All spec sections map to a task.
- **Placeholder scan:** none — every code/command block is concrete and ready to run.
- **Type consistency:** the helper functions `submit_step` (returns a job id on stdout) and `wait_job` (consumes step name + job id, side effects only) are used consistently throughout the orchestration section. The unified `config.yaml` step-group keys match the `-s <step>` arguments passed to `submit_step` exactly: `literature_ontoma_lut_generation`, `literature_publication_match`, `literature_entity_lut`, `literature_embedding`, `literature_vector`, `literature_cooccurrence_evidence`.
