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
AUTOSCALING_POLICY="pts-literature-50-secondary"

# ── Shared inputs / outputs ─────────────────────────────────────────────────
INPUT_BASE="gs://open-targets-pipeline-runs/ds/26.03-test5"
OUTPUT_BASE="gs://ot-team/dochoa/literature_runs"
EPMC_BASE="gs://otar025-epmc/ml02"
SPARK_NLP_VERSION="6.1.3"
SPARK_NLP_JARS_PREFIX="gs://opentargets-pipelines/up/pts/jars/spark-nlp-${SPARK_NLP_VERSION}/"
SPARK_NLP_JARS_MANIFEST="${SPARK_NLP_JARS_PREFIX%/}.manifest.csv"

# ── Partitioning tunables ───────────────────────────────────────────────────
# Set DATE_PREFIX='' to read all EPMC days. REPARTITION sizes the raw EPMC
# read for stage 2; SHUFFLE_* size the post-shuffle partition counts for
# each step's heavy joins/groupBys. Defaults assume a full-EPMC run on the
# autoscaled 4×n1-highmem-16 + ≤50 secondary cluster (~864 vCPU peak); dial
# down for smaller runs (AQE will coalesce, but write fan-out won't).
DATE_PREFIX=""
REPARTITION="25000"
SHUFFLE_PUBMATCH="15000"
SHUFFLE_EMBEDDING="15000"
SHUFFLE_COOC="15000"
SHUFFLE_ENTITY="10000"

# Output partition counts for publication_match writes. AQE coalesce alone
# does not consolidate the write here (likely because match_disambiguated.df
# is .persist()ed before the filter+write, freezing the post-shuffle
# partitioning), so we force the output shape explicitly. Targets ~256 MB
# per parquet file at run-014 scale:
#   148 GB match_valid  / 600 partitions ≈ 250 MB / file
#   130 GB match_failed / 530 partitions ≈ 250 MB / file
COALESCE_MATCH_VALID="600"
COALESCE_MATCH_FAILED="530"

# Output partition counts for cooccurrence_evidence writes. Same small-file
# pattern observed at run-016 scale (cooccurrence wrote 3752 files at ~2.7 MB
# each); targets ~250 MB per parquet file:
#   9.82 GB cooccurrence  / 40 partitions ≈ 245 MB / file
#   7.84 GB evidence      / 32 partitions ≈ 245 MB / file
COALESCE_COOCCURRENCE="40"
COALESCE_EVIDENCE="32"

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
      properties:
        # OnToma checks spark.jars.packages contains 'spark-nlp' before running
        # (ontoma/ontoma.py:55). Spark-NLP itself is loaded from pre-staged JARs
        # via cluster-level spark.jars, so this value is informational only —
        # it goes into SparkConf at session creation (post-SparkSubmit), which
        # does NOT re-trigger Ivy resolution.
        spark.jars.packages: 'com.johnsnowlabs.nlp:spark-nlp_2.12:${SPARK_NLP_VERSION}'

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
        match_valid_coalesce: ${COALESCE_MATCH_VALID}
        match_failed_coalesce: ${COALESCE_MATCH_FAILED}
      properties:
        spark.sql.shuffle.partitions: '${SHUFFLE_PUBMATCH}'
        spark.sql.adaptive.skewJoin.skewedPartitionFactor: '2'
        spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes: '64MB'
        # AQE coalesce settings still help downstream intermediate stages
        # (observed in run-014: 15000 -> 1496 and 15000 -> 124 task drops).
        # They do NOT consolidate the final write because the dataframe is
        # .persist()ed before the filter+write, freezing the post-shuffle
        # partitioning. The match_valid_coalesce / match_failed_coalesce
        # settings above are what shrink the output file count.
        spark.sql.adaptive.advisoryPartitionSizeInBytes: '268435456'
        spark.sql.adaptive.coalescePartitions.parallelismFirst: 'false'
        # Hold all executors for the duration of this step. Without these
        # settings the Dataproc autoscaler can decommission idle-looking
        # executors mid-job (run-015 lost ~40 executors at 10:08:08 between
        # stage boundaries). Block migration during decommission covers
        # shuffle blocks but NOT broadcast pieces, so the next stage's
        # broadcast fetches fail with BlockNotFoundException and abort the
        # job. By keeping Spark from releasing executors, YARN sees vCores
        # as allocated and the autoscaler does not trigger scale-down
        # within this step. Lighter downstream steps keep the defaults so
        # the cluster can shrink between them.
        spark.dynamicAllocation.executorIdleTimeout: '7200s'
        spark.dynamicAllocation.shuffleTracking.enabled: 'true'
        spark.dynamicAllocation.shuffleTracking.timeout: '7200s'
        # See literature_ontoma_lut_generation: Match.map_labels also instantiates
        # OnToma, which requires spark.jars.packages to contain 'spark-nlp'.
        spark.jars.packages: 'com.johnsnowlabs.nlp:spark-nlp_2.12:${SPARK_NLP_VERSION}'
        # Force Spark to subdivide large EPMC jsonl files. The default 128MB
        # allows multi-GB files to be processed by a single task, which on
        # the full-EPMC corpus produced catastrophic per-file skew (p50 2s,
        # p99 560s, max 17min). 32MB chunks bring per-task work back to a
        # reasonable size and let the downstream repartition do the rest.
        spark.sql.files.maxPartitionBytes: '33554432'

  literature_entity_lut:
    - name: pyspark literature entity lut
      pyspark: literature_entity_lut
      source:
        matches: intermediate/literature/match
      destination:
        literature_entity_lut: output/literature_entity_lut
      properties:
        spark.sql.shuffle.partitions: '${SHUFFLE_ENTITY}'

  literature_embedding:
    - name: pyspark literature embedding
      pyspark: literature_embedding
      source:
        matches: intermediate/literature/match
      destination:
        model: etc/model/w2v_model
      properties:
        spark.sql.shuffle.partitions: '${SHUFFLE_EMBEDDING}'

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
      settings:
        cooccurrence_coalesce: ${COALESCE_COOCCURRENCE}
        evidence_coalesce: ${COALESCE_EVIDENCE}
      properties:
        spark.sql.shuffle.partitions: '${SHUFFLE_COOC}'
        spark.sql.adaptive.skewJoin.skewedPartitionFactor: '2'
        spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes: '64MB'
EOF

# ── Upload runner + config + init script to GCS ─────────────────────────────
echo "Uploading runner, config, and init script to GCS..."
gcloud storage cp "${WORK_DIR}/dataproc_pts_run.py" "${RUNNER_GCS}" --quiet
gcloud storage cp "${WORK_DIR}/config.yaml" "${CONFIG_GCS}" --quiet
gcloud storage cp "${WORK_DIR}/install_dependencies_on_cluster.sh" "${INIT_SCRIPT_GCS}" --quiet

# ── Load pre-staged spark-nlp JAR list from GCS manifest ────────────────────
# spark-nlp's transitive dependency tree (~150 JARs) is pre-staged in GCS by
# scripts/literature/stage_spark_jars.sh, which also writes a single
# comma-separated manifest CSV alongside the JAR directory. We download the
# manifest to local disk and read its contents — `gcloud storage cat` is
# subject to stdout filtering by some harnesses, but `gcloud storage cp` to
# a local file is not. Passing spark.jars=gs://...jar1,jar2,... at cluster
# create lets Spark download the JARs from GCS in parallel and skips the
# per-app Ivy resolution that spark.jars.packages would force on every Spark
# driver startup. spark.jars must be set here (cluster level) because the
# property is read by SparkSubmit before the JVM starts — setting it via
# PTS YAML per-step properties is too late and fails with
# "JavaPackage object is not callable".
echo "Loading spark-nlp JAR manifest from ${SPARK_NLP_JARS_MANIFEST}..."
gcloud storage cp "${SPARK_NLP_JARS_MANIFEST}" "${WORK_DIR}/spark-nlp.manifest" --quiet
SPARK_NLP_JARS="$(< "${WORK_DIR}/spark-nlp.manifest")"
NUM_JARS="$(($(echo -n "${SPARK_NLP_JARS}" | tr ',' '\n' | wc -l) + 1))"
echo "  ${NUM_JARS} JARs"

# ── Create the shared cluster ───────────────────────────────────────────────
# pts_openfda-like topology + autoscaling, with spark-nlp JARs pre-staged in
# GCS (see above). Secondary workers are non-preemptible — preemptible
# secondaries lose shuffle data mid-job and tank shuffle-heavy stages
# (publication_match, embedding, cooccurrence). Component Gateway is enabled
# for Spark/YARN UI access via Cloud Console.
# Shuffle resilience knobs absorb the kind of transient External Shuffle
# Service connection drops we saw in run-016 (438 FetchFailed events
# stretched across 3 stage 244 retries; all "Connection from sw-XXXX:7337
# closed" errors). With Spark defaults (maxRetries=3, retryWait=5s) a 30s
# blip blew through retries and triggered stage recomputes. The bumped
# values give Spark 10 x 15s = 150s to absorb a transient drop silently.
# Capping in-flight blocks per source avoids overwhelming a wobbly ESS.
CLUSTER_PROPERTIES=(
  "spark:spark.jars=${SPARK_NLP_JARS}"
  "spark:spark.sql.autoBroadcastJoinThreshold=134217728"
  "spark:spark.shuffle.io.maxRetries=10"
  "spark:spark.shuffle.io.retryWait=15s"
  "spark:spark.network.timeout=300s"
  "spark:spark.reducer.maxBlocksInFlightPerAddress=128"
)
CLUSTER_PROPERTIES_JOINED="$(IFS='#'; echo "${CLUSTER_PROPERTIES[*]}")"

echo "Creating Dataproc cluster ${CLUSTER_NAME}..."
gcloud dataproc clusters create "${CLUSTER_NAME}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --zone="${ZONE}" \
  --image-version=2.2 \
  --num-masters=1 \
  --num-workers=4 \
  --master-machine-type=n1-standard-8 \
  --master-boot-disk-size=512GB \
  --worker-machine-type=n1-highmem-16 \
  --worker-boot-disk-size=256GB \
  --secondary-worker-type=non-preemptible \
  --autoscaling-policy="${AUTOSCALING_POLICY}" \
  --service-account="${SERVICE_ACCOUNT}" \
  --metadata="PTS_REF=${PTS_REF}" \
  --initialization-actions="${INIT_SCRIPT_GCS}" \
  --max-idle=3600s \
  --public-ip-address \
  --enable-component-gateway \
  --labels="workload=literature,run-id=${RUN_ID}" \
  --properties="^#^${CLUSTER_PROPERTIES_JOINED}"

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
    --labels="workload=literature,run-id=${RUN_ID},step=${step}" \
    --async \
    -- \
    -s "${step}" \
    -c config.yaml \
    -r "${RELEASE_URI}" >/dev/null
  echo "${job_id}"
}

# Wait for one job to reach a terminal state by polling the Dataproc REST API.
# Faster than `gcloud dataproc jobs wait`, which streams the driver log and
# can drain its buffers for minutes after the job actually entered DONE — that
# drain stretched the wall-clock between stages by ~10 min on run-006.
#
# Terminal states: DONE → success; ERROR | CANCELLED → failure (print banner,
# exit non-zero, leave the cluster up for inspection). Transient states keep
# polling. Driver output is no longer streamed live; inspect it via Component
# Gateway, the Cloud Console job page, or by reading the GCS URI from
# `gcloud dataproc jobs describe ${job_id} --format='value(driverOutputResourceUri)'`.
wait_job() {
  local step="$1"
  local job_id="$2"
  echo
  echo "Waiting for ${step} (job ${job_id})..."
  local state
  local poll_count=0
  local poll_interval=15
  while true; do
    state="$(gcloud dataproc jobs describe "${job_id}" \
              --project="${PROJECT}" --region="${REGION}" \
              --format='value(status.state)' 2>/dev/null || true)"
    case "${state}" in
      DONE)
        echo "${step} DONE"
        return 0
        ;;
      ERROR|CANCELLED)
        cat >&2 <<FAILEOF

========================================================================
FAILED: ${step} (state=${state})
  Job ID  : ${job_id}
  Console : https://console.cloud.google.com/dataproc/jobs/${job_id}?project=${PROJECT}&region=${REGION}
  Re-inspect:
    gcloud dataproc jobs describe ${job_id} --project=${PROJECT} --region=${REGION}

Cluster ${CLUSTER_NAME} is left running for inspection.
Tear it down when you're finished with:
  gcloud dataproc clusters delete ${CLUSTER_NAME} --project=${PROJECT} --region=${REGION} --quiet
========================================================================
FAILEOF
        exit 1
        ;;
    esac
    if (( poll_count % 4 == 0 )); then
      echo "  ${step} state=${state:-?} (elapsed $((poll_count * poll_interval))s)"
    fi
    poll_count=$((poll_count + 1))
    sleep "${poll_interval}"
  done
}

# ── Orchestration sequence ──────────────────────────────────────────────────
# Set STOP_AFTER=<step-name> in the environment to stop the launcher after a
# given step succeeds (the cluster is left running so the user can inspect
# Spark/YARN UIs or rerun a downstream step manually). Useful for eval runs
# that only need the first N steps.
STOP_AFTER="${STOP_AFTER:-}"
maybe_stop() {
  local stage="$1"
  if [[ "${STOP_AFTER}" == "${stage}" ]]; then
    cat <<MSG

========================================================================
STOP_AFTER=${stage}: stopping launcher after this step succeeded.
Cluster ${CLUSTER_NAME} is left running for inspection.
Tear it down when finished with:
  gcloud dataproc clusters delete ${CLUSTER_NAME} --project=${PROJECT} --region=${REGION} --quiet
========================================================================
MSG
    exit 0
  fi
}

JOB_ONTOMA="$(submit_step literature_ontoma_lut_generation)"
wait_job literature_ontoma_lut_generation "${JOB_ONTOMA}"
maybe_stop literature_ontoma_lut_generation

JOB_PUBMATCH="$(submit_step literature_publication_match)"
wait_job literature_publication_match "${JOB_PUBMATCH}"
maybe_stop literature_publication_match

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
