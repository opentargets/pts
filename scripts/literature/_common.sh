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
