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
