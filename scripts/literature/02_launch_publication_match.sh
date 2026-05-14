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
