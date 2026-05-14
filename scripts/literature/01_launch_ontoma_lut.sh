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
