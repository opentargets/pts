#!/usr/bin/env bash
# Resolve the spark-nlp transitive dependency tree once and stage every JAR in
# GCS, so the literature launcher can use `spark.jars=gs://…/jar1.jar,…`
# instead of `spark.jars.packages=…`. Skips per-app Ivy resolution on the
# Dataproc cluster (driver bootstrap drops from ~30 s–2 min to seconds).
#
# Re-run only when bumping spark-nlp to a new version.
#
# Usage:
#   ./scripts/literature/stage_spark_jars.sh [version] [gcs-prefix]
# Defaults:
#   version    : 6.1.3
#   gcs-prefix : gs://ot-team/dochoa/spark-jars/spark-nlp-<version>/

set -euo pipefail

VERSION="${1:-6.1.3}"
GCS_PREFIX="${2:-gs://opentargets-pipelines/up/pts/jars/spark-nlp-${VERSION}/}"
COURSIER_VERSION="2.1.24"

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
echo "Local workspace: ${WORK_DIR}"

# ── Bootstrap coursier (no system install) ──────────────────────────────────
ARCH="$(uname -m)"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
case "${OS}-${ARCH}" in
  darwin-arm64)   CS_TRIPLE="aarch64-apple-darwin" ;;
  darwin-x86_64)  CS_TRIPLE="x86_64-apple-darwin" ;;
  linux-x86_64)   CS_TRIPLE="x86_64-pc-linux" ;;
  linux-aarch64)  CS_TRIPLE="aarch64-pc-linux" ;;
  *)              echo "Unsupported platform: ${OS}-${ARCH}" >&2; exit 1 ;;
esac
CS_URL="https://github.com/coursier/coursier/releases/download/v${COURSIER_VERSION}/cs-${CS_TRIPLE}.gz"
echo "Downloading coursier ${COURSIER_VERSION} (${CS_TRIPLE})..."
curl -fL "${CS_URL}" | gunzip > "${WORK_DIR}/cs"
chmod +x "${WORK_DIR}/cs"

# ── Resolve transitive tree ─────────────────────────────────────────────────
COORDINATE="com.johnsnowlabs.nlp:spark-nlp_2.12:${VERSION}"
echo "Resolving ${COORDINATE}..."
"${WORK_DIR}/cs" fetch "${COORDINATE}" > "${WORK_DIR}/jars.txt"
NUM_JARS="$(wc -l < "${WORK_DIR}/jars.txt" | tr -d ' ')"
echo "Resolved ${NUM_JARS} JARs"

# ── Upload to GCS in parallel ───────────────────────────────────────────────
echo "Uploading ${NUM_JARS} JARs to ${GCS_PREFIX}..."
xargs -P 8 -I{} gcloud storage cp "{}" "${GCS_PREFIX}" --quiet < "${WORK_DIR}/jars.txt"

# ── Write the manifest ──────────────────────────────────────────────────────
# A single CSV file with the comma-separated GCS paths of every staged JAR.
# The launcher reads this via `gcloud storage cat` so it doesn't depend on
# `gcloud storage ls` (which can be filtered/paginated by harnesses).
MANIFEST_GCS="${GCS_PREFIX%/}.manifest.csv"
awk -v prefix="${GCS_PREFIX}" '{ n=split($0, parts, "/"); printf "%s%s%s", (NR==1?"":","), prefix, parts[n] }' \
  "${WORK_DIR}/jars.txt" > "${WORK_DIR}/manifest.csv"
gcloud storage cp "${WORK_DIR}/manifest.csv" "${MANIFEST_GCS}" --quiet

# ── Final report ────────────────────────────────────────────────────────────
cat <<EOF

========================================================================
Staged ${NUM_JARS} JARs to:
  ${GCS_PREFIX}
Manifest:
  ${MANIFEST_GCS}

The literature launcher reads the manifest at cluster create time:
  SPARK_NLP_JARS="\$(gcloud storage cat ${MANIFEST_GCS})"
  --properties="spark:spark.jars=\${SPARK_NLP_JARS}"
========================================================================
EOF
