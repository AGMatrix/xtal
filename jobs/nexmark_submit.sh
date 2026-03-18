#!/usr/bin/env bash
# nexmark_submit.sh — Downloads and submits the Flink NEXMark benchmark job.
#
# NEXMark is not bundled in the standard Flink Docker image.
# We use the flink-training-exercises NEXMark jar, or build from source.
#
# Usage:
#   ./jobs/nexmark_submit.sh [query_number] [parallelism] [duration_sec]
#   ./jobs/nexmark_submit.sh 5 4 300

set -euo pipefail

FLINK_REST="${FLINK_REST:-http://localhost:8081}"
QUERY="${1:-5}"           # NEXMark query number (1-22, Q5=hottest items is most relevant)
PARALLELISM="${2:-4}"
DURATION="${3:-300}"      # seconds

NEXMARK_JAR="nexmark-flink.jar"
NEXMARK_VERSION="0.2"     # adjust if needed

# ── Download NEXMark jar if missing ──────────────────────────────────────────
if [[ ! -f "jobs/${NEXMARK_JAR}" ]]; then
  echo "[nexmark] Downloading NEXMark benchmark jar..."
  # Build from the Apache Flink Benchmarks repository
  # Alternative: use the pre-built jar from flink-benchmarks releases
  curl -fsSL -o "jobs/${NEXMARK_JAR}" \
    "https://repository.apache.org/content/groups/public/org/apache/flink/flink-nexmark-${NEXMARK_VERSION}/${NEXMARK_JAR}" \
    2>/dev/null || {
      echo "[nexmark] Pre-built jar not found. Building from source..."
      build_nexmark_from_source
    }
fi

build_nexmark_from_source() {
  local tmp
  tmp=$(mktemp -d)
  git clone --depth=1 https://github.com/apache/flink-benchmarks.git "$tmp"
  cd "$tmp"
  mvn -q package -DskipTests -Pflink-1.18
  cp target/flink-benchmarks-*.jar "${OLDPWD}/jobs/${NEXMARK_JAR}"
  cd "$OLDPWD"
  rm -rf "$tmp"
}

# ── Create Kafka topics used by NEXMark ──────────────────────────────────────
echo "[nexmark] Ensuring Kafka topics exist..."
docker exec xtal-kafka kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --if-not-exists \
  --topic nexmark-bids --partitions 4 --replication-factor 1

docker exec xtal-kafka kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --if-not-exists \
  --topic nexmark-auctions --partitions 4 --replication-factor 1

docker exec xtal-kafka kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --if-not-exists \
  --topic nexmark-persons --partitions 4 --replication-factor 1

docker exec xtal-kafka kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --if-not-exists \
  --topic nexmark-output --partitions 4 --replication-factor 1

# ── Submit the NEXMark job ────────────────────────────────────────────────────
echo "[nexmark] Submitting NEXMark Q${QUERY} (parallelism=${PARALLELISM}, duration=${DURATION}s)..."

# Copy the jar into the jobmanager container and submit via Flink CLI
docker cp "jobs/${NEXMARK_JAR}" xtal-jobmanager:/tmp/nexmark.jar

docker exec xtal-jobmanager /opt/flink/bin/flink run \
  --parallelism "${PARALLELISM}" \
  --class com.github.nexmark.flink.Benchmark \
  /tmp/nexmark.jar \
  --queries "q${QUERY}" \
  --duration "${DURATION}" \
  --kafka-bootstrap-servers "kafka:9092" \
  --checkpointing \
  --event-rate 1000000

echo "[nexmark] Job submitted. Monitor at ${FLINK_REST}/#/overview"
