#!/usr/bin/env bash
# submit_sql_job.sh — Submits the NEXMark Q5 SQL job to the Flink cluster.
#
# Requires the Flink Kafka SQL connector jar in the Flink lib directory.
# This script downloads the connector jar if missing, copies it to the
# JobManager container, and submits the SQL via the Flink SQL client.
#
# Usage: ./jobs/submit_sql_job.sh

set -euo pipefail
cd "$(dirname "$0")/.."

FLINK_VERSION="1.18.1"
SCALA_VERSION="2.12"
KAFKA_CONNECTOR_JAR="flink-sql-connector-kafka-${FLINK_VERSION}.jar"
KAFKA_CONNECTOR_URL="https://repo.maven.apache.org/maven2/org/apache/flink/flink-sql-connector-kafka/${FLINK_VERSION}/${KAFKA_CONNECTOR_JAR}"

echo "[submit] Flink SQL NEXMark Q5 Job Submission"

# ── Download Kafka SQL connector if missing ───────────────────────────────────
if [[ ! -f "jobs/${KAFKA_CONNECTOR_JAR}" ]]; then
  echo "[submit] Downloading Kafka SQL connector..."
  curl -fsSL -o "jobs/${KAFKA_CONNECTOR_JAR}" "${KAFKA_CONNECTOR_URL}"
  echo "[submit] Downloaded ${KAFKA_CONNECTOR_JAR}"
fi

# ── Copy connector jar into JobManager ────────────────────────────────────────
echo "[submit] Copying connector jar to JobManager..."
docker cp "jobs/${KAFKA_CONNECTOR_JAR}" xtal-jobmanager:/opt/flink/lib/${KAFKA_CONNECTOR_JAR}

# ── Verify Kafka topics exist ─────────────────────────────────────────────────
for topic in nexmark-bids nexmark-auctions nexmark-output; do
  docker exec xtal-kafka kafka-topics.sh \
    --bootstrap-server localhost:9092 \
    --create --if-not-exists \
    --topic "$topic" \
    --partitions 4 \
    --replication-factor 1 2>&1 | grep -v "already exists" || true
done
echo "[submit] Kafka topics verified"

# ── Submit via Flink SQL client ───────────────────────────────────────────────
echo "[submit] Submitting SQL job..."
docker exec -i xtal-jobmanager /opt/flink/bin/sql-client.sh \
  --jar /opt/flink/lib/${KAFKA_CONNECTOR_JAR} \
  <<'EOF'
SET 'execution.runtime-mode' = 'streaming';
SET 'parallelism.default' = '4';
SET 'execution.checkpointing.interval' = '5000';
SET 'execution.checkpointing.mode' = 'EXACTLY_ONCE';
SET 'state.backend' = 'rocksdb';

EXECUTE FILE '/dev/stdin';
EOF

# Pipe the SQL file
cat jobs/nexmark_q5.sql | docker exec -i xtal-jobmanager /opt/flink/bin/sql-client.sh \
  --jar /opt/flink/lib/${KAFKA_CONNECTOR_JAR}

echo "[submit] Job submitted. Check Flink UI at http://localhost:8081"
echo "[submit] Verify job is running:"
curl -s http://localhost:8081/jobs | python3 -m json.tool | grep -E '"id"|"status"'
