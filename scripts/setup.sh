#!/usr/bin/env bash
# setup.sh — One-command bootstrap for the XTAL development environment.
#
# Usage:
#   chmod +x scripts/setup.sh && ./scripts/setup.sh

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║         XTAL — Week 1 Infrastructure Setup              ║"
echo "╚══════════════════════════════════════════════════════════╝"

# ── 1. Prerequisites check ───────────────────────────────────────────────────
echo ""
echo "[1/5] Checking prerequisites..."
for cmd in docker docker-compose curl python3 pip3; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "  ✗ Missing: $cmd — please install it first"
    exit 1
  fi
done
echo "  ✓ All prerequisites found"

# ── 2. Python dependencies ───────────────────────────────────────────────────
echo ""
echo "[2/5] Installing Python dependencies..."
$PYTHON -m pip install -q -r requirements.txt
echo "  ✓ Python packages installed"

# ── 3. Start the stack ───────────────────────────────────────────────────────
echo ""
echo "[3/5] Starting Docker Compose stack..."
docker compose up -d --build

echo "  Waiting for services to become healthy..."
MAX_WAIT=120
ELAPSED=0
while true; do
  HEALTHY=$(docker compose ps --format json 2>/dev/null \
    | python3 -c "
import sys, json
lines = [l for l in sys.stdin if l.strip()]
services = [json.loads(l) for l in lines]
healthy = sum(1 for s in services if s.get('Health') in ('healthy', ''))
print(healthy)
" 2>/dev/null || echo "0")
  TOTAL=$(docker compose ps -q | wc -l | tr -d ' ')
  echo "  $HEALTHY/$TOTAL services up..."
  if [[ "$HEALTHY" -ge "$TOTAL" && "$TOTAL" -gt "0" ]]; then
    break
  fi
  sleep 5
  ELAPSED=$((ELAPSED + 5))
  if [[ $ELAPSED -ge $MAX_WAIT ]]; then
    echo "  ✗ Timeout waiting for services. Check: docker compose logs"
    exit 1
  fi
done
echo "  ✓ All services running"

# ── 4. Create Kafka topics ───────────────────────────────────────────────────
echo ""
echo "[4/5] Creating Kafka topics..."
sleep 5  # give Kafka a moment after healthy

for topic in nexmark-bids nexmark-auctions nexmark-persons nexmark-output; do
  docker exec xtal-kafka kafka-topics.sh \
    --bootstrap-server localhost:9092 \
    --create --if-not-exists \
    --topic "$topic" \
    --partitions 4 \
    --replication-factor 1 2>&1 | grep -v "already exists" || true
  echo "  ✓ Topic: $topic"
done

# ── 5. Verify stack ──────────────────────────────────────────────────────────
echo ""
echo "[5/5] Verifying endpoints..."

check_endpoint() {
  local name="$1"
  local url="$2"
  if curl -sf "$url" >/dev/null 2>&1; then
    echo "  ✓ $name → $url"
  else
    echo "  ✗ $name unreachable at $url (may still be starting)"
  fi
}

check_endpoint "Flink REST API" "http://localhost:8081/overview"
check_endpoint "Prometheus"     "http://localhost:9090/-/healthy"
check_endpoint "Grafana"        "http://localhost:3000/api/health"
check_endpoint "Kafka Exporter" "http://localhost:9308/metrics"

echo ""
echo "══════════════════════════════════════════════════════════"
echo " XTAL stack is ready!"
echo ""
echo "  Flink UI:        http://localhost:8081"
echo "  Prometheus:      http://localhost:9090"
echo "  Grafana:         http://localhost:3000  (admin / xtal2024)"
echo ""
echo " Next steps:"
echo "  1. Submit a test job:   ./jobs/nexmark_submit.sh 5 4 300"
echo "  2. Generate load:       python scripts/load_generator.py --rate 5000"
echo "  3. Verify metrics:      python -c 'from controller.telemetry import *; print(TelemetryCollector().collect())'"
echo "══════════════════════════════════════════════════════════"
