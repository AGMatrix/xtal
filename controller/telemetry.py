"""
telemetry.py — Queries Prometheus for XTAL state observations.

State vector used by the controller (Week 3):
  - p99_latency_ms       : end-to-end latency at 99th percentile
  - p999_latency_ms      : end-to-end latency at 99.9th percentile
  - input_buffer_usage   : avg fraction of input buffers full (backpressure proxy)
  - checkpoint_duration  : last checkpoint duration in ms
  - kafka_consumer_lag   : total consumer lag across all partitions
  - records_in_per_sec   : ingestion throughput
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from typing import Optional

import requests

log = logging.getLogger(__name__)


@dataclass
class SystemState:
    timestamp: float
    p99_latency_ms: float
    p999_latency_ms: float
    input_buffer_usage: float       # 0.0 – 1.0
    checkpoint_duration_ms: float
    kafka_consumer_lag: float
    records_in_per_sec: float

    def as_feature_vector(self) -> list[float]:
        """Returns a flat list for the ML model (excludes timestamp)."""
        return [
            self.p99_latency_ms,
            self.p999_latency_ms,
            self.input_buffer_usage,
            self.checkpoint_duration_ms,
            self.kafka_consumer_lag,
            self.records_in_per_sec,
        ]

    def to_dict(self) -> dict:
        return asdict(self)


class TelemetryCollector:
    """Pulls metrics from Prometheus and returns a SystemState snapshot."""

    # PromQL queries — tuned for Flink 1.18 metric names
    QUERIES = {
        "p99_latency_ms": (
            "histogram_quantile(0.99, sum("
            "rate(flink_taskmanager_job_latency_source_id_operator_id_operator_subtask_index_bucket[30s])"
            ") by (le))"
        ),
        "p999_latency_ms": (
            "histogram_quantile(0.999, sum("
            "rate(flink_taskmanager_job_latency_source_id_operator_id_operator_subtask_index_bucket[30s])"
            ") by (le))"
        ),
        "input_buffer_usage": (
            "avg(flink_taskmanager_job_task_buffers_inPoolUsage)"
        ),
        "checkpoint_duration_ms": (
            "flink_jobmanager_job_lastCheckpointDuration"
        ),
        "kafka_consumer_lag": (
            "sum(kafka_consumergroup_lag)"
        ),
        "records_in_per_sec": (
            "sum(rate(flink_taskmanager_job_task_operator_numRecordsInPerSecond[30s]))"
        ),
    }

    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.url = prometheus_url.rstrip("/")
        self._session = requests.Session()

    def _query(self, promql: str) -> Optional[float]:
        """Executes an instant PromQL query, returns the scalar value or None."""
        try:
            resp = self._session.get(
                f"{self.url}/api/v1/query",
                params={"query": promql},
                timeout=3,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("data", {}).get("result", [])
            if not results:
                return None
            return float(results[0]["value"][1])
        except Exception as exc:
            log.warning("Prometheus query failed: %s | %s", promql[:60], exc)
            return None

    def collect(self) -> SystemState:
        """Returns the current SystemState. Missing metrics default to 0."""
        values: dict[str, float] = {}
        for key, query in self.QUERIES.items():
            val = self._query(query)
            values[key] = val if val is not None else 0.0

        return SystemState(
            timestamp=time.time(),
            p99_latency_ms=values["p99_latency_ms"],
            p999_latency_ms=values["p999_latency_ms"],
            input_buffer_usage=values["input_buffer_usage"],
            checkpoint_duration_ms=values["checkpoint_duration_ms"],
            kafka_consumer_lag=values["kafka_consumer_lag"],
            records_in_per_sec=values["records_in_per_sec"],
        )

    def tail_latency_ok(self, state: SystemState, sla_ms: float = 500.0) -> bool:
        """Returns True if p99 latency is within the SLA deadline."""
        return state.p99_latency_ms <= sla_ms


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collector = TelemetryCollector()
    state = collector.collect()
    print(state)
