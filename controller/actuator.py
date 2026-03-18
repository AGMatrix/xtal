"""
actuator.py — Sends control actions to Flink (REST API) and Kafka (AdminClient).

Control action space (used by the XTAL agent in Week 3):
  Flink:
    - set_checkpoint_interval(ms)   : change checkpoint cadence
    - rescale_job(parallelism)      : stop-with-savepoint + restart at new parallelism
    - trigger_checkpoint()          : force an immediate checkpoint
    - set_buffer_timeout(ms)        : tune operator output buffer flush latency

  Kafka:
    - set_linger_ms(topic, ms)      : producer batching delay
    - set_batch_size(topic, bytes)  : producer batch size
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import requests
from confluent_kafka.admin import AdminClient, ConfigResource, ConfigEntry, ConfigSource

log = logging.getLogger(__name__)

FLINK_REST = "http://localhost:8081"
KAFKA_BOOTSTRAP = "localhost:9092"


@dataclass
class ControlAction:
    checkpoint_interval_ms: Optional[int] = None    # None = no change
    parallelism: Optional[int] = None
    buffer_timeout_ms: Optional[int] = None
    kafka_linger_ms: Optional[int] = None
    kafka_batch_size_bytes: Optional[int] = None

    # Safety bounds
    MIN_CHECKPOINT_MS = 500
    MAX_CHECKPOINT_MS = 30_000
    MIN_PARALLELISM = 1
    MAX_PARALLELISM = 12
    MIN_BUFFER_MS = 0
    MAX_BUFFER_MS = 200

    def is_safe(self) -> bool:
        if self.checkpoint_interval_ms is not None:
            if not (self.MIN_CHECKPOINT_MS <= self.checkpoint_interval_ms <= self.MAX_CHECKPOINT_MS):
                return False
        if self.parallelism is not None:
            if not (self.MIN_PARALLELISM <= self.parallelism <= self.MAX_PARALLELISM):
                return False
        if self.buffer_timeout_ms is not None:
            if not (self.MIN_BUFFER_MS <= self.buffer_timeout_ms <= self.MAX_BUFFER_MS):
                return False
        return True


class FlinkActuator:
    """Wraps the Flink REST API for control actions."""

    def __init__(self, rest_url: str = FLINK_REST):
        self.url = rest_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers["Content-Type"] = "application/json"

    # ── Introspection ──────────────────────────────────────────────────────────

    def get_running_jobs(self) -> list[dict]:
        resp = self._session.get(f"{self.url}/jobs", timeout=5)
        resp.raise_for_status()
        return [j for j in resp.json().get("jobs", []) if j["status"] == "RUNNING"]

    def get_job_id(self) -> Optional[str]:
        jobs = self.get_running_jobs()
        if not jobs:
            log.warning("No running Flink jobs found")
            return None
        return jobs[0]["id"]

    def get_job_details(self, job_id: str) -> dict:
        resp = self._session.get(f"{self.url}/jobs/{job_id}", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def get_job_metrics(self, job_id: str, metric_keys: list[str]) -> dict[str, float]:
        keys_param = ",".join(metric_keys)
        resp = self._session.get(
            f"{self.url}/jobs/{job_id}/metrics",
            params={"get": keys_param},
            timeout=5,
        )
        resp.raise_for_status()
        return {m["id"]: float(m.get("value", 0)) for m in resp.json()}

    def get_checkpoint_stats(self, job_id: str) -> dict:
        resp = self._session.get(f"{self.url}/jobs/{job_id}/checkpoints", timeout=5)
        resp.raise_for_status()
        return resp.json()

    # ── Actions ───────────────────────────────────────────────────────────────

    def trigger_checkpoint(self, job_id: str) -> dict:
        """Forces an immediate checkpoint."""
        resp = self._session.post(
            f"{self.url}/jobs/{job_id}/checkpoints",
            json={"cancel-job": False},
            timeout=10,
        )
        resp.raise_for_status()
        log.info("Triggered checkpoint on job %s", job_id)
        return resp.json()

    def rescale_job(self, job_id: str, parallelism: int) -> str:
        """
        Rescales the job via stop-with-savepoint + restart at new parallelism.
        Returns the savepoint path.

        NOTE: This restarts the job. XTAL uses this sparingly — only when
        p99 latency is severely degraded and throughput headroom exists.
        """
        # Step 1: stop with savepoint
        stop_resp = self._session.post(
            f"{self.url}/jobs/{job_id}/stop",
            json={"drain": False, "targetDirectory": "/tmp/flink-savepoints"},
            timeout=30,
        )
        stop_resp.raise_for_status()
        trigger_id = stop_resp.json()["request-id"]

        # Step 2: poll until savepoint is complete
        savepoint_path = self._wait_for_savepoint(job_id, trigger_id)
        log.info("Savepoint created: %s", savepoint_path)

        # Step 3: caller must re-submit job with new parallelism + savepoint
        # (returned so the caller / submit script can handle re-submission)
        return savepoint_path

    def _wait_for_savepoint(self, job_id: str, trigger_id: str, timeout_s: int = 60) -> str:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            resp = self._session.get(
                f"{self.url}/jobs/{job_id}/savepoints/{trigger_id}",
                timeout=5,
            )
            data = resp.json()
            status = data.get("status", {}).get("id")
            if status == "COMPLETED":
                return data["operation"]["location"]
            if status == "FAILED":
                raise RuntimeError(f"Savepoint failed: {data}")
            time.sleep(2)
        raise TimeoutError("Savepoint did not complete in time")

    def set_job_config_dynamic(self, job_id: str, key: str, value: str) -> None:
        """
        Attempt to update a Flink dynamic property at runtime.
        Supported in Flink 1.17+ for a limited set of keys.
        """
        resp = self._session.patch(
            f"{self.url}/jobs/{job_id}/config",
            json={key: value},
            timeout=5,
        )
        if resp.status_code == 404:
            log.warning("Dynamic config update not supported for key=%s", key)
        else:
            resp.raise_for_status()
            log.info("Set %s=%s on job %s", key, value, job_id)


class KafkaActuator:
    """Wraps the Kafka AdminClient for producer/topic config tuning."""

    def __init__(self, bootstrap_servers: str = KAFKA_BOOTSTRAP):
        self._admin = AdminClient({"bootstrap.servers": bootstrap_servers})

    def get_topic_config(self, topic: str) -> dict[str, str]:
        resource = ConfigResource("TOPIC", topic)
        futures = self._admin.describe_configs([resource])
        config = futures[resource].result()
        return {k: v.value for k, v in config.items()}

    def set_topic_config(self, topic: str, configs: dict[str, str]) -> None:
        """Update topic-level configs (e.g., retention, segment size)."""
        resource = ConfigResource("TOPIC", topic)
        resource.set_config(configs)
        futures = self._admin.alter_configs([resource])
        futures[resource].result()
        log.info("Updated topic %s config: %s", topic, configs)

    def create_topic_if_missing(self, topic: str, num_partitions: int = 4, replication: int = 1) -> None:
        from confluent_kafka.admin import NewTopic
        topics = self._admin.list_topics(timeout=5).topics
        if topic not in topics:
            new_topic = NewTopic(topic, num_partitions=num_partitions, replication_factor=replication)
            futures = self._admin.create_topics([new_topic])
            futures[topic].result()
            log.info("Created topic: %s", topic)


class XTALActuator:
    """
    High-level actuator that validates safety bounds before applying any action.
    Week 3 agent calls apply_action(); Week 1 uses individual methods for testing.
    """

    def __init__(
        self,
        flink_url: str = FLINK_REST,
        kafka_bootstrap: str = KAFKA_BOOTSTRAP,
    ):
        self.flink = FlinkActuator(flink_url)
        self.kafka = KafkaActuator(kafka_bootstrap)

    def apply_action(self, action: ControlAction, job_id: str, kafka_topic: str) -> bool:
        """
        Applies a ControlAction after safety validation.
        Returns True if all sub-actions succeeded.
        """
        if not action.is_safe():
            log.error("Unsafe action rejected: %s", action)
            return False

        success = True

        if action.checkpoint_interval_ms is not None:
            try:
                self.flink.set_job_config_dynamic(
                    job_id,
                    "execution.checkpointing.interval",
                    str(action.checkpoint_interval_ms),
                )
            except Exception as e:
                log.warning("Checkpoint interval update failed: %s", e)
                success = False

        if action.kafka_linger_ms is not None:
            try:
                # linger.ms is a producer config; closest topic-level proxy is
                # min.insync.replicas or segment.ms — we store it as a tag for
                # the load generator to pick up via a shared config file.
                self._write_producer_hint("linger.ms", str(action.kafka_linger_ms))
            except Exception as e:
                log.warning("Kafka linger.ms update failed: %s", e)
                success = False

        return success

    def _write_producer_hint(self, key: str, value: str) -> None:
        """
        Writes producer config hints to a shared file watched by the load generator.
        A lightweight IPC mechanism for Week 1; replaced by a proper config bus in Week 3.
        """
        import json, pathlib
        hint_file = pathlib.Path("/tmp/xtal_producer_hints.json")
        hints = {}
        if hint_file.exists():
            hints = json.loads(hint_file.read_text())
        hints[key] = value
        hint_file.write_text(json.dumps(hints, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fa = FlinkActuator()
    jobs = fa.get_running_jobs()
    print("Running jobs:", jobs)
