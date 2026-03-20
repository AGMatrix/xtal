"""
main.py — XTAL Control Loop.

Runs the closed-loop controller:
  1. Collect system state from Prometheus
  2. Agent selects action + generates SHAP explanation
  3. Safety check
  4. Apply action via Flink REST + Kafka Admin APIs
  5. Wait for control interval
  6. Collect new state, compute reward, update agent

Usage:
    python -m controller.main --sla 500 --interval 10 --job-id <flink-job-id>
    python -m controller.main --sla 500 --interval 10  # auto-detect job ID
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pathlib
import signal
import time
from datetime import datetime

from .agent import XTALAgent, ActionExplanation
from .actuator import XTALActuator, ControlAction
from .telemetry import TelemetryCollector, SystemState
from .reward import compute_reward

log = logging.getLogger(__name__)

CONTROL_INTERVAL_S = 10      # seconds between control decisions
DEFAULT_KAFKA_TOPIC = "nexmark-bids"
LOG_DIR = pathlib.Path("logs")


class XTALController:
    def __init__(
        self,
        sla_ms: float = 500.0,
        control_interval: int = CONTROL_INTERVAL_S,
        flink_url: str = "http://localhost:8081",
        prometheus_url: str = "http://localhost:9090",
        kafka_bootstrap: str = "localhost:9092",
        kafka_topic: str = DEFAULT_KAFKA_TOPIC,
        model_path: str = "models/xtal_agent.pkl",
        log_dir: pathlib.Path = LOG_DIR,
    ):
        self.sla_ms = sla_ms
        self.control_interval = control_interval
        self.kafka_topic = kafka_topic
        self.model_path = model_path
        self.log_dir = log_dir

        self.telemetry = TelemetryCollector(prometheus_url)
        self.actuator = XTALActuator(flink_url, kafka_bootstrap)
        self.agent = XTALAgent(sla_ms=sla_ms, model_path=model_path)

        self._running = False
        self._step = 0
        self._csv_writer = None
        self._csv_file = None

        log_dir.mkdir(parents=True, exist_ok=True)
        pathlib.Path("models").mkdir(exist_ok=True)

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self, job_id: str | None = None) -> None:
        self._running = True
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        # Auto-detect Flink job
        if job_id is None:
            job_id = self.actuator.flink.get_job_id()
            if job_id is None:
                log.error("No running Flink job found. Submit a job first.")
                return
        log.info("XTAL controller starting | job=%s | SLA=%.0fms | interval=%ds",
                 job_id, self.sla_ms, self.control_interval)

        # Open decision log CSV
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"xtal_decisions_{ts}.csv"
        self._csv_file = open(log_file, "w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_columns())
        self._csv_writer.writeheader()
        log.info("Decision log: %s", log_file)

        prev_state: SystemState | None = None

        while self._running:
            step_start = time.monotonic()
            self._step += 1

            # 1. Observe
            state = self.telemetry.collect()
            log.info("[step %d] p99=%.1fms buf=%.2f ckpt=%.0fms lag=%.0f tps=%.0f",
                     self._step, state.p99_latency_ms, state.input_buffer_usage,
                     state.checkpoint_duration_ms, state.kafka_consumer_lag,
                     state.records_in_per_sec)

            # 2. Compute reward from previous transition
            if prev_state is not None:
                reward = compute_reward(prev_state, state, self.sla_ms)
                self.agent.record(prev_state, self._last_action_idx, reward, state)
                log.info("  reward=%.3f", reward)

            # 3. Select action
            action_idx, explanation = self.agent.select_action(state)
            self._last_action_idx = action_idx
            ckpt_ms, buf_ms, linger_ms = explanation.action

            log.info("  %s", explanation.human_readable().replace("\n", "\n  "))

            # 4. Build and apply ControlAction
            action = ControlAction(
                checkpoint_interval_ms=ckpt_ms,
                buffer_timeout_ms=buf_ms,
                kafka_linger_ms=linger_ms,
            )
            ok = self.actuator.apply_action(action, job_id, self.kafka_topic)
            if not ok:
                log.warning("  Action application failed — skipping step")

            # 5. Log decision
            self._log_decision(state, action_idx, explanation, reward if prev_state else None)

            # 6. Checkpoint agent periodically
            if self._step % 50 == 0:
                self.agent.save(self.model_path)
                summary = self.agent.experience_summary()
                log.info("Agent summary: %s", json.dumps(summary, indent=2))

            prev_state = state

            # Sleep for remainder of control interval
            elapsed = time.monotonic() - step_start
            sleep_time = max(0.0, self.control_interval - elapsed)
            time.sleep(sleep_time)

        self._cleanup()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _csv_columns(self) -> list[str]:
        return [
            "step", "timestamp",
            "p99_latency_ms", "p999_latency_ms", "input_buffer_usage",
            "checkpoint_duration_ms", "kafka_consumer_lag", "records_in_per_sec",
            "action_idx", "checkpoint_ms", "buffer_timeout_ms", "kafka_linger_ms",
            "predicted_reward", "actual_reward",
            "shap_feat_1", "shap_val_1", "shap_feat_2", "shap_val_2", "shap_feat_3", "shap_val_3",
            "sla_violated",
        ]

    def _log_decision(
        self,
        state: SystemState,
        action_idx: int,
        explanation: ActionExplanation,
        reward: float | None,
    ) -> None:
        if self._csv_writer is None:
            return

        row: dict = {
            "step": self._step,
            "timestamp": state.timestamp,
            "p99_latency_ms": state.p99_latency_ms,
            "p999_latency_ms": state.p999_latency_ms,
            "input_buffer_usage": state.input_buffer_usage,
            "checkpoint_duration_ms": state.checkpoint_duration_ms,
            "kafka_consumer_lag": state.kafka_consumer_lag,
            "records_in_per_sec": state.records_in_per_sec,
            "action_idx": action_idx,
            "checkpoint_ms": explanation.action[0],
            "buffer_timeout_ms": explanation.action[1],
            "kafka_linger_ms": explanation.action[2],
            "predicted_reward": explanation.predicted_reward,
            "actual_reward": reward,
            "sla_violated": int(state.p99_latency_ms > self.sla_ms),
        }

        for i, (feat, val) in enumerate(explanation.top_features[:3], start=1):
            row[f"shap_feat_{i}"] = feat
            row[f"shap_val_{i}"] = val

        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def _shutdown(self, *_) -> None:
        log.info("Shutdown signal received. Saving agent...")
        self._running = False

    def _cleanup(self) -> None:
        self.agent.save(self.model_path)
        if self._csv_file:
            self._csv_file.close()
        log.info("XTAL controller stopped after %d steps.", self._step)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    p = argparse.ArgumentParser(description="XTAL Tail-Latency Controller")
    p.add_argument("--sla", type=float, default=500.0, help="p99 SLA deadline in ms")
    p.add_argument("--interval", type=int, default=10, help="Control interval in seconds")
    p.add_argument("--job-id", default=None, help="Flink job ID (auto-detected if omitted)")
    p.add_argument("--flink-url", default="http://localhost:8081")
    p.add_argument("--prometheus-url", default="http://localhost:9090")
    p.add_argument("--kafka-bootstrap", default="localhost:9092")
    p.add_argument("--kafka-topic", default="nexmark-bids")
    p.add_argument("--model-path", default="models/xtal_agent.pkl")
    args = p.parse_args()

    controller = XTALController(
        sla_ms=args.sla,
        control_interval=args.interval,
        flink_url=args.flink_url,
        prometheus_url=args.prometheus_url,
        kafka_bootstrap=args.kafka_bootstrap,
        kafka_topic=args.kafka_topic,
        model_path=args.model_path,
    )
    controller.run(job_id=args.job_id)


if __name__ == "__main__":
    main()
