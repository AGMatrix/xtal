"""
baseline_runner.py — Baseline Characterization.

Runs the streaming pipeline under 4 static configurations and 3 load patterns,
recording p50/p99/p999 latency, throughput, and checkpoint metrics.
Results are saved to experiments/results/baseline_<timestamp>.csv.

Configurations tested:
  1. flink-default   : Flink defaults (checkpoint=5s, buffer=100ms)
  2. low-latency     : Low checkpoint interval (500ms), no buffer timeout
  3. high-throughput : Large checkpoint interval (10s), large buffer timeout
  4. hand-tuned      : Best static config found by manual tuning

Load patterns:
  A. steady    : constant 5k events/s
  B. bursty    : 5k/s with 10x spike at t=30s for 10s
  C. failure   : steady + kill one TaskManager at t=45s

Usage:
    python -m experiments.baseline_runner --duration 120 --rates 5000
"""

from __future__ import annotations

import argparse
import csv
import logging
import pathlib
import subprocess
import time
from datetime import datetime
from typing import NamedTuple

from controller.telemetry import TelemetryCollector, SystemState

log = logging.getLogger(__name__)

RESULTS_DIR = pathlib.Path("experiments/results")

STATIC_CONFIGS = {
    "flink-default": {
        "checkpoint_interval_ms": 5000,
        "buffer_timeout_ms": 100,
        "kafka_linger_ms": 5,
        "description": "Flink 1.18 out-of-the-box defaults",
    },
    "low-latency": {
        "checkpoint_interval_ms": 500,
        "buffer_timeout_ms": 0,
        "kafka_linger_ms": 0,
        "description": "Aggressively tuned for minimum latency",
    },
    "high-throughput": {
        "checkpoint_interval_ms": 10000,
        "buffer_timeout_ms": 100,
        "kafka_linger_ms": 50,
        "description": "Tuned for maximum throughput",
    },
    "hand-tuned": {
        "checkpoint_interval_ms": 2000,
        "buffer_timeout_ms": 10,
        "kafka_linger_ms": 5,
        "description": "Best static config from manual tuning experiments",
    },
}


class MetricSnapshot(NamedTuple):
    timestamp: float
    config_name: str
    load_pattern: str
    elapsed_s: float
    p99_latency_ms: float
    p999_latency_ms: float
    p50_latency_ms: float
    input_buffer_usage: float
    checkpoint_duration_ms: float
    kafka_consumer_lag: float
    records_in_per_sec: float
    sla_violated: int


def apply_static_config(config: dict, flink_url: str = "http://localhost:8081") -> None:
    """
    Applies a static configuration by updating the Kafka/Flink config hint files.
    For a real deployment, this would restart Flink with the new config.
    For baseline experiments, we write hints and note when the job was reconfigured.
    """
    import json
    hint = {
        "checkpoint_interval_ms": config["checkpoint_interval_ms"],
        "buffer_timeout_ms": config["buffer_timeout_ms"],
        "kafka_linger_ms": config["kafka_linger_ms"],
    }
    pathlib.Path("/tmp/xtal_static_config.json").write_text(json.dumps(hint, indent=2))
    pathlib.Path("/tmp/xtal_producer_hints.json").write_text(json.dumps(
        {"linger.ms": str(config["kafka_linger_ms"])}, indent=2
    ))
    log.info("Applied config: %s", config)


def inject_burst(rate: int = 50_000, duration_s: int = 10) -> subprocess.Popen:
    """Starts a burst load generator process."""
    cmd = [
        "python", "scripts/load_generator.py",
        "--rate", str(rate),
        "--duration", str(duration_s),
        "--topic", "nexmark-bids",
    ]
    log.info("Injecting burst: %d events/s for %ds", rate, duration_s)
    return subprocess.Popen(cmd)


def kill_taskmanager(tm_name: str = "xtal-taskmanager-3") -> None:
    """Kills a Flink TaskManager container to simulate node failure."""
    log.warning("FAILURE INJECTION: killing %s", tm_name)
    subprocess.run(["docker", "stop", tm_name], check=False, capture_output=True)


def restart_taskmanager(tm_name: str = "xtal-taskmanager-3") -> None:
    """Restarts the killed TaskManager."""
    log.info("Restarting %s", tm_name)
    subprocess.run(["docker", "start", tm_name], check=False, capture_output=True)


def run_experiment(
    config_name: str,
    load_pattern: str,
    duration_s: int,
    base_rate: int,
    collector: TelemetryCollector,
    sample_interval_s: int = 5,
    sla_ms: float = 500.0,
) -> list[MetricSnapshot]:
    """Runs a single (config, load_pattern) experiment and returns metric snapshots."""
    snapshots: list[MetricSnapshot] = []
    start = time.time()

    # Start baseline load generator in background
    load_proc = subprocess.Popen([
        "python", "scripts/load_generator.py",
        "--rate", str(base_rate),
        "--duration", str(duration_s + 10),
        "--topic", "nexmark-bids",
    ])

    burst_proc = None
    failure_injected = False
    failure_recovered = False

    try:
        while True:
            elapsed = time.time() - start
            if elapsed >= duration_s:
                break

            # Load pattern events
            if load_pattern == "bursty" and elapsed >= 30 and burst_proc is None:
                burst_proc = inject_burst(rate=base_rate * 10, duration_s=10)

            if load_pattern == "failure" and elapsed >= 45 and not failure_injected:
                kill_taskmanager()
                failure_injected = True

            if load_pattern == "failure" and failure_injected and elapsed >= 75 and not failure_recovered:
                restart_taskmanager()
                failure_recovered = True

            # Sample metrics
            state: SystemState = collector.collect()

            # Query p50 separately
            p50 = collector._query(
                "histogram_quantile(0.50, sum("
                "rate(flink_taskmanager_job_latency_source_id_operator_id_operator_subtask_index_bucket[30s])"
                ") by (le))"
            ) or 0.0

            snap = MetricSnapshot(
                timestamp=state.timestamp,
                config_name=config_name,
                load_pattern=load_pattern,
                elapsed_s=elapsed,
                p99_latency_ms=state.p99_latency_ms,
                p999_latency_ms=state.p999_latency_ms,
                p50_latency_ms=p50,
                input_buffer_usage=state.input_buffer_usage,
                checkpoint_duration_ms=state.checkpoint_duration_ms,
                kafka_consumer_lag=state.kafka_consumer_lag,
                records_in_per_sec=state.records_in_per_sec,
                sla_violated=int(state.p99_latency_ms > sla_ms),
            )
            snapshots.append(snap)

            log.info("[%s/%s t=%.0fs] p99=%.1fms p999=%.1fms tps=%.0f sla=%s",
                     config_name, load_pattern, elapsed,
                     snap.p99_latency_ms, snap.p999_latency_ms, snap.records_in_per_sec,
                     "VIOLATED" if snap.sla_violated else "OK")

            time.sleep(sample_interval_s)

    finally:
        load_proc.terminate()
        if burst_proc:
            burst_proc.terminate()
        if failure_injected and not failure_recovered:
            restart_taskmanager()

    return snapshots


def save_results(snapshots: list[MetricSnapshot], output_path: pathlib.Path) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(MetricSnapshot._fields)
        writer.writerows(snapshots)
    log.info("Results saved to %s (%d rows)", output_path, len(snapshots))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="XTAL Baseline Characterization")
    p.add_argument("--duration", type=int, default=120, help="Seconds per experiment")
    p.add_argument("--rates", type=int, default=5000, help="Baseline events/s")
    p.add_argument("--sla", type=float, default=500.0)
    p.add_argument("--prometheus-url", default="http://localhost:9090")
    p.add_argument("--configs", nargs="+", default=list(STATIC_CONFIGS.keys()))
    p.add_argument("--patterns", nargs="+", default=["steady", "bursty", "failure"])
    args = p.parse_args()

    collector = TelemetryCollector(args.prometheus_url)
    all_snapshots: list[MetricSnapshot] = []

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"baseline_{ts}.csv"

    total = len(args.configs) * len(args.patterns)
    run_num = 0
    for config_name in args.configs:
        config = STATIC_CONFIGS[config_name]
        apply_static_config(config)
        time.sleep(5)  # allow config to stabilize

        for pattern in args.patterns:
            run_num += 1
            log.info("=" * 60)
            log.info("Run %d/%d: config=%s pattern=%s", run_num, total, config_name, pattern)
            log.info("=" * 60)

            snapshots = run_experiment(
                config_name=config_name,
                load_pattern=pattern,
                duration_s=args.duration,
                base_rate=args.rates,
                collector=collector,
                sla_ms=args.sla,
            )
            all_snapshots.extend(snapshots)
            time.sleep(15)  # cooldown between runs

    save_results(all_snapshots, output_path)
    log.info("Baseline characterization complete. Results: %s", output_path)


if __name__ == "__main__":
    main()
