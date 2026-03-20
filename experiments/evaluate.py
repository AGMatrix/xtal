"""
evaluate.py — Full Evaluation Runner.

Compares XTAL controller against 3 baselines under each load pattern:
  1. static-default    : Flink defaults, no adaptation
  2. static-hand-tuned : Best hand-tuned static config
  3. reactive-scaler   : Simple threshold-based reactive autoscaler (trigger on p99 > SLA)
  4. xtal              : Our XGBoost contextual bandit controller

Metrics collected:
  - p50 / p99 / p999 latency
  - SLA violation rate (% of 5s windows where p99 > SLA)
  - Tail amplification ratio (p999/p50)
  - Throughput stability (std dev of records/s)
  - Recovery time after failure injection

Usage:
    python -m experiments.evaluate --duration 300 --runs 3
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pathlib
import subprocess
import time
from datetime import datetime
from typing import Any

import numpy as np

from controller.telemetry import TelemetryCollector, SystemState

log = logging.getLogger(__name__)
RESULTS_DIR = pathlib.Path("experiments/results")


class ReactiveScaler:
    """
    Baseline 3: Threshold-based reactive autoscaler.
    When p99 > SLA, it reduces checkpoint interval and buffer timeout to "low-latency" preset.
    When p99 < 0.5*SLA, it relaxes back to defaults.
    A simple heuristic representative of rule-based production systems.
    """

    def __init__(self, sla_ms: float = 500.0, flink_url: str = "http://localhost:8081"):
        self.sla_ms = sla_ms
        from controller.actuator import XTALActuator, ControlAction
        self.actuator = XTALActuator(flink_url)
        self._mode = "relaxed"

    def step(self, state: SystemState, job_id: str) -> str:
        from controller.actuator import ControlAction
        if state.p99_latency_ms > self.sla_ms and self._mode != "aggressive":
            action = ControlAction(checkpoint_interval_ms=500, buffer_timeout_ms=0, kafka_linger_ms=0)
            self.actuator.apply_action(action, job_id, "nexmark-bids")
            self._mode = "aggressive"
            return "aggressive"
        elif state.p99_latency_ms < self.sla_ms * 0.5 and self._mode != "relaxed":
            action = ControlAction(checkpoint_interval_ms=5000, buffer_timeout_ms=100, kafka_linger_ms=5)
            self.actuator.apply_action(action, job_id, "nexmark-bids")
            self._mode = "relaxed"
            return "relaxed"
        return self._mode


def run_xtal_controller(
    duration_s: int,
    job_id: str,
    sla_ms: float,
    model_path: str,
) -> subprocess.Popen:
    """Starts the XTAL controller as a subprocess."""
    cmd = [
        "python", "-m", "controller.main",
        "--sla", str(sla_ms),
        "--interval", "10",
        "--job-id", job_id,
        "--model-path", model_path,
    ]
    return subprocess.Popen(cmd)


def collect_metrics_for_duration(
    system_name: str,
    load_pattern: str,
    duration_s: int,
    base_rate: int,
    collector: TelemetryCollector,
    sla_ms: float = 500.0,
    sample_interval_s: int = 5,
    job_id: str | None = None,
    reactive_scaler: ReactiveScaler | None = None,
) -> list[dict]:
    """Runs a load pattern and collects metrics. Returns list of row dicts."""
    rows = []
    start = time.time()

    load_proc = subprocess.Popen([
        "python", "scripts/load_generator.py",
        "--rate", str(base_rate),
        "--burst-at", "60" if "bursty" in load_pattern else "0",
        "--burst-rate", str(base_rate * 10),
        "--duration", str(duration_s + 10),
        "--topic", "nexmark-bids",
    ])

    failure_injected = False
    recovery_time_s = None

    try:
        while True:
            elapsed = time.time() - start
            if elapsed >= duration_s:
                break

            if "failure" in load_pattern and elapsed >= 90 and not failure_injected:
                log.warning("[%s] Injecting TaskManager failure at t=%.0fs", system_name, elapsed)
                subprocess.run(["docker", "stop", "xtal-taskmanager-3"], capture_output=True)
                failure_injected = True
                failure_start = elapsed

            state = collector.collect()

            if reactive_scaler and job_id:
                reactive_scaler.step(state, job_id)

            p50 = collector._query(
                "histogram_quantile(0.50, sum("
                "rate(flink_taskmanager_job_latency_source_id_operator_id_operator_subtask_index_bucket[30s])"
                ") by (le))"
            ) or 0.0

            # Detect recovery (p99 back below SLA after failure)
            if failure_injected and recovery_time_s is None and state.p99_latency_ms < sla_ms:
                recovery_time_s = elapsed - failure_start
                log.info("[%s] Recovered in %.1fs", system_name, recovery_time_s)

            row = {
                "system": system_name,
                "load_pattern": load_pattern,
                "elapsed_s": round(elapsed, 1),
                "p50_ms": round(p50, 2),
                "p99_ms": round(state.p99_latency_ms, 2),
                "p999_ms": round(state.p999_latency_ms, 2),
                "buffer_usage": round(state.input_buffer_usage, 3),
                "checkpoint_duration_ms": round(state.checkpoint_duration_ms, 1),
                "kafka_lag": round(state.kafka_consumer_lag, 0),
                "records_per_sec": round(state.records_in_per_sec, 1),
                "sla_violated": int(state.p99_latency_ms > sla_ms),
                "tail_amplification": round(state.p999_latency_ms / max(p50, 1), 2),
                "recovery_time_s": recovery_time_s,
            }
            rows.append(row)

            log.info("[%s/%s t=%.0fs] p99=%.1fms p999=%.1fms tps=%.0f %s",
                     system_name, load_pattern, elapsed,
                     state.p99_latency_ms, state.p999_latency_ms,
                     state.records_in_per_sec,
                     "SLA!" if row["sla_violated"] else "")

            time.sleep(sample_interval_s)

    finally:
        load_proc.terminate()
        if failure_injected:
            subprocess.run(["docker", "start", "xtal-taskmanager-3"], capture_output=True)

    return rows


def summarize(rows: list[dict]) -> dict[str, Any]:
    """Computes aggregate statistics for a set of rows."""
    if not rows:
        return {}
    p99s = [r["p99_ms"] for r in rows]
    p999s = [r["p999_ms"] for r in rows]
    p50s = [r["p50_ms"] for r in rows]
    tps = [r["records_per_sec"] for r in rows]

    recovery_times = [r["recovery_time_s"] for r in rows if r["recovery_time_s"] is not None]

    return {
        "system": rows[0]["system"],
        "load_pattern": rows[0]["load_pattern"],
        "p99_mean": round(float(np.mean(p99s)), 2),
        "p99_median": round(float(np.median(p99s)), 2),
        "p99_max": round(float(np.max(p99s)), 2),
        "p999_mean": round(float(np.mean(p999s)), 2),
        "p999_max": round(float(np.max(p999s)), 2),
        "p50_mean": round(float(np.mean(p50s)), 2),
        "tail_amplification_mean": round(float(np.mean([r["tail_amplification"] for r in rows])), 2),
        "sla_violation_rate": round(float(np.mean([r["sla_violated"] for r in rows])), 4),
        "throughput_mean": round(float(np.mean(tps)), 1),
        "throughput_std": round(float(np.std(tps)), 1),
        "recovery_time_s": round(float(np.mean(recovery_times)), 2) if recovery_times else None,
        "n_samples": len(rows),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="XTAL Full Evaluation")
    p.add_argument("--duration", type=int, default=300, help="Seconds per run")
    p.add_argument("--runs", type=int, default=3, help="Repetitions per condition")
    p.add_argument("--base-rate", type=int, default=5000)
    p.add_argument("--sla", type=float, default=500.0)
    p.add_argument("--flink-url", default="http://localhost:8081")
    p.add_argument("--prometheus-url", default="http://localhost:9090")
    p.add_argument("--patterns", nargs="+", default=["steady", "bursty", "failure"])
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    collector = TelemetryCollector(args.prometheus_url)

    from controller.actuator import XTALActuator, ControlAction
    actuator = XTALActuator(args.flink_url)
    job_id = actuator.flink.get_job_id()
    if job_id is None:
        log.error("No running Flink job. Exiting.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_rows: list[dict] = []
    summaries: list[dict] = []

    SYSTEMS = {
        "static-default": {
            "checkpoint_interval_ms": 5000,
            "buffer_timeout_ms": 100,
            "kafka_linger_ms": 5,
        },
        "static-hand-tuned": {
            "checkpoint_interval_ms": 2000,
            "buffer_timeout_ms": 10,
            "kafka_linger_ms": 5,
        },
    }

    for pattern in args.patterns:
        for run_i in range(args.runs):
            # ── Baselines ──────────────────────────────────────────────────────
            for sys_name, config in SYSTEMS.items():
                log.info("=== %s | %s | run %d ===", sys_name, pattern, run_i + 1)
                action = ControlAction(
                    checkpoint_interval_ms=config["checkpoint_interval_ms"],
                    buffer_timeout_ms=config["buffer_timeout_ms"],
                    kafka_linger_ms=config["kafka_linger_ms"],
                )
                actuator.apply_action(action, job_id, "nexmark-bids")
                time.sleep(10)

                rows = collect_metrics_for_duration(
                    system_name=sys_name,
                    load_pattern=pattern,
                    duration_s=args.duration,
                    base_rate=args.base_rate,
                    collector=collector,
                    sla_ms=args.sla,
                    job_id=job_id,
                )
                all_rows.extend(rows)
                summaries.append(summarize(rows))
                time.sleep(20)

            # ── Reactive scaler ────────────────────────────────────────────────
            log.info("=== reactive-scaler | %s | run %d ===", pattern, run_i + 1)
            reactive = ReactiveScaler(args.sla, args.flink_url)
            rows = collect_metrics_for_duration(
                system_name="reactive-scaler",
                load_pattern=pattern,
                duration_s=args.duration,
                base_rate=args.base_rate,
                collector=collector,
                sla_ms=args.sla,
                job_id=job_id,
                reactive_scaler=reactive,
            )
            all_rows.extend(rows)
            summaries.append(summarize(rows))
            time.sleep(20)

            # ── XTAL ──────────────────────────────────────────────────────────
            log.info("=== xtal | %s | run %d ===", pattern, run_i + 1)
            model_path = f"models/xtal_agent_run{run_i}.pkl"
            xtal_proc = run_xtal_controller(args.duration, job_id, args.sla, model_path)
            rows = collect_metrics_for_duration(
                system_name="xtal",
                load_pattern=pattern,
                duration_s=args.duration,
                base_rate=args.base_rate,
                collector=collector,
                sla_ms=args.sla,
                job_id=job_id,
            )
            xtal_proc.terminate()
            all_rows.extend(rows)
            summaries.append(summarize(rows))
            time.sleep(20)

    # Save raw rows
    raw_path = RESULTS_DIR / f"eval_raw_{ts}.csv"
    if all_rows:
        with open(raw_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        log.info("Raw results: %s", raw_path)

    # Save summary
    summary_path = RESULTS_DIR / f"eval_summary_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    log.info("Summary: %s", summary_path)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'System':<22} {'Pattern':<10} {'p99 mean':>10} {'p99 max':>10} {'SLA viol%':>10} {'Tail amp':>10}")
    print("=" * 80)
    for s in summaries:
        print(f"{s['system']:<22} {s['load_pattern']:<10} "
              f"{s['p99_mean']:>10.1f} {s['p99_max']:>10.1f} "
              f"{s['sla_violation_rate']*100:>9.1f}% "
              f"{s['tail_amplification_mean']:>10.1f}x")
    print("=" * 80)


if __name__ == "__main__":
    main()
