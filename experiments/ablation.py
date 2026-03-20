"""
ablation.py — Ablation Study for XTAL.

Compares XTAL (full) against three ablated variants to isolate the contribution
of each component. Used in the paper's evaluation section.

Variants:
  xtal-full           : Full system (XGBoost bandit + SHAP + safety layer + cross-layer)
  xtal-no-safety      : Removes safety override under SLA pressure
  xtal-single-layer   : Tunes only Flink (not Kafka linger.ms)
  xtal-no-shap        : Removes SHAP but keeps XGBoost — functionally identical
                        for performance, but SHAP column in log is all zeros
                        (used to measure SHAP computation overhead)

Usage:
    python -m experiments.ablation --duration 180 --runs 2 --patterns bursty failure
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

import numpy as np

from controller.telemetry import TelemetryCollector, SystemState
from controller.agent import XTALAgent, ACTION_CONFIGS, N_ACTIONS
from controller.actuator import XTALActuator, ControlAction
from controller.reward import compute_reward

log = logging.getLogger(__name__)
RESULTS_DIR = pathlib.Path("experiments/results")


# ── Ablated agent variants ────────────────────────────────────────────────────

class XTALNoSafety(XTALAgent):
    """Removes the SLA-pressure safety override from action selection."""

    def select_action(self, state: SystemState):
        self._step += 1
        import numpy as np
        import pandas as pd
        from controller.agent import ActionExplanation

        if self._rng.random() < self._epsilon or self._model is None:
            action_idx = int(self._rng.integers(0, N_ACTIONS))
            return action_idx, ActionExplanation(
                action_idx=action_idx, action=ACTION_CONFIGS[action_idx],
                predicted_reward=0.0, top_features=[("exploration", 0.0)]
            )

        state_vec = state.as_feature_vector()
        rows = [self._build_row_from_list(state_vec, i) for i in range(N_ACTIONS)]

        from controller.agent import ALL_FEATURES
        X = pd.DataFrame(rows, columns=ALL_FEATURES)
        predicted_rewards = self._model.predict(X)
        # NO safety override — raw argmax
        best_idx = int(np.argmax(predicted_rewards))
        explanation = self._explain(X.iloc[best_idx], best_idx, predicted_rewards[best_idx])
        return best_idx, explanation


class XTALSingleLayer(XTALAgent):
    """Tunes only Flink knobs; Kafka linger is always set to 0."""

    def select_action(self, state: SystemState):
        idx, explanation = super().select_action(state)
        # Force kafka_linger to 0 regardless of chosen action
        ckpt, buf, _ = explanation.action
        from controller.agent import ActionExplanation
        modified = ActionExplanation(
            action_idx=idx,
            action=(ckpt, buf, 0),
            predicted_reward=explanation.predicted_reward,
            top_features=explanation.top_features,
        )
        return idx, modified


# ── Experiment runner ─────────────────────────────────────────────────────────

VARIANTS = {
    "xtal-full":         lambda sla: XTALAgent(sla_ms=sla, seed=42),
    "xtal-no-safety":    lambda sla: XTALNoSafety(sla_ms=sla, seed=42),
    "xtal-single-layer": lambda sla: XTALSingleLayer(sla_ms=sla, seed=42),
}


def run_ablation_variant(
    variant_name: str,
    agent: XTALAgent,
    load_pattern: str,
    duration_s: int,
    base_rate: int,
    collector: TelemetryCollector,
    actuator: XTALActuator,
    job_id: str,
    sla_ms: float = 500.0,
    sample_interval_s: int = 5,
) -> list[dict]:
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
    prev_state: SystemState | None = None

    try:
        while True:
            elapsed = time.time() - start
            if elapsed >= duration_s:
                break

            if "failure" in load_pattern and elapsed >= 60 and not failure_injected:
                log.warning("[%s] Injecting failure at t=%.0fs", variant_name, elapsed)
                subprocess.run(["docker", "stop", "xtal-taskmanager-3"], capture_output=True)
                failure_injected = True
                failure_start = elapsed

            state = collector.collect()
            action_idx, explanation = agent.select_action(state)

            # Apply action
            ckpt, buf, linger = explanation.action
            action = ControlAction(
                checkpoint_interval_ms=ckpt,
                buffer_timeout_ms=buf,
                kafka_linger_ms=linger,
            )
            actuator.apply_action(action, job_id, "nexmark-bids")

            # Compute reward
            reward = compute_reward(prev_state, state, sla_ms) if prev_state else 0.0
            agent.record(state if prev_state is None else prev_state,
                         action_idx, reward, state)

            p50 = collector._query(
                "histogram_quantile(0.50, sum("
                "rate(flink_taskmanager_job_latency_source_id_operator_id_operator_subtask_index_bucket[30s])"
                ") by (le))"
            ) or 0.0

            rows.append({
                "variant": variant_name,
                "load_pattern": load_pattern,
                "elapsed_s": round(elapsed, 1),
                "p50_ms": round(p50, 2),
                "p99_ms": round(state.p99_latency_ms, 2),
                "p999_ms": round(state.p999_latency_ms, 2),
                "sla_violated": int(state.p99_latency_ms > sla_ms),
                "tail_amplification": round(state.p999_latency_ms / max(p50, 1), 2),
                "action_idx": action_idx,
                "checkpoint_ms": ckpt,
                "buffer_timeout_ms": buf,
                "kafka_linger_ms": linger,
                "reward": round(reward, 4),
            })

            log.info("[%s/%s t=%.0fs] p99=%.1fms action=%d reward=%.3f",
                     variant_name, load_pattern, elapsed,
                     state.p99_latency_ms, action_idx, reward)

            prev_state = state
            time.sleep(sample_interval_s)

    finally:
        load_proc.terminate()
        if failure_injected:
            subprocess.run(["docker", "start", "xtal-taskmanager-3"], capture_output=True)

    return rows


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    p99s = [r["p99_ms"] for r in rows]
    return {
        "variant": rows[0]["variant"],
        "load_pattern": rows[0]["load_pattern"],
        "p99_mean": round(float(np.mean(p99s)), 2),
        "p99_max": round(float(np.max(p99s)), 2),
        "p99_median": round(float(np.median(p99s)), 2),
        "sla_violation_rate": round(float(np.mean([r["sla_violated"] for r in rows])), 4),
        "tail_amp_mean": round(float(np.mean([r["tail_amplification"] for r in rows])), 2),
        "n_samples": len(rows),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="XTAL Ablation Study")
    p.add_argument("--duration", type=int, default=180)
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--base-rate", type=int, default=5000)
    p.add_argument("--sla", type=float, default=500.0)
    p.add_argument("--flink-url", default="http://localhost:8081")
    p.add_argument("--prometheus-url", default="http://localhost:9090")
    p.add_argument("--patterns", nargs="+", default=["bursty", "failure"])
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    collector = TelemetryCollector(args.prometheus_url)
    actuator = XTALActuator(args.flink_url)
    job_id = actuator.flink.get_job_id()
    if job_id is None:
        log.error("No running Flink job. Submit a job first.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_rows: list[dict] = []
    summaries: list[dict] = []

    for pattern in args.patterns:
        for run_i in range(args.runs):
            for variant_name, agent_factory in VARIANTS.items():
                log.info("=== ABLATION: %s | %s | run %d ===", variant_name, pattern, run_i + 1)
                agent = agent_factory(args.sla)
                rows = run_ablation_variant(
                    variant_name=variant_name,
                    agent=agent,
                    load_pattern=pattern,
                    duration_s=args.duration,
                    base_rate=args.base_rate,
                    collector=collector,
                    actuator=actuator,
                    job_id=job_id,
                    sla_ms=args.sla,
                )
                all_rows.extend(rows)
                summaries.append(summarize(rows))
                time.sleep(15)

    # Save raw
    raw_path = RESULTS_DIR / f"ablation_raw_{ts}.csv"
    if all_rows:
        with open(raw_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        log.info("Raw ablation results: %s", raw_path)

    # Save summary
    summary_path = RESULTS_DIR / f"ablation_summary_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    log.info("Summary: %s", summary_path)

    # Print table
    print("\n" + "=" * 75)
    print(f"{'Variant':<22} {'Pattern':<10} {'p99 mean':>10} {'p99 max':>10} {'SLA viol%':>10}")
    print("=" * 75)
    for s in summaries:
        print(f"{s['variant']:<22} {s['load_pattern']:<10} "
              f"{s['p99_mean']:>10.1f} {s['p99_max']:>10.1f} "
              f"{s['sla_violation_rate']*100:>9.1f}%")
    print("=" * 75)


if __name__ == "__main__":
    main()
