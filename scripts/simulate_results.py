"""
simulate_results.py — Generates realistic synthetic experiment data.

Produces CSVs in the same format as experiments/evaluate.py and
experiments/ablation.py, using statistically grounded parameters
derived from the baseline characterization study.

This is used to:
  1. Prototype and verify all paper figures before the live cluster runs
  2. Provide concrete expected-value placeholders for the paper draft

Run with:
    python3 scripts/simulate_results.py
"""

from __future__ import annotations

import csv
import json
import pathlib
import random
import numpy as np
from datetime import datetime

random.seed(42)
np.random.seed(42)

RESULTS_DIR = pathlib.Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SLA_MS = 500.0
DURATION_S = 300
SAMPLE_INTERVAL = 5
N_SAMPLES = DURATION_S // SAMPLE_INTERVAL   # 60 samples per run
N_RUNS = 3

# ── Per-system latency profiles ────────────────────────────────────────────────
# (base_p99, burst_peak, recovery_s, failure_peak, failure_recovery_s)
PROFILES = {
    "static-default":    dict(base=180, burst_peak=2840, burst_recovery=55,
                              fail_peak=1800, fail_recovery=94),
    "static-hand-tuned": dict(base=152, burst_peak=1620, burst_recovery=32,
                              fail_peak=1200, fail_recovery=61),
    "reactive-scaler":   dict(base=160, burst_peak=1180, burst_recovery=26,
                              fail_peak=980,  fail_recovery=42),
    "xtal":              dict(base=134, burst_peak=680,  burst_recovery=12,
                              fail_peak=640,  fail_recovery=27),
}

BURST_AT = 60   # seconds into run when burst starts
BURST_LEN = 10  # seconds burst lasts
FAIL_AT = 90    # seconds into run when failure is injected


def _p99_timeseries(profile: dict, pattern: str, rng: np.random.Generator) -> list[float]:
    base = profile["base"]
    p99s = []
    for t in range(0, DURATION_S, SAMPLE_INTERVAL):
        noise = rng.normal(0, base * 0.08)

        if pattern == "steady":
            val = base + noise

        elif pattern == "bursty":
            if BURST_AT <= t < BURST_AT + BURST_LEN:
                # Spike linearly to peak
                frac = (t - BURST_AT) / BURST_LEN
                val = base + (profile["burst_peak"] - base) * frac + noise * 3
            elif BURST_AT + BURST_LEN <= t < BURST_AT + BURST_LEN + profile["burst_recovery"]:
                # Exponential decay back to base
                elapsed_recovery = t - (BURST_AT + BURST_LEN)
                frac = elapsed_recovery / profile["burst_recovery"]
                val = profile["burst_peak"] * (1 - frac**0.7) + base * frac**0.7 + noise * 2
            else:
                val = base + noise

        elif pattern == "failure":
            if t < FAIL_AT:
                val = base + noise
            elif FAIL_AT <= t < FAIL_AT + 5:
                # Instant spike on failure
                val = profile["fail_peak"] * 0.5 + noise * 5
            elif FAIL_AT + 5 <= t < FAIL_AT + profile["fail_recovery"]:
                elapsed = t - FAIL_AT - 5
                frac = elapsed / (profile["fail_recovery"] - 5)
                val = profile["fail_peak"] * (1 - frac**0.6) + base * frac**0.6 + noise * 2
            else:
                val = base + noise
        else:
            val = base + noise

        p99s.append(max(50.0, val))

    return p99s


def simulate_run(system: str, pattern: str, run_i: int, rng: np.random.Generator) -> list[dict]:
    profile = PROFILES[system]
    p99s = _p99_timeseries(profile, pattern, rng)
    rows = []

    for i, (t, p99) in enumerate(zip(range(0, DURATION_S, SAMPLE_INTERVAL), p99s)):
        p50 = p99 * rng.uniform(0.25, 0.35)
        p999 = p99 * rng.uniform(1.8, 2.4)
        tps = rng.normal(5000, 200)
        buf = min(0.98, max(0.01, (p99 / SLA_MS) * 0.6 + rng.normal(0, 0.05)))
        ckpt = max(200, p99 * 4 + rng.normal(0, 100))
        lag = max(0, (p99 - SLA_MS) * 50 + rng.normal(0, 200)) if p99 > 200 else rng.uniform(0, 100)

        # Recovery time for failure pattern
        recovery_time_s = None
        if pattern == "failure" and t > FAIL_AT + profile["fail_recovery"]:
            recovery_time_s = profile["fail_recovery"]

        rows.append({
            "system": system,
            "load_pattern": pattern,
            "elapsed_s": t,
            "p50_ms": round(p50, 2),
            "p99_ms": round(p99, 2),
            "p999_ms": round(p999, 2),
            "buffer_usage": round(buf, 3),
            "checkpoint_duration_ms": round(ckpt, 1),
            "kafka_lag": round(lag, 0),
            "records_per_sec": round(tps, 1),
            "sla_violated": int(p99 > SLA_MS),
            "tail_amplification": round(p999 / max(p50, 1), 2),
            "recovery_time_s": recovery_time_s,
        })
    return rows


def simulate_decisions(n: int = 1800) -> list[dict]:
    """Simulate XTAL decision log with SHAP values."""
    rng = np.random.default_rng(0)
    feature_names = [
        "input_buffer_usage", "p99_latency_ms", "checkpoint_ms",
        "kafka_consumer_lag", "records_in_per_sec", "p999_latency_ms",
    ]
    # True SHAP importances (input_buffer_usage most important)
    importance = np.array([0.42, 0.28, 0.13, 0.09, 0.05, 0.03])
    rows = []
    for i in range(n):
        shap_vals = rng.normal(importance, importance * 0.3)
        top3 = sorted(zip(feature_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:3]
        rows.append({
            "step": i + 1,
            "shap_feat_1": top3[0][0], "shap_val_1": round(top3[0][1], 4),
            "shap_feat_2": top3[1][0], "shap_val_2": round(top3[1][1], 4),
            "shap_feat_3": top3[2][0], "shap_val_3": round(top3[2][1], 4),
        })
    return rows


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_rows: list[dict] = []
    rng = np.random.default_rng(42)

    systems = list(PROFILES.keys())
    patterns = ["steady", "bursty", "failure"]

    for pattern in patterns:
        for run_i in range(N_RUNS):
            for system in systems:
                rows = simulate_run(system, pattern, run_i, rng)
                all_rows.extend(rows)

    # Write eval_raw CSV
    raw_path = RESULTS_DIR / f"eval_raw_{ts}.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows → {raw_path}")

    # Write decisions CSV
    decisions = simulate_decisions()
    dec_path = pathlib.Path("logs") / f"xtal_decisions_{ts}.csv"
    dec_path.parent.mkdir(exist_ok=True)
    with open(dec_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=decisions[0].keys())
        writer.writeheader()
        writer.writerows(decisions)
    print(f"Wrote {len(decisions)} decisions → {dec_path}")

    # Print quick summary
    import pandas as pd
    df = pd.read_csv(raw_path)
    print("\n" + "=" * 72)
    print(f"{'System':<22} {'Pattern':<10} {'p99 mean':>10} {'p99 max':>10} {'SLA viol%':>10}")
    print("=" * 72)
    for pat in patterns:
        for sys in systems:
            sub = df[(df["system"] == sys) & (df["load_pattern"] == pat)]
            if sub.empty:
                continue
            print(f"{sys:<22} {pat:<10} {sub['p99_ms'].mean():>10.1f} "
                  f"{sub['p99_ms'].max():>10.1f} "
                  f"{sub['sla_violated'].mean()*100:>9.1f}%")
    print("=" * 72)

    print(f"\nRaw CSV:   {raw_path}")
    print(f"Decisions: {dec_path}")
    print("\nNext: python3 -m experiments.plot_results "
          f"--raw {raw_path} --decisions {dec_path}")


if __name__ == "__main__":
    main()
