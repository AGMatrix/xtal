"""
reward.py — Reward function for the XTAL control agent.

Design goals:
  - Primary: minimize p99 latency
  - Hard constraint: p99 must stay below SLA deadline (default 500ms)
  - Secondary: don't sacrifice throughput
  - Penalty: checkpoint storms (very long checkpoint duration) indicate instability
"""

from __future__ import annotations
from .telemetry import SystemState


SLA_MS = 500.0          # p99 deadline (ms) — matches strict-deadline setting in paper
SLA_PENALTY = -20.0     # applied whenever p99 > SLA_MS
MAX_LATENCY_MS = 5000.0 # clip extreme values to prevent reward explosion


def compute_reward(prev: SystemState, curr: SystemState, sla_ms: float = SLA_MS) -> float:
    """
    Computes scalar reward signal for the (prev_state → action → curr_state) transition.

    Returns a value in roughly [-25, +1].
    Positive reward only when p99 < SLA and throughput is healthy.
    """
    # ── Latency term (primary) ────────────────────────────────────────────────
    p99 = min(curr.p99_latency_ms, MAX_LATENCY_MS)
    latency_reward = -(p99 / sla_ms)          # 0 → -1 when at SLA, worse beyond

    # ── SLA violation penalty ─────────────────────────────────────────────────
    sla_penalty = SLA_PENALTY if curr.p99_latency_ms > sla_ms else 0.0

    # ── Throughput term (secondary, weighted low) ─────────────────────────────
    # Reward improvement in throughput; penalise drops > 20%
    if prev.records_in_per_sec > 0:
        throughput_ratio = curr.records_in_per_sec / prev.records_in_per_sec
        throughput_reward = 0.5 * min(throughput_ratio - 1.0, 0.5)
    else:
        throughput_reward = 0.0

    # ── Checkpoint storm penalty ──────────────────────────────────────────────
    # Very long checkpoints block operator progress and inflate tail latency
    ckpt_penalty = -1.0 if curr.checkpoint_duration_ms > 10_000 else 0.0

    # ── Backpressure penalty ──────────────────────────────────────────────────
    bp_penalty = -2.0 if curr.input_buffer_usage > 0.9 else 0.0

    return latency_reward + sla_penalty + throughput_reward + ckpt_penalty + bp_penalty


def latency_improvement(prev: SystemState, curr: SystemState) -> float:
    """Returns fractional improvement in p99 latency (positive = better)."""
    if prev.p99_latency_ms == 0:
        return 0.0
    return (prev.p99_latency_ms - curr.p99_latency_ms) / prev.p99_latency_ms
