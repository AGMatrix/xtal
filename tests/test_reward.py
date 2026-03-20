"""Unit tests for controller/reward.py."""

import pytest
from controller.reward import compute_reward, latency_improvement
from controller.telemetry import SystemState
import time


def make_state(**kwargs) -> SystemState:
    defaults = dict(
        timestamp=time.time(),
        p99_latency_ms=200.0,
        p999_latency_ms=400.0,
        input_buffer_usage=0.2,
        checkpoint_duration_ms=800.0,
        kafka_consumer_lag=0.0,
        records_in_per_sec=5000.0,
    )
    defaults.update(kwargs)
    return SystemState(**defaults)


class TestComputeReward:
    def test_healthy_state_gives_negative_reward_below_one(self):
        prev = make_state(p99_latency_ms=200.0, records_in_per_sec=5000.0)
        curr = make_state(p99_latency_ms=200.0, records_in_per_sec=5000.0)
        r = compute_reward(prev, curr, sla_ms=500.0)
        # latency_reward = -(200/500) = -0.4, no penalties
        assert -1.0 < r < 0.0

    def test_sla_violation_applies_large_penalty(self):
        prev = make_state(p99_latency_ms=300.0, records_in_per_sec=5000.0)
        curr = make_state(p99_latency_ms=600.0, records_in_per_sec=5000.0)
        r = compute_reward(prev, curr, sla_ms=500.0)
        assert r < -10.0, "SLA violation should incur heavy penalty"

    def test_no_sla_penalty_when_within_sla(self):
        prev = make_state(p99_latency_ms=200.0)
        curr = make_state(p99_latency_ms=490.0)
        r = compute_reward(prev, curr, sla_ms=500.0)
        assert r > -5.0, "Should not apply SLA penalty when p99 < SLA"

    def test_high_backpressure_incurs_penalty(self):
        prev = make_state(input_buffer_usage=0.2)
        curr_low = make_state(input_buffer_usage=0.2)
        curr_high = make_state(input_buffer_usage=0.95)
        r_low = compute_reward(prev, curr_low)
        r_high = compute_reward(prev, curr_high)
        assert r_high < r_low, "High backpressure should reduce reward"

    def test_checkpoint_storm_incurs_penalty(self):
        prev = make_state()
        curr_normal = make_state(checkpoint_duration_ms=1000.0)
        curr_storm = make_state(checkpoint_duration_ms=12000.0)
        r_normal = compute_reward(prev, curr_normal)
        r_storm = compute_reward(prev, curr_storm)
        assert r_storm < r_normal, "Checkpoint storm should reduce reward"

    def test_throughput_drop_reduces_reward(self):
        prev = make_state(records_in_per_sec=5000.0)
        curr_stable = make_state(records_in_per_sec=5000.0)
        curr_dropped = make_state(records_in_per_sec=1000.0)
        r_stable = compute_reward(prev, curr_stable)
        r_dropped = compute_reward(prev, curr_dropped)
        assert r_dropped < r_stable, "Throughput drop should reduce reward"

    def test_reward_is_finite_for_extreme_latency(self):
        prev = make_state()
        curr = make_state(p99_latency_ms=999999.0)
        r = compute_reward(prev, curr)
        assert not (r != r), "Reward must not be NaN"
        assert r > float("-inf"), "Reward must not be -inf"


class TestLatencyImprovement:
    def test_improvement_positive_when_latency_decreases(self):
        prev = make_state(p99_latency_ms=400.0)
        curr = make_state(p99_latency_ms=200.0)
        assert latency_improvement(prev, curr) > 0

    def test_improvement_negative_when_latency_increases(self):
        prev = make_state(p99_latency_ms=200.0)
        curr = make_state(p99_latency_ms=400.0)
        assert latency_improvement(prev, curr) < 0

    def test_improvement_zero_when_prev_is_zero(self):
        prev = make_state(p99_latency_ms=0.0)
        curr = make_state(p99_latency_ms=200.0)
        assert latency_improvement(prev, curr) == 0.0
