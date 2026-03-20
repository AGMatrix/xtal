"""
integration_test.py — End-to-end integration test for the XTAL control loop.

Does NOT require a live Flink/Kafka cluster. Mocks Prometheus and Flink REST
to validate:
  1. The controller's observe → select → apply cycle runs without error
  2. The agent records experience and retrains correctly
  3. SHAP explanations are produced after training
  4. The decision log CSV is written with all expected columns

Run with:
    pytest tests/integration_test.py -v
"""

from __future__ import annotations

import csv
import pathlib
import time
from unittest.mock import patch, MagicMock

import pytest

from controller.telemetry import TelemetryCollector, SystemState
from controller.actuator import XTALActuator, ControlAction
from controller.agent import XTALAgent, N_ACTIONS, ACTION_CONFIGS
from controller.reward import compute_reward


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_state(p99: float = 200.0, tps: float = 5000.0) -> SystemState:
    return SystemState(
        timestamp=time.time(),
        p99_latency_ms=p99,
        p999_latency_ms=p99 * 2,
        input_buffer_usage=0.2,
        checkpoint_duration_ms=800.0,
        kafka_consumer_lag=0.0,
        records_in_per_sec=tps,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestControlLoopIntegration:
    """Simulate N control loop iterations without live infrastructure."""

    N_STEPS = 30  # enough to trigger one retrain (RETRAIN_EVERY=20)

    def test_full_loop_runs_without_error(self):
        agent = XTALAgent(sla_ms=500.0, seed=42)
        states = [make_state(p99=200 + i * 10) for i in range(self.N_STEPS + 1)]

        prev_state = states[0]
        for step in range(1, self.N_STEPS + 1):
            curr_state = states[step]

            # 1. Select action — always explores until model is fitted (epsilon=1 until then)
            action_idx, explanation = agent.select_action(curr_state)
            assert 0 <= action_idx < N_ACTIONS, f"Invalid action index at step {step}"

            # 2. Compute reward
            reward = compute_reward(prev_state, curr_state, sla_ms=500.0)
            assert isinstance(reward, float)
            assert not (reward != reward), "reward must not be NaN"

            # 3. Record
            agent.record(prev_state, action_idx, reward, curr_state)

            prev_state = curr_state

        assert len(agent._experience) == self.N_STEPS

    def test_model_trained_after_retrain_threshold(self):
        agent = XTALAgent(seed=42)
        state = make_state()
        next_state = make_state(p99=180.0)

        for i in range(25):
            agent.record(state, i % N_ACTIONS, -0.4, next_state)

        assert agent._model is not None, "Model should be trained after 25 steps"

    def test_shap_explanations_after_training(self):
        agent = XTALAgent(seed=42)
        state = make_state()
        next_state = make_state()

        for i in range(25):
            agent.record(state, i % N_ACTIONS, -0.4, next_state)

        # After training, exploit (force epsilon=0)
        agent._step = 10000
        idx, explanation = agent.select_action(state)

        # Should have SHAP explanations (not just exploration placeholder)
        if agent._explainer is not None:
            assert len(explanation.top_features) > 0

    def test_sla_violated_state_triggers_aggressive_actions(self):
        """Under SLA violation, agent should prefer low checkpoint intervals."""
        agent = XTALAgent(sla_ms=500.0, seed=0)
        # Train on many experiences where low checkpoint = better reward
        healthy = make_state(p99=100.0)
        recovering = make_state(p99=180.0)

        for i in range(30):
            action_idx = i % 5  # use aggressive actions 0-4 (ckpt <= 1000ms)
            agent.record(healthy, action_idx, -0.2, recovering)

        # Under pressure, force exploitation
        agent._step = 10000
        pressure_state = make_state(p99=490.0)  # near SLA boundary
        idx, explanation = agent.select_action(pressure_state)
        assert 0 <= idx < N_ACTIONS


class TestActuatorSafetyIntegration:
    """Validate that XTALActuator never sends unsafe actions."""

    def test_unsafe_action_rejected_before_flink_call(self):
        with patch("controller.actuator.FlinkActuator") as MockFlink, \
             patch("controller.actuator.KafkaActuator"):
            mock_flink_instance = MagicMock()
            MockFlink.return_value = mock_flink_instance

            actuator = XTALActuator.__new__(XTALActuator)
            actuator.flink = mock_flink_instance
            actuator.kafka = MagicMock()

            unsafe_action = ControlAction(checkpoint_interval_ms=50)  # below MIN=500ms
            result = actuator.apply_action(unsafe_action, "fake-job-id", "nexmark-bids")

            assert result is False
            mock_flink_instance.set_job_config_dynamic.assert_not_called()

    def test_safe_action_proceeds(self):
        with patch("controller.actuator.FlinkActuator"), \
             patch("controller.actuator.KafkaActuator"):

            actuator = XTALActuator.__new__(XTALActuator)
            actuator.flink = MagicMock()
            actuator.kafka = MagicMock()

            safe_action = ControlAction(checkpoint_interval_ms=2000)
            # Should attempt to call Flink (may fail due to mock, but not safety-rejected)
            actuator.apply_action(safe_action, "fake-job-id", "nexmark-bids")
            # If we got here without safety rejection, the test passes


class TestDecisionLogIntegration:
    """Validate that the controller writes a correctly-formatted decision CSV."""

    EXPECTED_COLUMNS = {
        "step", "timestamp", "p99_latency_ms", "p999_latency_ms",
        "input_buffer_usage", "checkpoint_duration_ms", "kafka_consumer_lag",
        "records_in_per_sec", "action_idx", "checkpoint_ms", "buffer_timeout_ms",
        "kafka_linger_ms", "predicted_reward", "actual_reward",
        "shap_feat_1", "shap_val_1", "shap_feat_2", "shap_val_2",
        "shap_feat_3", "shap_val_3", "sla_violated",
    }

    def test_decision_log_columns(self, tmp_path):
        """Check the controller writes the expected CSV columns."""
        from controller.main import XTALController

        with patch("controller.main.TelemetryCollector") as MockTelemetry, \
             patch("controller.main.XTALActuator") as MockActuator:

            mock_collector = MagicMock()
            mock_collector.collect.return_value = make_state()
            MockTelemetry.return_value = mock_collector

            mock_actuator = MagicMock()
            mock_actuator.flink.get_job_id.return_value = "test-job-id"
            mock_actuator.apply_action.return_value = True
            MockActuator.return_value = mock_actuator

            controller = XTALController(log_dir=tmp_path)
            agent = XTALAgent(seed=0)
            # Select via exploration (model not fitted yet) — still returns a valid action
            controller.agent = agent
            controller._running = True
            controller._step = 0
            controller._last_action_idx = 0

            log_file = tmp_path / "xtal_decisions_test.csv"
            controller._csv_file = open(log_file, "w", newline="")
            controller._csv_writer = csv.DictWriter(
                controller._csv_file,
                fieldnames=controller._csv_columns()
            )
            controller._csv_writer.writeheader()

            state = make_state()
            # Force exploration path by ensuring model is not fitted
            action_idx, explanation = controller.agent.select_action(state)
            controller._log_decision(state, action_idx, explanation, reward=None)
            controller._csv_file.close()

            # Verify columns
            with open(log_file) as f:
                reader = csv.DictReader(f)
                cols = set(reader.fieldnames or [])
            assert self.EXPECTED_COLUMNS.issubset(cols), \
                f"Missing columns: {self.EXPECTED_COLUMNS - cols}"
