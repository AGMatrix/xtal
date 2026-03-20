"""Unit tests for controller/telemetry.py."""

import time
import pytest
from unittest.mock import patch, MagicMock
from controller.telemetry import TelemetryCollector, SystemState


class TestSystemState:
    def test_as_feature_vector_length(self):
        state = SystemState(
            timestamp=time.time(),
            p99_latency_ms=200.0,
            p999_latency_ms=400.0,
            input_buffer_usage=0.3,
            checkpoint_duration_ms=900.0,
            kafka_consumer_lag=100.0,
            records_in_per_sec=5000.0,
        )
        vec = state.as_feature_vector()
        assert len(vec) == 6

    def test_as_feature_vector_excludes_timestamp(self):
        state = SystemState(
            timestamp=12345.0,
            p99_latency_ms=200.0,
            p999_latency_ms=400.0,
            input_buffer_usage=0.3,
            checkpoint_duration_ms=900.0,
            kafka_consumer_lag=100.0,
            records_in_per_sec=5000.0,
        )
        vec = state.as_feature_vector()
        assert 12345.0 not in vec

    def test_to_dict_contains_all_fields(self):
        state = SystemState(
            timestamp=time.time(),
            p99_latency_ms=200.0,
            p999_latency_ms=400.0,
            input_buffer_usage=0.3,
            checkpoint_duration_ms=900.0,
            kafka_consumer_lag=100.0,
            records_in_per_sec=5000.0,
        )
        d = state.to_dict()
        assert "p99_latency_ms" in d
        assert "timestamp" in d


class TestTelemetryCollector:
    def _make_mock_response(self, value: float) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "data": {"result": [{"value": [0, str(value)]}]}
        }
        return mock_resp

    def test_collect_returns_system_state(self):
        collector = TelemetryCollector()
        with patch.object(collector._session, "get") as mock_get:
            mock_get.return_value = self._make_mock_response(123.4)
            state = collector.collect()
        assert isinstance(state, SystemState)
        assert state.p99_latency_ms == pytest.approx(123.4)

    def test_collect_defaults_to_zero_on_empty_result(self):
        collector = TelemetryCollector()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"data": {"result": []}}
        with patch.object(collector._session, "get", return_value=mock_resp):
            state = collector.collect()
        assert state.p99_latency_ms == 0.0

    def test_collect_defaults_to_zero_on_exception(self):
        collector = TelemetryCollector()
        with patch.object(collector._session, "get", side_effect=Exception("timeout")):
            state = collector.collect()
        assert state.p99_latency_ms == 0.0

    def test_tail_latency_ok_within_sla(self):
        collector = TelemetryCollector()
        state = SystemState(
            timestamp=time.time(), p99_latency_ms=400.0, p999_latency_ms=800.0,
            input_buffer_usage=0.1, checkpoint_duration_ms=500.0,
            kafka_consumer_lag=0.0, records_in_per_sec=5000.0,
        )
        assert collector.tail_latency_ok(state, sla_ms=500.0) is True

    def test_tail_latency_ok_violates_sla(self):
        collector = TelemetryCollector()
        state = SystemState(
            timestamp=time.time(), p99_latency_ms=600.0, p999_latency_ms=1200.0,
            input_buffer_usage=0.8, checkpoint_duration_ms=2000.0,
            kafka_consumer_lag=5000.0, records_in_per_sec=5000.0,
        )
        assert collector.tail_latency_ok(state, sla_ms=500.0) is False
