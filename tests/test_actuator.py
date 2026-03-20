"""Unit tests for controller/actuator.py — safety bounds and action validation."""

import pytest
from controller.actuator import ControlAction


class TestControlActionSafety:
    def test_valid_action_is_safe(self):
        a = ControlAction(checkpoint_interval_ms=2000, buffer_timeout_ms=10, kafka_linger_ms=5)
        assert a.is_safe()

    def test_checkpoint_too_low_is_unsafe(self):
        a = ControlAction(checkpoint_interval_ms=100)
        assert not a.is_safe()

    def test_checkpoint_too_high_is_unsafe(self):
        a = ControlAction(checkpoint_interval_ms=60_001)
        assert not a.is_safe()

    def test_checkpoint_at_min_boundary_is_safe(self):
        a = ControlAction(checkpoint_interval_ms=ControlAction.MIN_CHECKPOINT_MS)
        assert a.is_safe()

    def test_checkpoint_at_max_boundary_is_safe(self):
        a = ControlAction(checkpoint_interval_ms=ControlAction.MAX_CHECKPOINT_MS)
        assert a.is_safe()

    def test_parallelism_zero_is_unsafe(self):
        a = ControlAction(parallelism=0)
        assert not a.is_safe()

    def test_parallelism_too_high_is_unsafe(self):
        a = ControlAction(parallelism=ControlAction.MAX_PARALLELISM + 1)
        assert not a.is_safe()

    def test_parallelism_at_min_is_safe(self):
        a = ControlAction(parallelism=ControlAction.MIN_PARALLELISM)
        assert a.is_safe()

    def test_buffer_timeout_negative_is_unsafe(self):
        a = ControlAction(buffer_timeout_ms=-1)
        assert not a.is_safe()

    def test_buffer_timeout_too_high_is_unsafe(self):
        a = ControlAction(buffer_timeout_ms=ControlAction.MAX_BUFFER_MS + 1)
        assert not a.is_safe()

    def test_none_fields_are_ignored_in_safety_check(self):
        # All None = no-op action, must be safe
        a = ControlAction()
        assert a.is_safe()

    def test_mixed_valid_invalid_is_unsafe(self):
        a = ControlAction(checkpoint_interval_ms=2000, buffer_timeout_ms=-5)
        assert not a.is_safe()

    def test_all_valid_fields_is_safe(self):
        a = ControlAction(
            checkpoint_interval_ms=1000,
            parallelism=4,
            buffer_timeout_ms=50,
            kafka_linger_ms=10,
        )
        assert a.is_safe()
