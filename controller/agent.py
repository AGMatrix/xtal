"""
agent.py — XTAL Control Agent: XGBoost Contextual Bandit + SHAP Explanations.

Architecture:
  - Action space: 14 discrete preset configurations of (checkpoint_ms, buffer_timeout_ms, kafka_linger_ms)
  - One XGBoost regressor that takes (state_features + action_features) → predicted reward
  - Epsilon-greedy exploration (epsilon decays from 0.3 → 0.05 over 500 steps)
  - Online learning: model retrained every RETRAIN_EVERY steps on collected experience
  - Explainability: SHAP TreeExplainer run on every selected action → top-3 feature importances

This design is intentional for the paper:
  - XGBoost is natively interpretable via SHAP (no post-hoc approximation)
  - Online retraining allows adaptation to changing load patterns
  - Discrete action space makes the safety layer trivial
"""

from __future__ import annotations

import logging
import pickle
import pathlib
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
import shap

from .telemetry import SystemState

log = logging.getLogger(__name__)

# ── Action Space ──────────────────────────────────────────────────────────────
# Each tuple: (checkpoint_interval_ms, buffer_timeout_ms, kafka_linger_ms)
ACTION_CONFIGS: list[tuple[int, int, int]] = [
    (500,    0,   0),   # 0: most aggressive — low ckpt, no buffering
    (500,    5,   5),   # 1
    (1000,   0,   0),   # 2
    (1000,   5,   5),   # 3
    (1000,  10,  10),   # 4
    (2000,   0,   0),   # 5
    (2000,  10,   5),   # 6
    (2000,  50,  25),   # 7
    (5000,  10,  10),   # 8: Flink default checkpoint interval
    (5000,  50,  25),   # 9
    (5000, 100,  50),   # 10
    (10000, 10,  10),   # 11
    (10000, 50,  25),   # 12
    (10000,100,  50),   # 13: most conservative
]

N_ACTIONS = len(ACTION_CONFIGS)

STATE_FEATURES = [
    "p99_latency_ms",
    "p999_latency_ms",
    "input_buffer_usage",
    "checkpoint_duration_ms",
    "kafka_consumer_lag",
    "records_in_per_sec",
]
ACTION_FEATURES = ["checkpoint_ms", "buffer_timeout_ms", "kafka_linger_ms"]
ALL_FEATURES = STATE_FEATURES + ACTION_FEATURES

RETRAIN_EVERY = 20          # retrain XGBoost every N steps
MIN_SAMPLES_TO_TRAIN = 14   # need at least one sample per action
EPSILON_START = 0.30
EPSILON_END   = 0.05
EPSILON_DECAY = 500         # steps to decay from start → end


@dataclass
class Experience:
    state: list[float]          # STATE_FEATURES values
    action_idx: int
    action: tuple[int, int, int]
    reward: float
    next_state: list[float]
    timestamp: float = field(default_factory=time.time)


@dataclass
class ActionExplanation:
    action_idx: int
    action: tuple[int, int, int]
    predicted_reward: float
    top_features: list[tuple[str, float]]   # [(feature_name, shap_value), ...]

    def human_readable(self) -> str:
        lines = [
            f"Action #{self.action_idx}: checkpoint={self.action[0]}ms, "
            f"buffer_timeout={self.action[1]}ms, kafka_linger={self.action[2]}ms",
            f"  Predicted reward: {self.predicted_reward:.3f}",
            "  Key drivers:",
        ]
        for feat, val in self.top_features:
            direction = "↑ increases" if val > 0 else "↓ decreases"
            lines.append(f"    {feat}: {direction} reward by {abs(val):.3f}")
        return "\n".join(lines)


class XTALAgent:
    """
    XGBoost Contextual Bandit for tail-latency control.

    Usage:
        agent = XTALAgent()
        action_idx, explanation = agent.select_action(state)
        # ... apply action, observe new state ...
        reward = compute_reward(state, new_state)
        agent.record(state, action_idx, reward, new_state)
    """

    def __init__(
        self,
        sla_ms: float = 500.0,
        model_path: Optional[str] = None,
        seed: int = 42,
    ):
        self.sla_ms = sla_ms
        self.seed = seed
        self._step = 0
        self._experience: list[Experience] = []
        self._model: Optional[xgb.XGBRegressor] = None
        self._model_fitted: bool = False
        self._explainer: Optional[shap.TreeExplainer] = None
        self._rng = np.random.default_rng(seed)

        if model_path and pathlib.Path(model_path).exists():
            self.load(model_path)
            log.info("Loaded agent from %s", model_path)
        else:
            self._init_model()

    # ── Model ─────────────────────────────────────────────────────────────────

    def _init_model(self) -> None:
        self._model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.seed,
            n_jobs=-1,
        )

    def _build_row(self, state: SystemState, action_idx: int) -> list[float]:
        ckpt, buf, linger = ACTION_CONFIGS[action_idx]
        return state.as_feature_vector() + [ckpt, buf, linger]

    def _train(self) -> None:
        if len(self._experience) < MIN_SAMPLES_TO_TRAIN:
            return
        rows = [self._build_row_from_list(e.state, e.action_idx) for e in self._experience]
        X = pd.DataFrame(rows, columns=ALL_FEATURES)
        y = np.array([e.reward for e in self._experience])
        self._model.fit(X, y)
        self._model_fitted = True
        self._explainer = shap.TreeExplainer(self._model)
        log.info("Model retrained on %d samples", len(self._experience))

    def _build_row_from_list(self, state_vec: list[float], action_idx: int) -> list[float]:
        ckpt, buf, linger = ACTION_CONFIGS[action_idx]
        return state_vec + [ckpt, buf, linger]

    # ── Action selection ───────────────────────────────────────────────────────

    @property
    def _epsilon(self) -> float:
        decay = min(self._step / EPSILON_DECAY, 1.0)
        return EPSILON_START + (EPSILON_END - EPSILON_START) * decay

    def select_action(
        self, state: SystemState
    ) -> tuple[int, ActionExplanation]:
        """
        Selects the best action given the current system state.
        Returns (action_index, explanation).
        """
        self._step += 1

        # Epsilon-greedy exploration
        if self._rng.random() < self._epsilon or not self._model_fitted:
            action_idx = int(self._rng.integers(0, N_ACTIONS))
            predicted_reward = 0.0
            explanation = ActionExplanation(
                action_idx=action_idx,
                action=ACTION_CONFIGS[action_idx],
                predicted_reward=predicted_reward,
                top_features=[("exploration", 0.0)],
            )
            return action_idx, explanation

        # Predict reward for all actions
        state_vec = state.as_feature_vector()
        rows = [self._build_row_from_list(state_vec, i) for i in range(N_ACTIONS)]
        X = pd.DataFrame(rows, columns=ALL_FEATURES)
        predicted_rewards = self._model.predict(X)

        # Safety override: prefer actions that are unlikely to violate SLA
        if state.p99_latency_ms > self.sla_ms * 0.8:
            # Under pressure: bias toward lower checkpoint intervals
            for i, (ckpt, _, _) in enumerate(ACTION_CONFIGS):
                if ckpt > 2000:
                    predicted_rewards[i] -= 5.0

        best_idx = int(np.argmax(predicted_rewards))

        # SHAP explanation for the chosen action
        explanation = self._explain(X.iloc[best_idx], best_idx, predicted_rewards[best_idx])

        return best_idx, explanation

    def _explain(
        self, row: pd.Series, action_idx: int, predicted_reward: float
    ) -> ActionExplanation:
        top_features: list[tuple[str, float]] = []
        if self._explainer is not None:
            try:
                shap_vals = self._explainer.shap_values(row.values.reshape(1, -1))[0]
                pairs = sorted(
                    zip(ALL_FEATURES, shap_vals), key=lambda x: abs(x[1]), reverse=True
                )
                top_features = [(f, float(v)) for f, v in pairs[:3]]
            except Exception as e:
                log.warning("SHAP explanation failed: %s", e)

        return ActionExplanation(
            action_idx=action_idx,
            action=ACTION_CONFIGS[action_idx],
            predicted_reward=float(predicted_reward),
            top_features=top_features,
        )

    # ── Online learning ────────────────────────────────────────────────────────

    def record(
        self,
        state: SystemState,
        action_idx: int,
        reward: float,
        next_state: SystemState,
    ) -> None:
        """Records a transition and periodically retrains the model."""
        exp = Experience(
            state=state.as_feature_vector(),
            action_idx=action_idx,
            action=ACTION_CONFIGS[action_idx],
            reward=reward,
            next_state=next_state.as_feature_vector(),
        )
        self._experience.append(exp)

        if len(self._experience) % RETRAIN_EVERY == 0:
            self._train()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "model_fitted": self._model_fitted,
                "experience": self._experience,
                "step": self._step,
            }, f)
        log.info("Agent saved to %s", path)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._model = data["model"]
        self._model_fitted = data.get("model_fitted", False)
        self._experience = data["experience"]
        self._step = data["step"]
        if self._model is not None and self._model_fitted:
            self._explainer = shap.TreeExplainer(self._model)

    def experience_summary(self) -> dict:
        if not self._experience:
            return {}
        rewards = [e.reward for e in self._experience]
        action_counts = {}
        for e in self._experience:
            action_counts[e.action_idx] = action_counts.get(e.action_idx, 0) + 1
        return {
            "total_steps": self._step,
            "total_experience": len(self._experience),
            "mean_reward": float(np.mean(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "epsilon": round(self._epsilon, 3),
            "action_distribution": action_counts,
        }
