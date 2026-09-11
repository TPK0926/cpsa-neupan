"""
Adaptive Conformal Inference (ACI) for online distribution shift.

When calibration and deployment distributions diverge, the exchangeability
assumption of standard CP is violated. ACI dynamically adjusts the error
rate alpha_t based on observed miscoverage events, maintaining approximate
validity under gradual distribution shift.

Algorithm (Gibbs & Candes, NeurIPS 2021):
  Given target error rate ε and step size γ:
    alpha_0 = ε
    For each timestep t:
      Predict set C_t at level 1 - alpha_t
      Observe err_t = 1[y_t not in C_t]
      alpha_{t+1} = alpha_t + γ * (ε - err_t)
      Clip alpha to [alpha_min, alpha_max]

For navigation: we define a "coverage violation" as a scene where the
true minimum distance d_gt is less than the predicted minimum distance
minus the safety margin (d_pred - q_hat), for any obstacle point.

Reference:
  Gibbs & Candes, "Adaptive Conformal Inference Under Distribution
  Shift", NeurIPS 2021.
"""

import numpy as np
from typing import Optional


class AdaptiveConformalInference:
    """Online adaptive conformal inference with time-varying error rate.

    Maintains a running alpha_t that increases when coverage violations
    are observed and decreases when coverage is satisfied, tracking
    gradual distribution shift.
    """

    def __init__(
        self,
        epsilon: float = 0.05,         # target error rate
        gamma: float = 0.005,          # step size (smaller = slower adaptation)
        q_hat_initial: float = 0.0121, # initial quantile from offline calibration
        alpha_min: float = 0.01,
        alpha_max: float = 0.20,
    ):
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_hat = q_hat_initial
        self.alpha = epsilon
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        self._calibration_scores: Optional[np.ndarray] = None
        self._calibration_window: list = []

        self.err_history: list[float] = []
        self.alpha_history: list[float] = []
        self.q_hat_history: list[float] = []

    def set_calibration(self, scores: np.ndarray):
        """Set offline calibration scores (sorted scene-level risks)."""
        self._calibration_scores = np.sort(scores)
        self._calibration_window = list(scores)

    def update(self, violation: bool):
        """Update alpha_t based on observed coverage violation.

        Args:
            violation: True if d_gt < d_pred - q_hat for any point in the
                       current scene (i.e., the conformal margin was violated).
        """
        err_t = 1.0 if violation else 0.0
        self.err_history.append(err_t)

        self.alpha = self.alpha + self.gamma * (self.epsilon - err_t)
        self.alpha = np.clip(self.alpha, self.alpha_min, self.alpha_max)
        self.alpha_history.append(self.alpha)

        self._recompute_q_hat()

    def update_with_score(self, new_score: float):
        """Sliding-window variant: maintain a window of recent scores.

        Args:
            new_score: scene-level risk R(s) for the current scene.
        """
        self._calibration_window.append(new_score)
        max_window = 500
        if len(self._calibration_window) > max_window:
            self._calibration_window = self._calibration_window[-max_window:]
        self._recompute_q_hat()

    def _recompute_q_hat(self):
        """Recompute conformal quantile from current calibration window."""
        if len(self._calibration_window) == 0:
            return
        N = len(self._calibration_window)
        scores = np.sort(self._calibration_window)
        idx = int(np.ceil((1 - self.alpha) * (N + 1))) - 1
        idx = np.clip(idx, 0, N - 1)
        self.q_hat = float(scores[idx])
        self.q_hat_history.append(self.q_hat)

    def get_q_hat(self) -> float:
        return self.q_hat

    def get_alpha(self) -> float:
        return self.alpha

    @property
    def effective_coverage(self) -> float:
        if len(self.err_history) == 0:
            return 1.0
        return 1.0 - np.mean(self.err_history)
