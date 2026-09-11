"""
Risk Calibrator: Asymmetric conformal risk calibration for NeuPAN.

Collects (d_pred, d_gt) pairs from DUNE, computes asymmetric risk scores
r(p) = max(0, d_pred - d_gt), and derives conformal quantile q_hat(epsilon)
providing Pr[R(scene) <= q_hat] >= 1 - epsilon.
"""

import json
import numpy as np
import os
from typing import Optional


class RiskCalibrator:
    """Conformal risk calibration with asymmetric nonconformity score."""

    def __init__(self):
        self.calibration_scores = None  # per-scene risk scores
        self.q_hat_cache = {}           # epsilon -> q_hat

    @staticmethod
    def asymmetric_risk(d_pred: np.ndarray, d_gt: np.ndarray) -> np.ndarray:
        """r(p) = max(0, d_pred - d_gt). Only overestimation is dangerous."""
        return np.maximum(0.0, d_pred - d_gt)

    @staticmethod
    def symmetric_risk(d_pred: np.ndarray, d_gt: np.ndarray) -> np.ndarray:
        """Baseline: |d_pred - d_gt|."""
        return np.abs(d_pred - d_gt)

    def collect_scene_risk(self, d_pred: np.ndarray, d_gt: np.ndarray,
                           score_type: str = "asymmetric") -> float:
        """Compute scene-level risk R(s) = max_i r(p_i)."""
        if score_type == "asymmetric":
            risks = self.asymmetric_risk(d_pred, d_gt)
        elif score_type == "symmetric":
            risks = self.symmetric_risk(d_pred, d_gt)
        else:
            raise ValueError(f"Unknown score_type: {score_type}")

        return float(np.max(risks))

    def fit(self, scene_risks: np.ndarray):
        """Store calibration set of per-scene risk scores."""
        self.calibration_scores = np.sort(scene_risks)
        self.q_hat_cache = {}

    def compute_q_hat(self, epsilon: float) -> float:
        """Compute conformal quantile q_hat(epsilon) from calibration set.

        q_hat = Quantile(sorted_scores, ceil((1-epsilon)(N+1)/N))
        """
        if self.calibration_scores is None:
            raise RuntimeError("Call fit() first with calibration data.")

        N = len(self.calibration_scores)
        if epsilon in self.q_hat_cache:
            return self.q_hat_cache[epsilon]

        level = min(np.ceil((1 - epsilon) * (N + 1)) / N, 1.0)
        q_hat = float(np.quantile(self.calibration_scores, level))
        self.q_hat_cache[epsilon] = q_hat
        return q_hat

    @staticmethod
    def compute_underestimation_q_hat(
        d_pred: np.ndarray, d_gt: np.ndarray, epsilon: float
    ) -> float:
        """q_hat for the underestimation direction: max(0, d_gt - d_pred).

        Used to adjust collision_threshold when lidar noise causes DUNE to
        predict shorter distances than ground truth.

        Returns q_hat such that Pr[d_gt - d_pred <= q_hat] >= 1-epsilon.
        """
        underest = np.maximum(0.0, d_gt - d_pred)
        # Scene-level: take max across all points in each scene
        # If inputs are 1D (already scene-level), use as-is
        if underest.ndim == 1:
            scene_scores = np.sort(underest)
        else:
            scene_scores = np.sort(np.max(underest, axis=-1))
        N = len(scene_scores)
        level = min(np.ceil((1 - epsilon) * (N + 1)) / N, 1.0)
        return float(np.quantile(scene_scores, level))

    def compute_q_hat_symmetric(self, d_pred_list, d_gt_list, epsilon: float) -> float:
        """Compute q_hat using symmetric score for comparison (ablation)."""
        risks = []
        for d_pred, d_gt in zip(d_pred_list, d_gt_list):
            risks.append(np.max(self.symmetric_risk(d_pred, d_gt)))
        sorted_risks = np.sort(np.array(risks))
        N = len(sorted_risks)
        level = min(np.ceil((1 - epsilon) * (N + 1)) / N, 1.0)
        return float(np.quantile(sorted_risks, level))

    @staticmethod
    def compute_weighted_q_hat(
        scene_risks: np.ndarray,
        d_pred_list: list,
        epsilon: float,
        d_max: float = 5.0,
    ) -> float:
        """Compute q_hat with distance-based weighting (Chee et al. CDC 2023).

        Weights: w(p) = max(0, 1 - d_pred(p) / d_max)
        Closer obstacles get higher weight in the empirical CDF.

        Args:
            scene_risks: (N,) scene-level risk scores
            d_pred_list: list of per-scene predicted distances arrays
            epsilon: target error rate
            d_max: maximum distance for weighting (default 5m)
        Returns:
            weighted_q_hat
        """
        N = len(scene_risks)
        if N == 0:
            return 0.0

        # Compute per-scene weight as mean weight of points in scene
        weights = np.ones(N)
        for i, d_pred in enumerate(d_pred_list):
            if len(d_pred) > 0:
                w = np.maximum(0.0, 1.0 - np.abs(d_pred) / d_max)
                weights[i] = float(np.mean(w))

        weights = np.clip(weights, 0.01, 1.0)
        weights = weights / weights.sum()

        # Weighted empirical CDF → weighted quantile
        sorted_idx = np.argsort(scene_risks)
        sorted_risks = scene_risks[sorted_idx]
        sorted_weights = weights[sorted_idx]
        cum_weights = np.cumsum(sorted_weights)

        target = 1.0 - epsilon
        idx = np.searchsorted(cum_weights, target)
        idx = min(idx, N - 1)
        return float(sorted_risks[idx])

    def save(self, path: str):
        """Save calibration data to JSON."""
        data = {
            "calibration_scores": self.calibration_scores.tolist() if self.calibration_scores is not None else [],
            "q_hat_cache": {str(k): v for k, v in self.q_hat_cache.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Load calibration data from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        self.calibration_scores = np.array(data["calibration_scores"])
        self.q_hat_cache = {float(k): v for k, v in data["q_hat_cache"].items()}
