"""
Risk Budget Allocator: distributes global q_hat to per-point margins.

Strategies:
  - uniform: equal split across (horizon * n_points)
  - distance_weighted: nearer points get larger margin
  - global: scalar q_hat applied uniformly (baseline, equivalent to current behavior)
"""

import numpy as np
import torch
from typing import Optional


class RiskBudgetAllocator:
    """Allocate global risk budget q_hat to per-point safety margins."""

    def __init__(self, strategy: str = "global"):
        """
        Args:
            strategy: one of "global", "uniform", "distance_weighted"
        """
        assert strategy in ("global", "uniform", "distance_weighted", "cpsa")
        self.strategy = strategy

    def allocate(
        self,
        q_hat: float,
        d_pred: Optional[np.ndarray] = None,
        horizon: int = 10,
        n_points: int = 10,
        delta: float = 0.01,
    ) -> np.ndarray:
        """Compute per-point margin array.

        Args:
            q_hat: global conformal risk quantile
            d_pred: predicted distances, shape (n_points,). Required for
                    distance_weighted strategy.
            horizon: planning horizon T
            n_points: number of obstacle points M' in NRMP
            delta: small constant to avoid division by zero

        Returns:
            margins: shape (T, M') for per-point, or (1, 1) for global
        """
        if self.strategy == "global":
            return np.array([[q_hat]])

        total = horizon * n_points

        if self.strategy == "uniform":
            per_point = q_hat / total
            return np.full((horizon, n_points), per_point)

        # distance_weighted: nearer obstacles → larger margin
        if d_pred is None:
            raise ValueError("d_pred required for distance_weighted strategy")

        if d_pred.ndim == 1:
            d_pred = d_pred.reshape(-1)

        n = min(len(d_pred), n_points)
        d = np.abs(d_pred[:n]) + delta  # avoid div-by-zero
        w = 1.0 / d
        w = w / w.sum()

        # Total budget = q_hat, allocate across (T * n_points) but apply
        # uniformly across timesteps. Each point gets q_hat * w[i] / horizon
        # so the sum over (T, M') equals q_hat.
        margins = np.zeros((horizon, n_points))
        for i in range(n):
            margins[:, i] = q_hat * w[i] / horizon
        return margins

    @staticmethod
    def margins_to_flat(margins: np.ndarray) -> np.ndarray:
        """Flatten (T, M') margin matrix to 1D for NRMP parameter."""
        return margins.flatten()

    @staticmethod
    def flat_to_margins(flat: np.ndarray, horizon: int, n_points: int) -> np.ndarray:
        """Reshape flat margin vector to (T, M')."""
        return flat.reshape(horizon, n_points)
