"""
Input Perturbation uncertainty estimation for neural distance prediction.

Perturbs obstacle point coordinates with Gaussian noise and runs DUNE
multiple times to estimate prediction variance. Uses 2-sigma of the
variance as a per-point safety margin.

This is a standard sensitivity analysis technique that quantifies how
sensitive DUNE's distance predictions are to small perturbations in
input point coordinates.

Reference:
  - "Simple and Scalable Predictive Uncertainty Estimation using Deep
    Ensembles", Lakshminarayanan et al., NeurIPS 2017 (conceptually
    related, though we perturb inputs rather than using ensembles).
"""

import numpy as np
import torch
from typing import Optional


class InputPerturbationWrapper:
    """Uncertainty estimation by perturbing input point coordinates.

    Adds Gaussian noise to obstacle point coordinates and runs DUNE
    N times. The standard deviation of distance predictions across
    perturbed forward passes is used as the uncertainty estimate.
    The safety margin is set to safety_factor * max(std).
    """

    def __init__(
        self,
        n_samples: int = 10,
        noise_std: float = 0.01,   # 1 cm perturbation (meters)
        safety_factor: float = 2.0,  # 2-sigma for ~95% confidence
        margin_cap: float = 0.10,    # cap margin at 10 cm
        margin_floor: float = 0.0,   # floor margin at 0
    ):
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.safety_factor = safety_factor
        self.margin_cap = margin_cap
        self.margin_floor = margin_floor

    def estimate(self, point_flow, R_list, obs_points_list, dune_layer) -> float:
        """Run MC perturbed forward passes, return max per-point uncertainty margin.

        Args:
            point_flow: list of (2, N) tensors in robot frame
            R_list: rotation matrices
            obs_points_list: obstacle points in global frame
            dune_layer: DUNE module with forward() method

        Returns:
            margin: scalar safety margin (meters), capped at margin_cap
        """
        all_distances = []

        for _ in range(self.n_samples):
            noisy_point_flow = []
            for pf in point_flow:
                noise = torch.randn_like(pf) * self.noise_std
                noisy_point_flow.append(pf + noise)

            with torch.no_grad():
                dune_layer(noisy_point_flow, R_list, obs_points_list)

            if hasattr(dune_layer, 'distances_0'):
                dist = dune_layer.distances_0.cpu().numpy()
                all_distances.append(dist)

        if not all_distances:
            return self.margin_floor

        stacked = np.stack(all_distances, axis=0)  # (n_samples, M')
        std = stacked.std(axis=0)                   # per-point std
        margin = float(np.max(self.safety_factor * std))
        return float(np.clip(margin, self.margin_floor, self.margin_cap))
