"""
Control Barrier Function (CBF) safety filter for differential-drive robots.

Implements a distance-based CBF that minimally modifies the NRMP action
to guarantee forward invariance of the safe set {x : d(x) >= d_safe}.

Theory (Ames et al., ECC 2019):
  Given a barrier function h(x) = d(x) - d_safe where d(x) is the
  minimum predicted distance to obstacles, the CBF condition is:

      sup_u [ ḣ(x, u) + α h(x) ] >= 0

  For a unicycle model with state [x, y, θ] and control [v, ω]:
      ẋ = v cos(θ),  ẏ = v sin(θ)

  For an obstacle at angle φ in the robot frame:
      d/dt d(x) = -v cos(φ)

  So the CBF constraint becomes:
      -v cos(φ) + α (d - d_safe) >= 0
  =>  v cos(φ) <= α (d - d_safe)

  This is enforced via a QP that minimally adjusts (v, ω):
      min  (v - v_nom)² + η (ω - ω_nom)²
      s.t. v cos(φ_i) <= α (d_i - d_safe)  ∀i with cos(φ_i) > 0
           0 <= v <= v_max
           -ω_max <= ω <= ω_max

Reference:
  Ames et al., "Control Barrier Functions: Theory and Applications", ECC 2019.
"""

import numpy as np
from typing import Optional


class CBFSafetyFilter:
    """Distance-based CBF safety filter for post-hoc action correction.

    After NRMP produces an action (v_nom, ω_nom), the CBF filter solves
    a minimal-intervention QP to ensure safety while staying as close
    as possible to the nominal action.
    """

    def __init__(
        self,
        d_safe: float = 0.15,       # minimum safe distance (m)
        alpha: float = 2.0,          # CBF class-Kappa rate (larger = more permissive)
        eta: float = 0.3,            # weight on ω deviation in QP
        v_max: float = 1.0,          # max linear velocity (m/s)
        omega_max: float = 3.14,     # max angular velocity (rad/s)
        d_danger: float = 0.05,      # distance below which emergency stop is triggered
    ):
        self.d_safe = d_safe
        self.alpha = alpha
        self.eta = eta
        self.v_max = v_max
        self.omega_max = omega_max
        self.d_danger = d_danger

        self._intervention_count = 0
        self._total_count = 0
        self._prev_v = 0.0

    def filter_action(
        self,
        action_nom: np.ndarray,          # [v_nom, ω_nom]
        point_distances: np.ndarray,     # (M',) predicted distance per obstacle point
        point_angles: np.ndarray = None, # (M',) angle φ_i in robot frame (rad), default=0
        dt: float = 0.1,
    ) -> np.ndarray:
        """Filter nominal action through CBF-QP.

        Args:
            action_nom: [v, ω] from NRMP
            point_distances: predicted distances d_i to each obstacle point
            point_angles: angles φ_i of each point in robot frame
            dt: time step (for rate limiting, not used in CBF constraint)

        Returns:
            safe_action: filtered [v, ω]
        """
        self._total_count += 1

        v_nom = float(action_nom[0])
        omega_nom = float(action_nom[1])

        # Ensure non-negative velocity
        v_nom = max(v_nom, 0.0)

        if point_angles is None:
            point_angles = np.zeros(len(point_distances))

        if len(point_distances) == 0:
            self._prev_v = v_nom
            return action_nom

        d_min = float(np.min(point_distances))

        # Emergency stop: obstacle too close
        if d_min <= self.d_danger:
            self._intervention_count += 1
            self._prev_v = 0.0
            return np.array([0.0, omega_nom])

        # Find the most restrictive CBF constraint among points in front
        # Only points with cos(φ) > 0 (in front of robot) constrain forward speed
        v_safe = self.v_max

        for i in range(len(point_distances)):
            d_i = float(point_distances[i])
            phi_i = float(point_angles[i])
            cos_phi = np.cos(phi_i)

            if cos_phi <= 0:
                continue  # obstacle behind or to the side, can't hit it by moving forward

            # h(x) = d_i - d_safe, should be >= 0
            h = d_i - self.d_safe

            if h <= 0:
                # Already in unsafe region, enforce stricter constraint
                v_safe = 0.0
                break

            # CBF: v * cos(phi) <= alpha * h
            # =>  v <= alpha * h / cos(phi)
            v_bound = self.alpha * h / cos_phi
            v_safe = min(v_safe, v_bound)

        v_safe = max(v_safe, 0.0)

        if v_nom > v_safe + 1e-6:
            self._intervention_count += 1

        # QP solution: v is clamped by CBF, ω is unchanged
        # (ω doesn't affect distance-to-obstacle under the unicycle model
        #  since d/dt d(x) = -v cos(φ) is independent of ω)
        v_out = min(v_nom, v_safe)
        omega_out = omega_nom

        # Rate-limit deceleration for physical plausibility
        max_dv = 2.0 * dt  # max 2 m/s² deceleration
        if self._prev_v - v_out > max_dv:
            v_out = self._prev_v - max_dv

        self._prev_v = v_out
        return np.array([v_out, omega_out])

    @property
    def intervention_rate(self) -> float:
        if self._total_count == 0:
            return 0.0
        return self._intervention_count / self._total_count

    def reset_stats(self):
        self._intervention_count = 0
        self._total_count = 0
        self._prev_v = 0.0
