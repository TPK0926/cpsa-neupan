"""
CPSA Risk Network for NeuPAN.

Key components:
  - 17-D per-point features with temporal and passage awareness
  - Global context aggregation (PointNet++ style max-pool)
  - Collision probability loss + narrow passage loss
  - Dynamic q_star and dynamic tau_max
  - Teacher pretraining + SWA support

Reference: CPSA (arXiv 2509.21955)
"""

import torch
import torch.nn as nn
import numpy as np


class CPSAFeatureExtractor:
    """17-D per-point feature extraction for CPSA risk prediction."""

    FEATURE_DIM = 17

    @staticmethod
    def compute_features(
        pts_local: torch.Tensor,
        d_pred: torch.Tensor,
        speed: float = 0.0,
        noise_std: float = 0.0,
        d_pred_prev: torch.Tensor = None,
        radius_prev: torch.Tensor = None,
        lateral_margin: float = 5.0,
        passage_ratio: float = 3.0,
        env_type: float = 0.0,
    ) -> torch.Tensor:
        N = pts_local.shape[0]
        device = pts_local.device
        eps = 1e-6

        radius = torch.sqrt(pts_local[:, 0]**2 + pts_local[:, 1]**2 + eps)
        angle = torch.arctan2(pts_local[:, 1], pts_local[:, 0])
        proximity_ratio = torch.abs(d_pred) / (radius + eps)

        if N > 5:
            diff = pts_local.unsqueeze(1) - pts_local.unsqueeze(0)
            dists = torch.sqrt((diff**2).sum(dim=2) + eps)
            knn_dists, _ = torch.topk(dists, min(6, N), dim=1, largest=False)
            local_density = knn_dists[:, 1:].mean(dim=1)
            in_radius_count = (dists < 1.0).float().sum(dim=1) - 1
            in_radius = torch.clamp(in_radius_count, 0, 50) / 50.0
        elif N > 1:
            diff = pts_local.unsqueeze(1) - pts_local.unsqueeze(0)
            dists = torch.sqrt((diff**2).sum(dim=2) + eps)
            local_density = dists[dists > 0].mean().expand(N)
            in_radius = torch.zeros(N, device=device)
        else:
            local_density = torch.full((N,), 10.0, device=device)
            in_radius = torch.zeros(N, device=device)

        speed_t = torch.full((N,), speed, device=device)
        d_pred_rank = torch.argsort(torch.argsort(torch.abs(d_pred))).float() / max(N - 1, 1)
        noise_std_t = torch.full((N,), noise_std, device=device)

        d_pred_delta = torch.zeros(N, device=device)
        if d_pred_prev is not None and d_pred_prev.shape[0] == N:
            d_pred_delta = d_pred - d_pred_prev

        radius_delta = torch.zeros(N, device=device)
        if radius_prev is not None and radius_prev.shape[0] == N:
            radius_delta = radius - radius_prev

        features = torch.stack([
            d_pred, radius, angle, proximity_ratio, d_pred ** 2,
            torch.cos(angle), torch.sin(angle),
            local_density, in_radius,
            speed_t, d_pred_rank, noise_std_t,
            d_pred_delta, radius_delta,
            torch.full((N,), lateral_margin, device=device),
            torch.full((N,), passage_ratio, device=device),
            torch.full((N,), env_type, device=device),
        ], dim=1)

        return torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)


class CPSARiskNet(nn.Module):
    """Predicts per-point adaptive safety margin (tau) with global context.

    Architecture:
      Input(17) -> Linear(128) -> LN -> ReLU -> Dropout(0.1)
               -> global max-pool -> concat to per-point features
               -> Linear(192) -> LN -> ReLU -> Residual -> Dropout(0.1)
               -> Linear(32)  -> LN -> ReLU
               -> Linear(1)   -> Softplus(beta=5)
      Output: tau = Softplus(f(x))
    """

    def __init__(self, feature_dim: int = 17, hidden_dims=(128, 64, 32)):
        super().__init__()
        self.feature_dim = feature_dim

        # Stage 1: per-point encoding
        self.fc1 = nn.Linear(feature_dim, hidden_dims[0])
        self.ln1 = nn.LayerNorm(hidden_dims[0])

        # Global context: pool -> project -> concat with per-point
        self.global_proj = nn.Linear(hidden_dims[0], hidden_dims[0])

        # Stage 2: per-point + global context
        self.input_proj = nn.Linear(feature_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0] * 2, hidden_dims[1])
        self.fc2_perpoint = nn.Linear(hidden_dims[0], hidden_dims[1])  # no-global variant
        self.ln2 = nn.LayerNorm(hidden_dims[1])

        # Stage 3
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.ln3 = nn.LayerNorm(hidden_dims[2])

        self.fc_out = nn.Linear(hidden_dims[2], 1)

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus(beta=5)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.ndim < 2:
                if 'bias' in name:
                    nn.init.zeros_(param)
                continue
            nn.init.xavier_uniform_(param, gain=1.0)

    def forward(self, features: torch.Tensor, noise_std: torch.Tensor = None,
                disable_global_ctx: bool = False) -> torch.Tensor:
        """Forward pass with global context aggregation.

        Args:
            features: (N, 17) feature tensor
            noise_std: (N,) or scalar tensor (unused, kept for API compatibility)
            disable_global_ctx: if True, skip global max-pool aggregation (ablation)
        Returns:
            tau: (N,) per-point safety margins
        """
        # Stage 1: per-point encoding
        x = self.dropout(self.relu(self.ln1(self.fc1(features))))  # (N, 128)

        if disable_global_ctx:
            # Skip global context: use per-point features directly
            # Project x back to match fc2 input dim (128→64, no concat needed)
            x = self.relu(self.ln2(self.fc2_perpoint(x)))            # (N, 64)
        else:
            # Global context: max-pool over all points
            global_ctx = x.max(dim=0, keepdim=True)[0]          # (1, 128)
            global_ctx = self.global_proj(global_ctx)             # (1, 128)
            global_ctx = global_ctx.expand(x.shape[0], -1)       # (N, 128)

            # Concatenate per-point + global
            x = torch.cat([x, global_ctx], dim=1)                # (N, 256)
            x = self.relu(self.ln2(self.fc2(x)))                  # (N, 64)

        x = x + self.input_proj(features)[:, :64]            # residual (take first 64)
        x = self.dropout(x)

        # Stage 3
        x = self.relu(self.ln3(self.fc3(x)))                  # (N, 32)

        tau_base = self.softplus(self.fc_out(x)).squeeze(-1)  # (N,)

        return tau_base

    @property
    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class CPSAAsymmetricLoss(nn.Module):
    """Loss: coverage + collision_prob + narrow_passage + ranking + smoothness + L2."""

    def __init__(
        self,
        coverage_weight: float = 5.0,
        collision_prob_weight: float = 3.0,
        narrow_passage_weight: float = 1.0,
        ranking_weight: float = 1.5,
        smoothness_weight: float = 0.01,
        reg_weight: float = 0.001,
        l1_weight: float = 0.05,
        ranking_margin: float = 0.005,
        coverage_rate: float = 0.90,
        unsafe_penalty: float = 4.0,
        safety_threshold: float = 0.15,
        collision_temp: float = 0.02,
    ):
        super().__init__()
        self.coverage_weight = coverage_weight
        self.collision_prob_weight = collision_prob_weight
        self.narrow_passage_weight = narrow_passage_weight
        self.ranking_weight = ranking_weight
        self.smoothness_weight = smoothness_weight
        self.reg_weight = reg_weight
        self.l1_weight = l1_weight
        self.ranking_margin = ranking_margin
        self.coverage_rate = coverage_rate
        self.unsafe_penalty = unsafe_penalty
        self.safety_threshold = safety_threshold
        self.collision_temp = collision_temp

    def forward(
        self,
        tau: torch.Tensor,
        d_pred: torch.Tensor,
        d_gt: torch.Tensor,
        model: nn.Module = None,
        lateral_margins: torch.Tensor = None,
        is_approaching: torch.Tensor = None,
        noise_std: torch.Tensor = None,
    ) -> dict:
        N = tau.shape[0]
        device = tau.device

        residual = d_pred - d_gt
        overest_mask = residual > 0

        # === 1. Coverage Penalty ===
        overest = torch.relu(residual)
        coverage_violation = torch.relu(overest - tau)
        per_point_penalty = torch.where(
            overest_mask,
            self.unsafe_penalty * coverage_violation ** 2,
            coverage_violation * 0.01,
        )
        coverage_loss = per_point_penalty.mean()

        # === 2. Collision Probability Loss ===
        effective_dist = d_pred - tau
        collision_prob = torch.sigmoid(
            (self.safety_threshold - effective_dist) / self.collision_temp
        )
        collision_prob_loss = torch.where(
            collision_prob > 0.1,
            (collision_prob - 0.1) ** 2,
            torch.zeros_like(collision_prob),
        ).mean()

        # === 3. Narrow Passage Loss ===
        narrow_loss = torch.tensor(0.0, device=device)
        if lateral_margins is not None and N > 0:
            robot_half_width = 1.0
            max_allowed = torch.clamp(lateral_margins - robot_half_width, min=0.01)
            narrow_loss = torch.relu(tau - max_allowed).mean()

        # === 4. Pairwise Ranking Loss ===
        ranking_loss = torch.tensor(0.0, device=device)
        n_pairs = 0
        if N > 1:
            n_sample = min(N * (N - 1) // 2, 256)
            idx_i = torch.randint(0, N, (n_sample,), device=device)
            idx_j = torch.randint(0, N, (n_sample,), device=device)

            ri = residual[idx_i]
            rj = residual[idx_j]
            tau_i = tau[idx_i]
            tau_j = tau[idx_j]

            should_rank = ri > rj
            if should_rank.any():
                margin = torch.full_like(tau_i, self.ranking_margin)
                if is_approaching is not None:
                    approaching_i = is_approaching[idx_i].bool()
                    margin = torch.where(approaching_i, margin * 2.0, margin)
                violations = torch.relu(margin - (tau_i - tau_j))
                ranking_loss = (violations * should_rank.float()).mean()
                n_pairs = should_rank.sum().item()

        # === 5. Smoothness Loss ===
        if N > 1:
            smoothness_loss = torch.mean((tau[1:] - tau[:-1]) ** 2)
        else:
            smoothness_loss = torch.tensor(0.0, device=device)

        # === 6. L1 penalty on tau: discourage unnecessarily large margins ===
        l1_loss = tau.mean()

        # === 7. L2 Regularization ===
        reg_loss = torch.tensor(0.0, device=device)
        if model is not None:
            for param in model.parameters():
                reg_loss = reg_loss + param.pow(2).sum()
            reg_loss = self.reg_weight * reg_loss

        total_loss = (
            self.coverage_weight * coverage_loss
            + self.collision_prob_weight * collision_prob_loss
            + self.narrow_passage_weight * narrow_loss
            + self.ranking_weight * ranking_loss
            + self.smoothness_weight * smoothness_loss
            + self.l1_weight * l1_loss
            + reg_loss
        )

        if overest_mask.any():
            covered = (tau[overest_mask] >= residual[overest_mask]).float().mean()
        else:
            covered = torch.tensor(1.0, device=device)

        return {
            'total': total_loss,
            'coverage': covered.detach(),
            'coverage_loss': coverage_loss.detach(),
            'collision_prob_loss': collision_prob_loss.detach(),
            'narrow_passage_loss': narrow_loss.detach() if isinstance(narrow_loss, torch.Tensor) else narrow_loss,
            'ranking_loss': ranking_loss.detach(),
            'smoothness': smoothness_loss.detach(),
            'reg': reg_loss.detach(),
            'tau_mean': tau.mean().detach(),
            'tau_std': tau.std().detach() if N > 1 else torch.tensor(0.0),
            'n_ranking_pairs': n_pairs,
        }


class CPSACalibrator:
    """Post-hoc conformal calibration with dynamic q_star support."""

    def __init__(self, epsilon: float = 0.05, window_size: int = 100):
        self.epsilon = epsilon
        self.window_size = window_size
        self.q_star = 0.0
        self.q_star_conditional = {}
        self.residuals_window = []

    def compute_q_star(self, tau_raw: np.ndarray, d_pred: np.ndarray,
                       d_gt: np.ndarray, d_min: float = 0.1):
        overestimation = d_pred - d_gt
        residuals = np.maximum(0, overestimation - tau_raw)

        n = len(residuals)
        if n == 0:
            self.q_star = 0.0
            return 0.0

        level = min(np.ceil((1 - self.epsilon) * (n + 1)) / n, 1.0)
        self.q_star = float(np.quantile(residuals, level))
        return self.q_star

    def compute_q_star_conditional(self, tau_raw: np.ndarray, d_pred: np.ndarray,
                                    d_gt: np.ndarray, noise_std: np.ndarray):
        self.q_star_conditional = {}
        for ns in np.unique(noise_std):
            mask = noise_std == ns
            if mask.sum() < 10:
                continue
            q = self.compute_q_star(tau_raw[mask], d_pred[mask], d_gt[mask])
            self.q_star_conditional[float(ns)] = q
        return self.q_star_conditional

    def update_window(self, tau_raw: np.ndarray, d_pred: np.ndarray,
                      d_gt: np.ndarray, d_min: float = 0.1):
        overestimation = d_pred - d_gt
        residuals = np.maximum(0, overestimation - tau_raw)
        self.residuals_window.extend(residuals.tolist())
        if len(self.residuals_window) > self.window_size * 2:
            self.residuals_window = self.residuals_window[-self.window_size:]

    def compute_dynamic_q(self, tau_raw_mean: float, q_star: float) -> float:
        """Scene-adaptive q_star: reduce when network already outputs large tau."""
        tau_max_target = 0.04
        scale = max(0.4, 1.0 - tau_raw_mean / tau_max_target)
        return q_star * scale


class OnlineCPSAAdapter:
    """Online fine-tuning for CPSA risk network during deployment.

    Maintains a replay buffer of recent hard samples and runs 1-2 gradient
    steps after each episode to adapt tau predictions to the current
    environment/noise distribution. This combines CPSA's per-point granularity
    with ACI's online adaptation ability.

    Usage:
        adapter = OnlineCPSAAdapter(net, feat_mean, feat_std)
        # ... each episode ...
        for each step: features = build_features(...); adapter.collect(features, d_pred)
        adapter.feedback(d_pred, d_gt, outcome='collision')  # triggers update
    """

    def __init__(
        self,
        net: CPSARiskNet,
        feat_mean: torch.Tensor,
        feat_std: torch.Tensor,
        buffer_size: int = 2000,
        lr: float = 1e-4,
        update_steps: int = 2,
        coverage_weight: float = 10.0,
        l1_weight: float = 0.01,
    ):
        self.net = net
        self.feat_mean = feat_mean
        self.feat_std = feat_std
        self.buffer_size = buffer_size
        self.epsilon = 0.05     # ACI-style target error rate
        self.update_steps = update_steps

        self.optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        self.loss_fn = CPSAAsymmetricLoss(
            coverage_weight=coverage_weight,
            collision_prob_weight=1.0,
            narrow_passage_weight=0.5,
            ranking_weight=0.5,
            smoothness_weight=0.001,
            reg_weight=0.0,
            l1_weight=l1_weight,
            unsafe_penalty=8.0,
        )

        # Ring buffer: list of (features_norm, d_pred, d_gt, noise_std, lateral_margin, is_approaching)
        self.buffer: list = []
        self.buffer_write = 0

        # Stats
        self.n_updates = 0
        self.n_collisions = 0
        self.history: list[dict] = []

    def feedback(
        self,
        features: torch.Tensor,
        d_pred: torch.Tensor,
        d_gt: torch.Tensor,
        outcome: str = 'unknown',
        noise_std: float = 0.0,
        lateral_margin: float = 5.0,
        is_approaching: torch.Tensor = None,
    ):
        """Process episode feedback and trigger online update.

        Args:
            features: raw (unnormalized) per-point features, shape (N, 17)
            d_pred: predicted distances, shape (N,)
            d_gt: ground-truth distances (or proxy), shape (N,)
            outcome: 'arrived', 'collision', 'timeout', 'done'
            noise_std: sensor noise level for this episode
            lateral_margin: scene lateral margin
            is_approaching: per-point approaching flag
        """
        if features is None or features.shape[0] == 0:
            return

        n = features.shape[0]

        if is_approaching is None:
            is_approaching = torch.zeros(n, dtype=torch.bool)

        # Normalize features
        fm = self.feat_mean.to(features.device)
        fs = self.feat_std.to(features.device)
        if fm.shape[0] < features.shape[1]:
            pad = features.shape[1] - fm.shape[0]
            fm = torch.cat([fm, torch.zeros(pad, device=features.device)])
            fs = torch.cat([fs, torch.ones(pad, device=features.device)])
        features_norm = (features - fm) / (fs + 1e-6)
        features_norm = torch.nan_to_num(features_norm, nan=0.0, posinf=10.0, neginf=-10.0)

        ns_t = torch.full((n,), noise_std)
        lm_t = torch.full((n,), lateral_margin)

        # ACI-style asymmetric loss scaling
        # collision: err_t=1 → scale = 1.0       (full update, push tau up)
        # safe:      err_t=0 → scale = 0.00025   (tiny, prevent tau drift)
        if outcome == 'collision':
            loss_scale = 1.0
            multiplier = 5
            self.n_collisions += 1
        else:
            loss_scale = 0.00025  # gamma * epsilon = 0.005 * 0.05
            multiplier = 1

        for _ in range(multiplier):
            if len(self.buffer) < self.buffer_size:
                self.buffer.append((features_norm, d_pred, d_gt, ns_t, lm_t, is_approaching, loss_scale))
            else:
                idx = self.buffer_write % self.buffer_size
                self.buffer[idx] = (features_norm, d_pred, d_gt, ns_t, lm_t, is_approaching, loss_scale)
            self.buffer_write += 1

        # Run gradient updates
        self._update()

        # Log
        with torch.no_grad():
            tau = self.net(features_norm.to(next(self.net.parameters()).device))
            self.history.append({
                'outcome': outcome,
                'n_points': n,
                'tau_mean': tau.mean().item(),
                'buffer_size': len(self.buffer),
            })


    def _update(self):
        """Run gradient steps on replay buffer (one entry per call for variable-length episodes)."""
        if len(self.buffer) < 1:
            return

        device = next(self.net.parameters()).device
        self.net.train()

        for _ in range(self.update_steps):
            idx = torch.randint(0, len(self.buffer), (1,)).item()
            entry = self.buffer[idx]
            batch_feat = entry[0].to(device)
            batch_dp   = entry[1].to(device)
            batch_dg   = entry[2].to(device)
            batch_ns   = entry[3].to(device)
            batch_lm   = entry[4].to(device)
            batch_ia   = entry[5].to(device)
            loss_scale = entry[6]

            self.optimizer.zero_grad()
            tau = self.net(batch_feat)
            loss_dict = self.loss_fn(
                tau, batch_dp, batch_dg, noise_std=batch_ns,
                lateral_margins=batch_lm, is_approaching=batch_ia,
            )
            (loss_dict['total'] * loss_scale).backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()

        self.net.eval()
        self.n_updates += 1

    def get_stats(self) -> dict:
        if not self.history:
            return {'n_updates': 0, 'n_collisions': 0, 'buffer_size': 0, 'recent_tau_mean': 0}
        recent = self.history[-10:]
        return {
            'n_updates': self.n_updates,
            'n_collisions': self.n_collisions,
            'buffer_size': len(self.buffer),
            'recent_tau_mean': sum(h['tau_mean'] for h in recent) / len(recent),
            'recent_outcomes': [h['outcome'] for h in recent],
        }
