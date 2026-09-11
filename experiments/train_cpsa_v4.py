"""
Train CPSA-v4 Risk Network for NeuPAN.

V4 upgrades:
  - 17-D features: +d_pred_delta, +radius_delta, +lateral_margin, +passage_ratio
  - Global context aggregation (PointNet++ style max-pool)
  - Collision prob loss + narrow passage loss
  - Teacher pretraining from CP-DW tau (10 epochs MSE)
  - SWA for last 20 epochs
  - Noise injection on input features during training
  - Hard sample (collision/timeout) 5x oversampling

Usage:
    python experiments/train_cpsa_v4.py collect --noise_levels 0.0 0.02 0.03 0.05 --n_episodes 50
    python experiments/train_cpsa_v4.py train --epochs 300
    python experiments/train_cpsa_v4.py validate --noise_levels 0.0 0.02 0.05 --n_episodes 30
"""

import sys, os, argparse, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cvxpy as cp
import yaml
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, SWALR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neupan import neupan
from neupan.robot import robot
from neupan.blocks.cpsa_risk_net import (
    CPSARiskNet, CPSAAsymmetricLoss, CPSACalibrator,
)
from neupan.blocks import ObsPointNet
from neupan.configuration import np_to_tensor, tensor_to_np
from irsim.env import EnvBase


# ============================================================================
# Dataset with 17-D features
# ============================================================================

class NavigationPointDataset(Dataset):
    """Per-point dataset from navigation episodes with full 17-D features."""

    def __init__(self, data_path: str):
        d = torch.load(data_path, map_location='cpu', weights_only=False)
        self.d_pred = d['d_pred']
        self.d_gt = d['d_gt']
        self.residual = d['residual']
        self.radius = d.get('point_radius', torch.zeros(len(self.d_pred)))
        self.angle = d.get('angle_to_robot', torch.zeros(len(self.d_pred)))
        self.density = d.get('local_density', torch.ones(len(self.d_pred)) * 5.0)
        self.speed = d.get('speed', torch.zeros(len(self.d_pred)))
        self.noise_std = d.get('noise_std', torch.zeros(len(self.d_pred)))
        self.env_type = d.get('env_type', torch.zeros(len(self.d_pred), dtype=torch.long))
        self.outcomes = d.get('episode_outcome', ['unknown'] * len(self.d_pred))
        # Temporal features (may not exist in old data)
        self.d_pred_delta = d.get('d_pred_delta', torch.zeros(len(self.d_pred)))
        self.radius_delta = d.get('radius_delta', torch.zeros(len(self.d_pred)))
        self.lateral_margin = d.get('lateral_margin', torch.ones(len(self.d_pred)) * 5.0)
        self.passage_ratio = d.get('passage_ratio', torch.ones(len(self.d_pred)) * 3.0)
        # Teacher tau from CP-DW (optional)
        self.teacher_tau = d.get('teacher_tau', None)
        # is_approaching flag for dynamic points
        self.is_approaching = d.get('is_approaching', torch.zeros(len(self.d_pred), dtype=torch.bool))

    def __len__(self):
        return len(self.d_pred)

    def __getitem__(self, idx):
        item = {
            'features': self._build_features(idx),
            'd_pred': self.d_pred[idx],
            'd_gt': self.d_gt[idx],
            'residual': self.residual[idx],
            'noise_std': self.noise_std[idx],
            'is_approaching': self.is_approaching[idx],
            'lateral_margin': self.lateral_margin[idx],
        }
        if self.teacher_tau is not None:
            item['teacher_tau'] = self.teacher_tau[idx]
        return item

    def _build_features(self, indices):
        eps = 1e-6
        if isinstance(indices, (list, np.ndarray, torch.Tensor)):
            idx = indices
        else:
            idx = [indices]

        d_pred = self.d_pred[idx]
        radius = self.radius[idx]
        angle = self.angle[idx]
        proximity_ratio = torch.abs(d_pred) / (radius + eps)
        d_pred_sq = d_pred ** 2
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        density = self.density[idx]
        in_radius = torch.clamp(density / 5.0, 0, 1)
        speed = self.speed[idx]
        d_pred_rank = torch.zeros(len(d_pred))
        noise_std = self.noise_std[idx]
        d_pred_delta = self.d_pred_delta[idx]
        radius_delta = self.radius_delta[idx]
        lateral_margin = self.lateral_margin[idx]
        passage_ratio = self.passage_ratio[idx]
        env_type = self.env_type[idx].float() / 4.0

        features = torch.stack([
            d_pred, radius, angle, proximity_ratio, d_pred_sq,
            cos_angle, sin_angle,
            density, in_radius,
            speed, d_pred_rank, noise_std,
            d_pred_delta, radius_delta,
            lateral_margin, passage_ratio,
            env_type,
        ], dim=1)

        features = torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
        return features

    def build_all_features(self):
        return self._build_features(list(range(len(self))))

    def get_sample_weights(self):
        """Compute per-sample weights: balance across noise levels + 5x for hard samples."""
        noise_np = self.noise_std.numpy()
        unique_ns, counts = np.unique(noise_np, return_counts=True)
        total = len(noise_np)
        weight_per_ns = total / (len(unique_ns) * counts.astype(float))
        ns_to_weight = dict(zip(unique_ns, weight_per_ns))
        weights = torch.tensor([ns_to_weight[ns] for ns in noise_np], dtype=torch.float32)

        # 5x oversampling for hard samples (collision or timeout episodes)
        outcomes = self.outcomes
        for i, outcome in enumerate(outcomes):
            if outcome in ('collision', 'timeout'):
                weights[i] *= 5.0

        return weights


# ============================================================================
# Data Collection with 17-D features + teacher tau
# ============================================================================

def compute_d_gt_batch(points_local, G_np, h_np):
    edge_dim = G_np.shape[0]
    n = len(points_local)
    d_gt = np.zeros(n)

    mu_var = cp.Variable((edge_dim, 1), nonneg=True)
    p = cp.Parameter((2, 1))
    prob = cp.Problem(
        cp.Maximize(mu_var.T @ (G_np @ p - h_np)),
        [cp.norm(G_np.T @ mu_var) <= 1],
    )

    for i in range(n):
        p.value = points_local[i].reshape(2, 1)
        try:
            prob.solve(solver=cp.ECOS, warm_start=True)
            d_gt[i] = prob.value if prob.value is not None else 0.0
        except Exception:
            d_gt[i] = 0.0

    return d_gt


def compute_teacher_tau(d_pred, noise_std, q_over, robot_influence_range=5.0):
    """Compute CP-DW style teacher tau for pretraining."""
    d_pred_np = d_pred if isinstance(d_pred, np.ndarray) else d_pred.numpy()
    noise_np = noise_std if isinstance(noise_std, np.ndarray) else noise_std.numpy()

    weights = np.clip(1.0 - np.abs(d_pred_np) / robot_influence_range, 0.1, 1.0)
    teacher_tau = np.maximum(0, q_over * weights)
    return teacher_tau


def global_to_local(points_global, state):
    x, y, theta = state.flatten()[:3]
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx = points_global[0] - x
    dy = points_global[1] - y
    local_x = cos_t * dx + sin_t * dy
    local_y = -sin_t * dx + cos_t * dy
    return np.column_stack([local_x, local_y])


def compute_lateral_margin_np(local_pts):
    """Compute lateral margin from local point cloud (numpy)."""
    if len(local_pts) < 2:
        return 10.0, 3.0
    y = local_pts[:, 1]
    left_mask = y > 0
    right_mask = y < 0
    left_min = np.abs(y[left_mask]).min() if left_mask.any() else 10.0
    right_min = np.abs(y[right_mask]).min() if right_mask.any() else 10.0
    return min(left_min, right_min), (left_min + right_min) / 2.0


ENVIRONMENTS = {
    'corridor': ('example/corridor/diff/env.yaml', 'example/corridor/diff/planner.yaml'),
    'convex_obs': ('example/convex_obs/diff/env.yaml', 'example/convex_obs/diff/planner.yaml'),
    'pf_obs': ('example/pf_obs/diff/env.yaml', 'example/pf_obs/diff/planner.yaml'),
    'dyna_obs': ('example/dyna_obs/diff/env.yaml', 'example/dyna_obs/diff/planner.yaml'),
    'non_obs': ('example/non_obs/diff/env.yaml', 'example/non_obs/diff/planner.yaml'),
}


def collect_data(args):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    noise_levels = args.noise_levels
    n_episodes = args.n_episodes
    max_steps = args.max_steps
    max_points = args.max_points_per_step
    sample_interval = args.sample_interval

    r = robot(receding=10, step_time=0.1, kinematics='diff', length=1.6, width=2.0)
    G_np, h_np = r.G, r.h
    G_t, h_t = np_to_tensor(G_np), np_to_tensor(h_np)

    model = ObsPointNet(input_dim=2, output_dim=G_np.shape[0])
    orig_state = torch.load(
        'example/model/diff_robot_default/model_5000.pth', map_location='cpu',
        weights_only=False,
    )
    model.load_from_obs_point_net(orig_state)
    model.eval()

    # Calibrate q_over for teacher tau
    q_over_per_noise = {}
    for ns in noise_levels:
        q_over_per_noise[ns] = {0.0: 0.011, 0.02: 0.062, 0.03: 0.096, 0.05: 0.15}.get(ns, 0.0)

    all_data = []
    env_type_map = {'corridor': 0, 'convex_obs': 1, 'pf_obs': 2, 'dyna_obs': 3, 'non_obs': 4}

    for noise_std in noise_levels:
        print(f"\n{'='*60}")
        print(f"COLLECTING — noise={noise_std*100:.0f}cm")
        print(f"{'='*60}")

        for env_name, (env_file, plan_file) in ENVIRONMENTS.items():
            print(f"\n--- {env_name} ---")

            env_cfg = yaml.safe_load(open(os.path.join(base_dir, env_file)))
            for rob in env_cfg['robot']:
                for s in rob.get('sensors', []):
                    if noise_std > 0:
                        s['noise'] = True
                        s['std'] = noise_std
                    else:
                        s['noise'] = False
            plan_cfg = yaml.safe_load(open(os.path.join(base_dir, plan_file)))
            plan_cfg['time_print'] = False

            tmp_e = f'/tmp/v4_collect_{env_name}_{noise_std*100:.0f}cm.yaml'
            tmp_p = f'/tmp/v4_plan_{env_name}_{noise_std*100:.0f}cm.yaml'
            yaml.dump(env_cfg, open(tmp_e, 'w'))
            yaml.dump(plan_cfg, open(tmp_p, 'w'))

            for ep in range(n_episodes):
                env = EnvBase(tmp_e, display=False, save_ani=False)
                planner = neupan.init_from_yaml(tmp_p)

                outcome = 'timeout'
                ep_data = []
                t0 = time.time()
                prev_d_pred = None
                prev_radius = None

                for step in range(max_steps):
                    state = env.get_robot_state()
                    scan = env.get_lidar_scan()
                    points_global = planner.scan_to_point(state, scan)

                    if points_global is not None and points_global.shape[1] > 3:
                        action, info = planner(state, points_global, None)
                    else:
                        action = np.zeros((2, 1))
                        info = {'stop': False, 'arrive': False}
                        points_global = None

                    if points_global is not None and step % sample_interval == 0:
                        local_pts = global_to_local(points_global, state)
                        n_pts = len(local_pts)

                        if n_pts > 3:
                            local_t = torch.tensor(local_pts, dtype=torch.float32)
                            with torch.no_grad():
                                features_enc = model.encoder(local_t)
                                mu = model.distance_head(features_enc)

                            Gp_h = G_t @ local_t.T - h_t
                            d_pred = torch.sum(mu * Gp_h.T, dim=1).numpy()

                            if n_pts > max_points:
                                idx = np.argsort(np.abs(d_pred))[:max_points]
                                local_pts = local_pts[idx]
                                d_pred = d_pred[idx]

                            n_keep = len(local_pts)
                            d_gt = compute_d_gt_batch(local_pts, G_np, h_np)

                            speed = 0.0
                            if action is not None:
                                try:
                                    speed = float(np.linalg.norm(action))
                                except Exception:
                                    speed = 0.0

                            radii = np.sqrt(local_pts[:, 0]**2 + local_pts[:, 1]**2)

                            # Temporal deltas
                            d_pred_delta = np.zeros(n_keep)
                            radius_delta = np.zeros(n_keep)
                            if prev_d_pred is not None and len(prev_d_pred) == n_keep:
                                d_pred_delta = d_pred - prev_d_pred
                                radius_delta = radii - prev_radius[:n_keep] if prev_radius is not None else np.zeros(n_keep)
                            elif prev_d_pred is not None:
                                min_len = min(len(prev_d_pred), n_keep)
                                d_pred_delta[:min_len] = d_pred[:min_len] - prev_d_pred[:min_len]
                                if prev_radius is not None:
                                    radius_delta[:min_len] = radii[:min_len] - prev_radius[:min_len]

                            prev_d_pred = d_pred.copy()
                            prev_radius = radii.copy()

                            # Passage features
                            lateral_margin_val, passage_ratio_val = compute_lateral_margin_np(local_pts)

                            # Teacher tau (CP-DW style)
                            q_over = q_over_per_noise.get(noise_std, 0.0)
                            teacher_tau = compute_teacher_tau(d_pred, np.full(n_keep, noise_std), q_over)

                            # is_approaching: radius shrinking
                            is_approaching = radius_delta < -0.01

                            for i in range(n_keep):
                                ep_data.append({
                                    'd_pred': float(d_pred[i]),
                                    'd_gt': float(d_gt[i]),
                                    'residual': float(d_pred[i] - d_gt[i]),
                                    'radius': float(radii[i]),
                                    'angle': float(np.arctan2(local_pts[i, 1], local_pts[i, 0])),
                                    'density': 5.0,
                                    'speed': speed,
                                    'step': step,
                                    'noise_std': noise_std,
                                    'env_type': env_type_map.get(env_name, 0),
                                    'd_pred_delta': float(d_pred_delta[i]),
                                    'radius_delta': float(radius_delta[i]),
                                    'lateral_margin': float(lateral_margin_val),
                                    'passage_ratio': float(passage_ratio_val),
                                    'teacher_tau': float(teacher_tau[i]),
                                    'is_approaching': bool(is_approaching[i]),
                                })

                    env.step(action)

                    if info.get('stop'):
                        outcome = 'collision'
                        break
                    if info.get('arrive'):
                        outcome = 'arrived'
                        break
                    if env.done():
                        outcome = 'done'
                        break

                env.end(0)

                for d in ep_data:
                    d['outcome'] = outcome
                all_data.extend(ep_data)

                dt = time.time() - t0
                print(f"  Ep {ep+1:2d}/{n_episodes}: {outcome:8s} | "
                      f"{len(ep_data):5d} pts | {dt:.1f}s")

    # Save combined data
    out_path = 'experiments/cp_training_data/v4_combined_data.pt'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    N = len(all_data)
    save_dict = {
        'd_pred': torch.tensor([d['d_pred'] for d in all_data], dtype=torch.float32),
        'd_gt': torch.tensor([d['d_gt'] for d in all_data], dtype=torch.float32),
        'residual': torch.tensor([d['residual'] for d in all_data], dtype=torch.float32),
        'point_radius': torch.tensor([d['radius'] for d in all_data], dtype=torch.float32),
        'angle_to_robot': torch.tensor([d['angle'] for d in all_data], dtype=torch.float32),
        'local_density': torch.tensor([d['density'] for d in all_data], dtype=torch.float32),
        'speed': torch.tensor([d['speed'] for d in all_data], dtype=torch.float32),
        'noise_std': torch.tensor([d['noise_std'] for d in all_data], dtype=torch.float32),
        'env_type': torch.tensor([d.get('env_type', 0) for d in all_data], dtype=torch.long),
        'step_index': torch.tensor([d['step'] for d in all_data], dtype=torch.long),
        'episode_outcome': [d['outcome'] for d in all_data],
        'd_pred_delta': torch.tensor([d.get('d_pred_delta', 0) for d in all_data], dtype=torch.float32),
        'radius_delta': torch.tensor([d.get('radius_delta', 0) for d in all_data], dtype=torch.float32),
        'lateral_margin': torch.tensor([d.get('lateral_margin', 5.0) for d in all_data], dtype=torch.float32),
        'passage_ratio': torch.tensor([d.get('passage_ratio', 3.0) for d in all_data], dtype=torch.float32),
        'teacher_tau': torch.tensor([d.get('teacher_tau', 0) for d in all_data], dtype=torch.float32),
        'is_approaching': torch.tensor([d.get('is_approaching', False) for d in all_data], dtype=torch.bool),
        'n_total': N,
    }
    torch.save(save_dict, out_path)

    # Summary
    residuals = save_dict['residual'].numpy()
    noise_stds = save_dict['noise_std'].numpy()
    outcomes = save_dict['episode_outcome']

    print(f"\n{'='*60}")
    print(f"COMBINED DATA SUMMARY")
    print(f"{'='*60}")
    print(f"  Total points: {N}")

    from collections import Counter
    print(f"  Outcomes: {dict(Counter(outcomes))}")
    for ns in np.unique(noise_stds):
        mask = noise_stds == ns
        n_pts = mask.sum()
        over_ns = (residuals[mask] > 0).mean()
        print(f"    noise={ns*100:.0f}cm: {n_pts} pts, overest={over_ns:.1%}")

    print(f"  Saved to {out_path}")
    return out_path


# ============================================================================
# Training (v4: teacher pretrain + SWA + noise injection + hard sampling)
# ============================================================================

class FeatureNoiseInjector:
    """Inject Gaussian noise into input features during training."""

    def __init__(self, noise_ratio: float = 0.1):
        self.noise_ratio = noise_ratio

    def __call__(self, features: torch.Tensor, feat_std: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(features) * feat_std.unsqueeze(0) * self.noise_ratio
        return features + noise


def train_network(args):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    if args.data:
        data_path = args.data
    else:
        data_path = 'experiments/cp_training_data/v4_combined_data.pt'

    if not os.path.exists(data_path):
        print(f"ERROR: No training data at {data_path}")
        print("Run 'collect' first")
        return

    print(f"\n{'='*60}")
    print(f"TRAIN CPSA-v4 (17D + global context + teacher + SWA)")
    print(f"{'='*60}")

    dataset = NavigationPointDataset(data_path)
    N = len(dataset)
    print(f"  Dataset: {N} points")

    # Split: 70% train, 10% val, 20% calibration
    n_train = int(0.7 * N)
    n_val = int(0.1 * N)
    n_cal = N - n_train - n_val

    perm = np.random.RandomState(42).permutation(N)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    cal_idx = perm[n_train + n_val:]

    print(f"  Split: {n_train} train, {n_val} val, {n_cal} calibration")

    # Build features
    all_features = dataset.build_all_features()
    all_d_pred = dataset.d_pred
    all_d_gt = dataset.d_gt
    all_noise_std = dataset.noise_std
    all_lateral_margin = dataset.lateral_margin
    all_is_approaching = dataset.is_approaching.float()
    all_teacher_tau = dataset.teacher_tau

    train_features = all_features[train_idx]
    train_d_pred = all_d_pred[train_idx]
    train_d_gt = all_d_gt[train_idx]
    train_noise_std = all_noise_std[train_idx]
    train_lateral_margin = all_lateral_margin[train_idx]
    train_is_approaching = all_is_approaching[train_idx]
    train_teacher_tau = all_teacher_tau[train_idx] if all_teacher_tau is not None else None

    val_features = all_features[val_idx]
    val_d_pred = all_d_pred[val_idx]
    val_d_gt = all_d_gt[val_idx]
    val_noise_std = all_noise_std[val_idx]
    val_lateral_margin = all_lateral_margin[val_idx]
    val_is_approaching = all_is_approaching[val_idx]

    cal_features = all_features[cal_idx]
    cal_d_pred = all_d_pred[cal_idx]
    cal_d_gt = all_d_gt[cal_idx]
    cal_noise_std = all_noise_std[cal_idx]

    # Feature normalization
    feat_mean = train_features.mean(dim=0)
    feat_std = train_features.std(dim=0).clamp(min=1e-6)
    print(f"  Feature mean: {feat_mean.numpy().round(4)}")
    print(f"  Feature std:  {feat_std.numpy().round(4)}")

    train_features_norm = (train_features - feat_mean) / feat_std
    val_features_norm = (val_features - feat_mean) / feat_std
    cal_features_norm = (cal_features - feat_mean) / feat_std

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    net = CPSARiskNet(feature_dim=17, hidden_dims=(128, 64, 32)).to(device)
    print(f"  Parameters: {net.param_count}")

    loss_fn = CPSAAsymmetricLoss(
        coverage_weight=5.0,
        collision_prob_weight=3.0,
        narrow_passage_weight=1.0,
        ranking_weight=1.5,
        smoothness_weight=0.01,
        reg_weight=0.001,
    )

    noise_injector = FeatureNoiseInjector(noise_ratio=0.1)

    # Move to device
    train_features_d = train_features_norm.to(device)
    train_d_pred_d = train_d_pred.to(device)
    train_d_gt_d = train_d_gt.to(device)
    train_noise_std_d = train_noise_std.to(device)
    train_lateral_margin_d = train_lateral_margin.to(device)
    train_is_approaching_d = train_is_approaching.to(device)
    train_teacher_tau_d = train_teacher_tau.to(device) if train_teacher_tau is not None else None

    val_features_d = val_features_norm.to(device)
    val_d_pred_d = val_d_pred.to(device)
    val_d_gt_d = val_d_gt.to(device)
    val_noise_std_d = val_noise_std.to(device)
    val_lateral_margin_d = val_lateral_margin.to(device)
    val_is_approaching_d = val_is_approaching.to(device)

    feat_std_d = feat_std.to(device)

    # Sample weights (balanced + hard sample oversampling)
    weights = dataset.get_sample_weights()[train_idx].to(device)

    batch_size = args.batch_size
    n_batches = (n_train + batch_size - 1) // batch_size
    best_val_loss = float('inf')
    best_state = None

    log = {'train_loss': [], 'val_loss': [], 'tau_mean': [], 'tau_std': []}

    # ===== Phase 1: Teacher pretraining (10 epochs MSE) =====
    if train_teacher_tau_d is not None and train_teacher_tau_d.abs().sum() > 0:
        print(f"\n--- Phase 1: Teacher Pretraining (10 epochs MSE) ---")
        teacher_optimizer = optim.Adam(net.parameters(), lr=1e-3)
        teacher_loss_fn = nn.MSELoss()

        for epoch in range(10):
            net.train()
            epoch_losses = []
            weighted_indices = torch.multinomial(weights, n_train, replacement=True)

            for b in range(n_batches):
                start = b * batch_size
                end = min(start + batch_size, n_train)
                idx = weighted_indices[start:end]

                batch_feat = train_features_d[idx]
                batch_teacher = train_teacher_tau_d[idx]
                batch_ns = train_noise_std_d[idx]

                # Noise injection
                batch_feat_noisy = noise_injector(batch_feat, feat_std_d)

                teacher_optimizer.zero_grad()
                tau = net(batch_feat_noisy, noise_std=batch_ns)
                loss = teacher_loss_fn(tau, batch_teacher)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                teacher_optimizer.step()
                epoch_losses.append(loss.item())

            print(f"  Teacher epoch {epoch+1}: MSE={np.mean(epoch_losses):.6f}")
    else:
        print(f"\n--- No teacher tau data, skipping Phase 1 ---")

    # ===== Phase 2: Full loss training (300 epochs) =====
    print(f"\n--- Phase 2: Full Loss Training ({args.epochs} epochs) ---")
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    # SWA setup for last 20 epochs
    swa_start = args.epochs - 20
    swa_model = AveragedModel(net)
    swa_scheduler = SWALR(optimizer, swa_lr=args.lr * 0.1)

    for epoch in range(args.epochs):
        net.train()
        epoch_losses = []

        weighted_indices = torch.multinomial(weights, n_train, replacement=True)

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, n_train)
            idx = weighted_indices[start:end]

            batch_feat = train_features_d[idx]
            batch_dp = train_d_pred_d[idx]
            batch_dg = train_d_gt_d[idx]
            batch_ns = train_noise_std_d[idx]
            batch_lm = train_lateral_margin_d[idx]
            batch_ia = train_is_approaching_d[idx]

            # Noise injection
            batch_feat_noisy = noise_injector(batch_feat, feat_std_d)

            optimizer.zero_grad()
            tau = net(batch_feat_noisy, noise_std=batch_ns)
            loss_dict = loss_fn(
                tau, batch_dp, batch_dg, model=net, noise_std=batch_ns,
                lateral_margins=batch_lm, is_approaching=batch_ia,
            )
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

            if epoch >= swa_start:
                swa_model.update_parameters(net)
                swa_scheduler.step()
            else:
                optimizer.step()
                scheduler.step()

            epoch_losses.append(loss_dict)

        # Validation
        net.eval()
        with torch.no_grad():
            val_tau = net(val_features_d, noise_std=val_noise_std_d)
            val_loss_dict = loss_fn(
                val_tau, val_d_pred_d, val_d_gt_d, noise_std=val_noise_std_d,
                lateral_margins=val_lateral_margin_d, is_approaching=val_is_approaching_d,
            )

        avg_train_loss = np.mean([d['total'].item() for d in epoch_losses])
        avg_tau_mean = np.mean([d['tau_mean'].item() for d in epoch_losses])
        avg_tau_std = np.mean([d['tau_std'].item() for d in epoch_losses])

        log['train_loss'].append(avg_train_loss)
        log['val_loss'].append(val_loss_dict['total'].item())
        log['tau_mean'].append(avg_tau_mean)
        log['tau_std'].append(avg_tau_std)

        if val_loss_dict['total'].item() < best_val_loss:
            best_val_loss = val_loss_dict['total'].item()
            best_state = {k: v.clone() for k, v in net.state_dict().items()}

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
                  f"train={avg_train_loss:.4f} val={val_loss_dict['total'].item():.4f} "
                  f"tau={avg_tau_mean*100:.2f}cm+/-{avg_tau_std*100:.2f}cm "
                  f"cov={val_loss_dict['coverage'].item():.1%} "
                  f"cp={val_loss_dict['collision_prob_loss'].item():.4f} "
                  f"np={val_loss_dict['narrow_passage_loss'].item():.4f}")

    # Update SWA batch norm statistics
    if args.epochs > swa_start:
        torch.optim.swa_utils.update_bn(train_features_d[:1000], net)
        print(f"  SWA applied for epochs {swa_start+1}-{args.epochs}")

    # Load best model
    net.load_state_dict(best_state)
    net.eval()

    # Conformal calibration
    print(f"\n--- Conformal Calibration ---")
    cal_features_device = cal_features_norm.to(device)
    cal_d_pred_device = cal_d_pred.to(device)
    cal_noise_std_device = cal_noise_std.to(device)

    with torch.no_grad():
        cal_tau_raw = net(cal_features_device).cpu().numpy()

    calibrator = CPSACalibrator(epsilon=0.05, window_size=100)
    q_star = calibrator.compute_q_star(
        cal_tau_raw, cal_d_pred.numpy(), cal_d_gt.numpy(), d_min=0.1,
    )
    print(f"  Global q_star = {q_star*100:.2f}cm")

    q_star_cond = calibrator.compute_q_star_conditional(
        cal_tau_raw, cal_d_pred.numpy(), cal_d_gt.numpy(), cal_noise_std.numpy(),
    )
    print(f"  Conditional q_star:")
    for ns, q in sorted(q_star_cond.items()):
        print(f"    noise={ns*100:.0f}cm: q_star={q*100:.2f}cm")

    # Verify coverage
    cal_tau_final = np.maximum(0, cal_tau_raw + q_star)
    overestimation = cal_d_pred.numpy() - cal_d_gt.numpy()
    overest_mask = overestimation > 0
    if overest_mask.any():
        covered = (cal_tau_final[overest_mask] >= overestimation[overest_mask]).mean()
        print(f"  Calibration coverage: {covered:.1%} (target >= 95%)")

    # Save checkpoint
    out_path = 'experiments/cp_head_output/cpsa_v4_universal.pth'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    torch.save({
        'model_state': best_state,
        'config': {
            'version': 4,
            'feature_dim': 17,
            'hidden_dims': [128, 64, 32],
            'ranking_loss': True,
            'target_margin': 0.005,
            'noise_levels': 'mixed',
            'n_train': n_train,
            'n_val': n_val,
            'n_cal': n_cal,
        },
        'q_star': q_star,
        'q_star_fixed': q_star,
        'q_star_conditional': q_star_cond,
        'feature_mean': feat_mean,
        'feature_std': feat_std,
        'calibrator_epsilon': 0.05,
        'training_log': {k: [float(x) for x in v] for k, v in log.items()},
    }, out_path)

    print(f"\n  Saved to {out_path}")
    print(f"  q_star = {q_star*100:.2f}cm")
    print(f"  Best val loss = {best_val_loss:.4f}")

    return out_path


# ============================================================================
# Validation
# ============================================================================

def run_nav(
    method='vanilla', cp_checkpoint=None, noise_std=0.03,
    n_episodes=30, max_steps=500, env_name='corridor',
):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    env_file, plan_file = ENVIRONMENTS[env_name]

    env_cfg = yaml.safe_load(open(env_file))
    for rob in env_cfg['robot']:
        for s in rob.get('sensors', []):
            if noise_std > 0:
                s['noise'] = True
                s['std'] = noise_std
            else:
                s['noise'] = False

    plan_cfg = yaml.safe_load(open(plan_file))
    plan_cfg['time_print'] = False

    tmp_e = f'/tmp/v4_val_{method}_{noise_std*100:.0f}cm_{env_name}.yaml'
    tmp_p = f'/tmp/v4_val_plan_{method}_{noise_std*100:.0f}cm_{env_name}.yaml'
    yaml.dump(env_cfg, open(tmp_e, 'w'))
    yaml.dump(plan_cfg, open(tmp_p, 'w'))

    metrics = []
    for ep in range(n_episodes):
        env = EnvBase(tmp_e, display=False, save_ani=False)
        planner = neupan.init_from_yaml(tmp_p)

        if method == 'cpsa_v4' and cp_checkpoint is not None:
            planner.enable_risk_calibration(
                budget_strategy='cpsa',
                cp_checkpoint=cp_checkpoint,
                noise_std=noise_std,
            )

        min_dist = float('inf')
        prev = env.get_robot_state().flatten()[:2]
        path_len = 0.0
        speeds = []
        outcome = 'timeout'

        for step in range(max_steps):
            state = env.get_robot_state()
            scan = env.get_lidar_scan()
            points = planner.scan_to_point(state, scan)

            if hasattr(planner, 'min_distance') and planner.min_distance < min_dist:
                md = planner.min_distance
                min_dist = float(md.item() if hasattr(md, 'item') else md)

            if points is not None:
                action, info = planner(state, points, None)
                if action is not None:
                    try:
                        spd = float(np.linalg.norm(action))
                    except Exception:
                        spd = float(np.linalg.norm(action.detach().cpu().numpy()))
                    speeds.append(spd)
            else:
                action = np.zeros((2, 1))
                info = {'stop': False, 'arrive': False}

            env.step(action)
            cur = state.flatten()[:2]
            path_len += np.linalg.norm(cur - prev)
            prev = cur

            if info.get('stop'):
                outcome = 'collision'
                break
            if info.get('arrive'):
                outcome = 'arrived'
                break
            if env.done():
                outcome = 'done'
                break

        env.end(0)
        metrics.append({
            'outcome': outcome, 'steps': step + 1,
            'path_length': path_len, 'min_distance': min_dist,
            'avg_speed': float(np.mean(speeds)) if speeds else 0,
        })

    coll = sum(1 for m in metrics if m['outcome'] == 'collision')
    arrv = sum(1 for m in metrics if m['outcome'] == 'arrived')
    min_dists = [m['min_distance'] for m in metrics]
    paths = [m['path_length'] for m in metrics if m['outcome'] == 'arrived']

    return {
        'collisions': coll, 'collision_rate': coll / n_episodes,
        'arrivals': arrv, 'success_rate': arrv / n_episodes,
        'avg_path': float(np.mean(paths)) if paths else 0,
        'avg_min_distance': float(np.mean(min_dists)),
        'min_min_distance': float(np.min(min_dists)),
    }


def validate(args):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    ckpt = 'experiments/cp_head_output/cpsa_v4_universal.pth'
    if not os.path.exists(ckpt):
        print(f"ERROR: No checkpoint at {ckpt}")
        print("Run 'train' first")
        return

    print("=" * 70)
    print("VALIDATE CPSA-v4 (17D + global context)")
    print("=" * 70)

    all_results = {}

    for env_name in ENVIRONMENTS:
        for noise in args.noise_levels:
            key = f'{env_name}_{noise*100:.0f}cm'
            print(f"\n  {key}:", flush=True)

            r_vanilla = run_nav('vanilla', noise_std=noise, env_name=env_name, n_episodes=args.n_episodes)
            r_cpsa = run_nav('cpsa_v4', cp_checkpoint=ckpt, noise_std=noise, env_name=env_name, n_episodes=args.n_episodes)

            print(f"    vanilla: Succ={r_vanilla['success_rate']*100:.0f}% Coll={r_vanilla['collision_rate']*100:.0f}% MinD={r_vanilla['avg_min_distance']*100:.1f}cm")
            print(f"    cpsa_v4:  Succ={r_cpsa['success_rate']*100:.0f}% Coll={r_cpsa['collision_rate']*100:.0f}% MinD={r_cpsa['avg_min_distance']*100:.1f}cm")

            all_results[key] = {'vanilla': r_vanilla, 'cpsa_v4': r_cpsa}

    os.makedirs('experiments/exp_v4_output', exist_ok=True)
    with open('experiments/exp_v4_output/v4_validation.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to experiments/exp_v4_output/v4_validation.json")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='CPSA-v4 Training Pipeline')
    subparsers = parser.add_subparsers(dest='command')

    # Collect
    p_collect = subparsers.add_parser('collect')
    p_collect.add_argument('--noise_levels', type=float, nargs='+', default=[0.0, 0.02, 0.03, 0.05])
    p_collect.add_argument('--n_episodes', type=int, default=50)
    p_collect.add_argument('--max_steps', type=int, default=500)
    p_collect.add_argument('--max_points_per_step', type=int, default=25)
    p_collect.add_argument('--sample_interval', type=int, default=3)

    # Train
    p_train = subparsers.add_parser('train')
    p_train.add_argument('--data', type=str, default=None)
    p_train.add_argument('--epochs', type=int, default=100)
    p_train.add_argument('--lr', type=float, default=2e-3)
    p_train.add_argument('--batch_size', type=int, default=512)

    # Validate
    p_val = subparsers.add_parser('validate')
    p_val.add_argument('--noise_levels', type=float, nargs='+', default=[0.0, 0.02, 0.05])
    p_val.add_argument('--n_episodes', type=int, default=30)

    args = parser.parse_args()

    if args.command == 'collect':
        collect_data(args)
    elif args.command == 'train':
        train_network(args)
    elif args.command == 'validate':
        validate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
