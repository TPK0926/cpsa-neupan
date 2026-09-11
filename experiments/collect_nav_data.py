"""
Collect navigation-scenario training data for learnable CP head.

Runs vanilla NeuPAN in multiple environments with noisy lidar, collecting
per-point features and ground-truth labels at each timestep. This replaces
the old random-point calibration with realistic navigation data.

Saved data per point:
  - encoder_features (32): frozen DUNE encoder output
  - d_pred: DUNE predicted distance
  - d_gt: ground truth distance (CVX)
  - residual: d_pred - d_gt
  - point_radius: distance from robot center
  - angle_to_robot: angle relative to robot heading
  - local_density: average kNN distance in current frame
  - speed: current robot speed
  - step_index, episode_id, episode_outcome

Usage:
    python experiments/collect_nav_data.py --n_episodes 30 --noise_std 0.03
"""

import sys, os, argparse, time
import numpy as np
import torch
import cvxpy as cp
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neupan import neupan
from neupan.robot import robot
from neupan.blocks import ObsPointNet
from neupan.configuration import np_to_tensor, tensor_to_np
from irsim.env import EnvBase


def global_to_local(points_global, state):
    """Transform (2, N) global points to (N, 2) robot-local frame."""
    x, y, theta = state.flatten()[:3]
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx = points_global[0] - x
    dy = points_global[1] - y
    local_x = cos_t * dx + sin_t * dy
    local_y = -sin_t * dx + cos_t * dy
    return np.column_stack([local_x, local_y])


def compute_d_gt_batch(points_local, G_np, h_np):
    """Compute ground truth distance for each point using CVX (LP).

    Solves: max mu^T (G p - h)  s.t.  ||G^T mu|| <= 1, mu >= 0
    """
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


def compute_local_density(points, k=5):
    """Average distance to k nearest neighbors."""
    from scipy.spatial import cKDTree

    n = len(points)
    if n <= k:
        return np.full(n, 10.0)
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=min(k + 1, n))
    if dists.ndim == 1:
        return np.full(n, 10.0)
    return np.mean(dists[:, 1:], axis=1)


def compute_d_pred_batch(local_pts, feature_model, G_t, h_t):
    """Compute DUNE-predicted distance for each point.

    d_pred[i] = mu[i]^T @ (G @ p_i - h)
    """
    local_t = torch.tensor(local_pts, dtype=torch.float32)
    with torch.no_grad():
        features = feature_model.encoder(local_t)
        mu = feature_model.distance_head(features)

    Gp_h = G_t @ local_t.T - h_t
    d_pred = torch.sum(mu * Gp_h.T, dim=1)
    return d_pred.numpy(), features.numpy()


def collect_from_env(
    env_file, planner_file, feature_model, G_np, h_np, G_t, h_t,
    noise_std=0.03, n_episodes=30, max_steps=500,
    max_points_per_step=25, sample_interval=3,
):
    """Run vanilla NeuPAN and collect per-point training data."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_cfg = yaml.safe_load(open(os.path.join(base_dir, env_file)))
    for rob in env_cfg['robot']:
        for s in rob.get('sensors', []):
            s['noise'] = True
            s['std'] = noise_std

    plan_cfg = yaml.safe_load(open(os.path.join(base_dir, planner_file)))
    plan_cfg['time_print'] = False

    tmp_e = '/tmp/collect_env.yaml'
    tmp_p = '/tmp/collect_plan.yaml'
    yaml.dump(env_cfg, open(tmp_e, 'w'))
    yaml.dump(plan_cfg, open(tmp_p, 'w'))

    all_data = []

    for ep in range(n_episodes):
        env = EnvBase(tmp_e, display=False, save_ani=False)
        planner = neupan.init_from_yaml(tmp_p)

        outcome = 'timeout'
        ep_data = []
        t0 = time.time()

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

            # Collect data at intervals
            if points_global is not None and step % sample_interval == 0:
                local_pts = global_to_local(points_global, state)
                n_pts = len(local_pts)

                if n_pts > 3:
                    d_pred_np, enc_features = compute_d_pred_batch(
                        local_pts, feature_model, G_t, h_t,
                    )

                    # Subsample: keep closest points (most informative)
                    if n_pts > max_points_per_step:
                        idx = np.argsort(np.abs(d_pred_np))[:max_points_per_step]
                        local_pts = local_pts[idx]
                        d_pred_np = d_pred_np[idx]
                        enc_features = enc_features[idx]

                    n_keep = len(local_pts)

                    # Ground truth
                    d_gt = compute_d_gt_batch(local_pts, G_np, h_np)

                    # Auxiliary features
                    radius = np.sqrt(local_pts[:, 0]**2 + local_pts[:, 1]**2)
                    angle = np.arctan2(local_pts[:, 1], local_pts[:, 0])
                    density = compute_local_density(local_pts, k=min(5, n_keep - 1))

                    speed = 0.0
                    if action is not None:
                        try:
                            speed = float(np.linalg.norm(action))
                        except Exception:
                            speed = 0.0

                    for i in range(n_keep):
                        ep_data.append({
                            'enc': enc_features[i],
                            'd_pred': float(d_pred_np[i]),
                            'd_gt': float(d_gt[i]),
                            'residual': float(d_pred_np[i] - d_gt[i]),
                            'radius': float(radius[i]),
                            'angle': float(angle[i]),
                            'density': float(density[i]),
                            'speed': float(speed),
                            'step': step,
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

    return all_data


def save_data(all_data, output_path, noise_std, n_episodes):
    """Convert list of dicts to tensor dict and save."""
    N = len(all_data)
    enc = np.array([d['enc'] for d in all_data])
    d_pred = np.array([d['d_pred'] for d in all_data])
    d_gt = np.array([d['d_gt'] for d in all_data])
    residual = np.array([d['residual'] for d in all_data])
    radius = np.array([d['radius'] for d in all_data])
    angle = np.array([d['angle'] for d in all_data])
    density = np.array([d['density'] for d in all_data])
    speed = np.array([d['speed'] for d in all_data])
    step_idx = np.array([d['step'] for d in all_data])
    ep_id = np.array([d.get('ep_id', 0) for d in all_data])
    outcomes = [d['outcome'] for d in all_data]

    save_dict = {
        'encoder_features': torch.tensor(enc, dtype=torch.float32),
        'd_pred': torch.tensor(d_pred, dtype=torch.float32),
        'd_gt': torch.tensor(d_gt, dtype=torch.float32),
        'residual': torch.tensor(residual, dtype=torch.float32),
        'point_radius': torch.tensor(radius, dtype=torch.float32),
        'angle_to_robot': torch.tensor(angle, dtype=torch.float32),
        'local_density': torch.tensor(density, dtype=torch.float32),
        'speed': torch.tensor(speed, dtype=torch.float32),
        'step_index': torch.tensor(step_idx, dtype=torch.long),
        'episode_outcome': outcomes,
        'noise_std': noise_std,
        'n_episodes_per_env': n_episodes,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(save_dict, output_path)
    return save_dict


def print_summary(save_dict, residuals, d_pred, d_gt, outcomes):
    """Print data statistics."""
    print(f"\n{'='*60}")
    print("DATA SUMMARY")
    print(f"{'='*60}")
    print(f"  Total points: {len(residuals)}")
    print(f"  d_pred: mean={d_pred.mean():.4f}, std={d_pred.std():.4f}, "
          f"range=[{d_pred.min():.3f}, {d_pred.max():.3f}]")
    print(f"  d_gt:   mean={d_gt.mean():.4f}, std={d_gt.std():.4f}, "
          f"range=[{d_gt.min():.3f}, {d_gt.max():.3f}]")
    print(f"  Residual (d_pred - d_gt): mean={residuals.mean():.4f}, "
          f"std={residuals.std():.4f}")

    over_mask = residuals > 0
    under_mask = residuals < 0
    print(f"  Overestimation rate:  {over_mask.mean():.1%} "
          f"(mean mag: {residuals[over_mask].mean()*100:.2f}cm)" if over_mask.sum() > 0 else
          "  Overestimation: none")
    print(f"  Underestimation rate: {under_mask.mean():.1%} "
          f"(mean mag: {(-residuals[under_mask]).mean()*100:.2f}cm)" if under_mask.sum() > 0 else
          "  Underestimation: none")

    n_arr = sum(1 for o in outcomes if o == 'arrived')
    n_coll = sum(1 for o in outcomes if o == 'collision')
    print(f"  Episode outcomes: {n_arr} arrived, {n_coll} collision")


def main():
    parser = argparse.ArgumentParser(description='Collect navigation training data')
    parser.add_argument('--n_episodes', type=int, default=30)
    parser.add_argument('--noise_std', type=float, default=0.03)
    parser.add_argument('--max_points_per_step', type=int, default=25)
    parser.add_argument('--sample_interval', type=int, default=3)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    out_name = args.output or (
        f'experiments/cp_training_data/nav_data_noise{args.noise_std*100:.0f}cm.pt'
    )

    print("=" * 60)
    print("COLLECT NAVIGATION TRAINING DATA")
    print("=" * 60)
    print(f"  noise_std       = {args.noise_std*100:.0f}cm")
    print(f"  n_episodes      = {args.n_episodes} per env")
    print(f"  max_pts/step    = {args.max_points_per_step}")
    print(f"  sample_interval = every {args.sample_interval} steps")

    # Setup robot and feature model
    r = robot(receding=10, step_time=0.1, kinematics='diff', length=1.6, width=2.0)
    G_np, h_np = r.G, r.h
    edge_dim = G_np.shape[0]
    G_t, h_t = np_to_tensor(G_np), np_to_tensor(h_np)

    model = ObsPointNet(
        input_dim=2, output_dim=edge_dim,
    )
    orig_state = torch.load(
        'example/model/diff_robot_default/model_5000.pth', map_location='cpu',
    )
    model.load_from_obs_point_net(orig_state)
    model.eval()

    # Environments
    environments = [
        ('example/corridor/diff/env.yaml', 'example/corridor/diff/planner.yaml', 'corridor'),
        ('example/convex_obs/diff/env.yaml', 'example/convex_obs/diff/planner.yaml', 'convex_obs'),
        ('example/pf_obs/diff/env.yaml', 'example/pf_obs/diff/planner.yaml', 'pf_obs'),
    ]

    all_data = []
    global_ep_id = 0
    t_start = time.time()

    for env_file, plan_file, env_name in environments:
        print(f"\n--- {env_name} ---")
        data = collect_from_env(
            env_file, plan_file, model, G_np, h_np, G_t, h_t,
            noise_std=args.noise_std,
            n_episodes=args.n_episodes,
            max_points_per_step=args.max_points_per_step,
            sample_interval=args.sample_interval,
        )
        # Assign episode IDs
        for d in data:
            d['ep_id'] = global_ep_id
        all_data.extend(data)
        global_ep_id += args.n_episodes
        print(f"  Subtotal: {len(all_data)} points")

    # Save
    save_dict = save_data(all_data, out_name, args.noise_std, args.n_episodes)
    residuals = np.array([d['residual'] for d in all_data])
    d_pred = np.array([d['d_pred'] for d in all_data])
    d_gt = np.array([d['d_gt'] for d in all_data])
    outcomes = [d['outcome'] for d in all_data]

    print_summary(save_dict, residuals, d_pred, d_gt, outcomes)
    print(f"\n  Total time: {time.time()-t_start:.0f}s")
    print(f"  Saved to {out_name}")


if __name__ == "__main__":
    main()
