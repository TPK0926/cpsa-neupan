"""
Collect d_pred/d_gt training data from multiple environments.

Uses cp_dw (distance-weighted CP) to navigate each environment while recording
DUNE predicted distances and ground-truth distances at each step.
"""

import sys, os, yaml, time, argparse
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from neupan import neupan
from irsim.env import EnvBase

ENVIRONMENTS = {
    'corridor': {
        'desc': 'Narrow corridor with walls and obstacles',
        'diff': {'env': 'example/corridor/diff/env.yaml',
                 'planner': 'example/corridor/diff/planner.yaml'},
        'acker': {'env': 'example/corridor/acker/env.yaml',
                  'planner': 'example/corridor/acker/planner.yaml'},
    },
    'convex_obs': {
        'desc': 'Convex obstacle field',
        'diff': {'env': 'example/convex_obs/diff/env.yaml',
                 'planner': 'example/convex_obs/diff/planner.yaml'},
    },
    'pf_obs': {
        'desc': 'Path following with obstacles',
        'diff': {'env': 'example/pf_obs/diff/env.yaml',
                 'planner': 'example/pf_obs/diff/planner.yaml'},
    },
    'dyna_obs': {
        'desc': 'Dynamic obstacle field',
        'diff': {'env': 'example/dyna_obs/diff/env.yaml',
                 'planner': 'example/dyna_obs/diff/planner.yaml'},
    },
    'non_obs': {
        'desc': 'Non-convex obstacle field',
        'diff': {'env': 'example/non_obs/diff/env.yaml',
                 'planner': 'example/non_obs/diff/planner.yaml'},
    },
}


def make_env_yaml(env_name, robot_type, noise_std):
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    env_path = os.path.join(base_dir, ENVIRONMENTS[env_name][robot_type]['env'])
    with open(env_path) as f:
        cfg = yaml.safe_load(f)
    if noise_std > 0 and 'lidar' in cfg and isinstance(cfg['lidar'], dict):
        scenarios = cfg['lidar'].get('scenarios', [])
        if scenarios:
            for s in scenarios:
                s['std'] = noise_std
    return cfg


def collect_data(env_name, robot_type, noise_std, n_episodes=30,
                 max_steps=500, method='cp_dw', q_over=0.0, q_under=0.0):
    """Run navigation episodes and collect d_pred/d_gt pairs."""
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    os.chdir(base_dir)

    plan_path = os.path.join(base_dir, ENVIRONMENTS[env_name][robot_type]['planner'])
    plan_cfg = yaml.safe_load(open(plan_path))
    plan_cfg['time_print'] = False

    import tempfile
    env_cfg = make_env_yaml(env_name, robot_type, noise_std)
    tf_env = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(env_cfg, tf_env)
    tf_env.close()

    tf_plan = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(plan_cfg, tf_plan)
    tf_plan.close()

    all_d_pred = []
    all_d_gt = []
    all_pts_local = []
    all_noise_std = []
    all_env_type = []
    all_speed = []

    env_map = {'corridor': 0, 'convex_obs': 1, 'pf_obs': 2, 'dyna_obs': 3, 'non_obs': 4}

    for ep in range(n_episodes):
        env = EnvBase(tf_env.name, display=False, save_ani=False)
        planner = neupan.init_from_yaml(tf_plan.name)

        # Apply calibration
        if method == 'cp_dw' and noise_std > 0:
            planner.enable_risk_calibration(
                q_hat=q_over, budget_strategy='distance_weighted',
                collision_q_hat=q_under, noise_std=noise_std)

        prev_state = None
        for step in range(max_steps):
            state = env.get_robot_state()
            scan = env.get_lidar_scan()
            points = planner.scan_to_point(state, scan)

            # Collect d_pred/d_gt if available
            if (points is not None and points.shape[1] > 0
                    and hasattr(planner.pan, 'dune_layer')
                    and hasattr(planner.pan.dune_layer, 'distances_0')):
                d_pred_t = planner.pan.dune_layer.distances_0
                if d_pred_t is not None and d_pred_t.numel() > 0:
                    d_pred_np = d_pred_t.detach().cpu().numpy()

                    # Compute ground-truth distances
                    state_np = state.flatten()[:3]
                    pts_np = points  # (2, N)
                    dx = pts_np[0, :] - state_np[0]
                    dy = pts_np[1, :] - state_np[1]
                    d_gt_np = np.sqrt(dx**2 + dy**2)

                    # Robot-local coords
                    cos_t, sin_t = np.cos(-state_np[2]), np.sin(-state_np[2])
                    pts_local_x = dx * cos_t - dy * sin_t
                    pts_local_y = dx * sin_t + dy * cos_t

                    # Speed
                    speed = 0.0
                    if prev_state is not None:
                        ds = state.flatten()[:2] - prev_state[:2]
                        speed = np.linalg.norm(ds) / 0.1  # assuming dt=0.1

                    all_d_pred.append(d_pred_np)
                    all_d_gt.append(d_gt_np)
                    all_pts_local.append(np.stack([pts_local_x, pts_local_y], axis=0))
                    all_noise_std.append(noise_std)
                    all_env_type.append(env_map.get(env_name, 0))
                    all_speed.append(speed)

            action, info = planner(state, points, None)
            prev_state = state.flatten()[:3].copy()

            if info.get('stop') or info.get('arrive') or env.done():
                break
            env.step(action)

        env.end(0)

    os.unlink(tf_env.name)
    os.unlink(tf_plan.name)

    total_pts = sum(len(d) for d in all_d_pred)
    print(f"  Collected {len(all_d_pred)} frames, {total_pts} points from {env_name}")

    return {
        'd_pred': all_d_pred,
        'd_gt': all_d_gt,
        'pts_local': all_pts_local,
        'noise_std': all_noise_std,
        'env_type': all_env_type,
        'speed': all_speed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='experiments/training_data/multienv_data.npz')
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--noise-levels', nargs='+', type=float, default=[0.0, 0.02, 0.03, 0.05])
    parser.add_argument('--envs', nargs='+', default=['corridor', 'dyna_obs', 'non_obs', 'convex_obs', 'pf_obs'])
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load existing data if any
    existing_data = {}
    if os.path.exists(args.output):
        print(f"Loading existing data from {args.output}")
        existing = np.load(args.output, allow_pickle=True)
        for k in existing.files:
            existing_data[k] = list(existing[k])

    # Collect data
    all_data = defaultdict(list)
    for env_name in args.envs:
        for noise_std in args.noise_levels:
            print(f"\nCollecting {env_name} @ {noise_std*100:.0f}cm...")
            data = collect_data(
                env_name, 'diff', noise_std,
                n_episodes=args.episodes,
                method='cp_dw' if noise_std > 0 else 'vanilla',
            )
            for k, v in data.items():
                all_data[k].extend(v)

    # Also merge existing data
    for k, v in existing_data.items():
        if k in all_data:
            all_data[k] = v + all_data[k]
        else:
            all_data[k] = v

    # Save
    np.savez(args.output, **{k: np.array(v, dtype=object) for k, v in all_data.items()})
    total_pts = sum(len(d) for d in all_data['d_pred'])
    total_frames = len(all_data['d_pred'])
    print(f"\nTotal: {total_frames} frames, {total_pts} points")
    print(f"Saved to {args.output}")

    # Print per-env statistics
    env_names = ['corridor', 'convex_obs', 'pf_obs', 'dyna_obs', 'non_obs']
    for i, en in enumerate(env_names):
        env_count = sum(1 for e in all_data['env_type'] if e == i)
        if env_count > 0:
            pts = sum(len(all_data['d_pred'][j])
                      for j in range(len(all_data['env_type']))
                      if all_data['env_type'][j] == i)
            print(f"  {en}: {env_count} frames, {pts} points")


if __name__ == '__main__':
    main()
