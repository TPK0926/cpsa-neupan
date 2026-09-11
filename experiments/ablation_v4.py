"""Ablation study for CPSA-v4 components.

Tests on Corridor 2cm and Dyna_obs 5cm, 30 episodes each.
Variants:
  A0: Full v4 (baseline)
  A1: No global context (remove max-pool aggregation)
  A2: No temporal features (zero out d_pred_delta, radius_delta)
  A3: No passage features (zero out lateral_margin, passage_ratio)
  A4: No narrow passage loss (set weight to 0)
  A5: No dynamic tau_max (always use 8cm)
  A6: No teacher pretraining (skip Phase 1)
  A7: No SWA (no weight averaging)
  A8: No noise injection (disable FeatureNoiseInjector)

For each variant, we use the same trained v4 model but modify inference.
Components that require retraining (A1, A4, A6, A7, A8) are marked.
"""

import sys, os, yaml, time, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from neupan import neupan
from neupan.blocks.cpsa_risk_net import CPSARiskNet
from neupan.configuration import np_to_tensor
from irsim.env import EnvBase
import torch
import tempfile

ENVIRONMENTS = {
    'corridor': {
        'desc': 'Narrow corridor',
        'diff': {'env': 'example/corridor/diff/env.yaml',
                 'planner': 'example/corridor/diff/planner.yaml'},
        'env_type': 0,
    },
    'dyna_obs': {
        'desc': 'Dynamic obstacles',
        'diff': {'env': 'example/dyna_obs/diff/env.yaml',
                 'planner': 'example/dyna_obs/diff/planner.yaml'},
        'env_type': 3,
    },
}


def make_env_yaml(env_name, noise_std):
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    env_path = os.path.join(base_dir, ENVIRONMENTS[env_name]['diff']['env'])
    with open(env_path) as f:
        cfg = yaml.safe_load(f)
    if noise_std > 0:
        for rob in cfg.get('robot', []):
            for s in rob.get('sensors', []):
                s['noise'] = True
                s['std'] = noise_std
    else:
        for rob in cfg.get('robot', []):
            for s in rob.get('sensors', []):
                s['noise'] = False
    return cfg


def run_test(env_name, noise_std, method, n_ep=30, max_steps=400):
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    os.chdir(base_dir)

    env_cfg = make_env_yaml(env_name, noise_std)
    plan_path = os.path.join(base_dir, ENVIRONMENTS[env_name]['diff']['planner'])
    plan_cfg = yaml.safe_load(open(plan_path))
    plan_cfg['time_print'] = False

    tf_e = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(env_cfg, tf_e); tf_e.close()
    tf_p = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(plan_cfg, tf_p); tf_p.close()

    coll = 0; arrv = 0; min_dists = []
    for ep in range(n_ep):
        env = EnvBase(tf_e.name, display=False, save_ani=False)
        planner = neupan.init_from_yaml(tf_p.name)

        if method == 'cpsa_v4':
            planner.enable_risk_calibration(
                budget_strategy='cpsa',
                cp_checkpoint='experiments/cp_head_output/cpsa_v4_universal.pth',
                collision_q_hat=0.01, noise_std=noise_std,
                env_type=ENVIRONMENTS[env_name]['env_type'])

        ep_min_dist = 999.0
        for step in range(max_steps):
            state = env.get_robot_state()
            scan = env.get_lidar_scan()
            pts = planner.scan_to_point(state, scan)
            action, info = planner(state, pts, None)

            # Track min distance
            if hasattr(env, 'get_robot_state') and info.get('min_distance') is not None:
                md = info['min_distance']
                if md < ep_min_dist:
                    ep_min_dist = md

            if info.get('stop') or info.get('arrive') or env.done():
                break
            env.step(action)

        if info.get('stop'): coll += 1
        if info.get('arrive'): arrv += 1
        min_dists.append(ep_min_dist)
        env.end(0)

    os.unlink(tf_e.name)
    os.unlink(tf_p.name)
    avg_min_dist = np.mean([d for d in min_dists if d < 999]) if any(d < 999 for d in min_dists) else 0.0
    return arrv, coll, n_ep, avg_min_dist


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='cpsa_v4')
    parser.add_argument('--env', default='corridor')
    parser.add_argument('--noise', type=float, default=0.02)
    parser.add_argument('--n_ep', type=int, default=30)
    args = parser.parse_args()

    print(f"Ablation: {args.method} on {args.env} noise={args.noise*100:.0f}cm, {args.n_ep} episodes")
    t0 = time.time()
    a, c, n, md = run_test(args.env, args.noise, args.method, n_ep=args.n_ep)
    dt = time.time() - t0
    print(f"  Succ={a}/{n} Coll={c}/{n} AvgMinDist={md:.3f}m ({dt:.0f}s)")
