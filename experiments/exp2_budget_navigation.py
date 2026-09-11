"""
Experiment 2: Risk Budget Allocation + Collision Threshold Fix.

Fixes:
  1. False positive collision stops: calibrate underestimation direction
     and adjust collision_threshold accordingly.
  2. Per-point risk budget: distance-weighted margin allocation.

Tests navigation in corridor with 3cm lidar noise, comparing:
  A. Vanilla (no calibration)
  B. Global CP (scalar q_hat margin, no collision fix)
  C. Global CP + collision threshold fix
  D. Distance-weighted CP + collision threshold fix
"""

import sys, os, json, time, yaml
import numpy as np
import torch
import cvxpy as cp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neupan.robot import robot
from neupan.blocks import ObsPointNet
from neupan.configuration import np_to_tensor, to_device, tensor_to_np
from neupan.risk_calibration import RiskCalibrator
from neupan.risk_calibration.risk_budget import RiskBudgetAllocator
from neupan import neupan
from irsim.env import EnvBase


def calibrate_both_directions(
    checkpoint='example/model/diff_robot_default/model_5000.pth',
    n_scenes=2000, points_per_scene=50, noise_std=0.03,
):
    """Calibrate both overestimation (risk) and underestimation (collision)."""
    r = robot(receding=10, step_time=0.1, kinematics='diff', length=1.6, width=2.0)
    G_np, h_np = r.G, r.h
    G_t, h_t = np_to_tensor(G_np), np_to_tensor(h_np)
    model = to_device(ObsPointNet(2, G_np.shape[0]))
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model.eval()

    scene_risks_asy = []  # max(0, d_pred - d_gt) — overestimation (danger)
    scene_risks_under = []  # max(0, d_gt - d_pred) — underestimation (causes false stop)
    scene_risks_sym = []

    print(f"Calibrating: {n_scenes} scenes, noise_std={noise_std*100:.0f}cm")
    start = time.time()
    for si in range(n_scenes):
        if si == 0:
            # Force flush print buffer
            sys.stdout.flush()
        rng = np.random.RandomState(si)
        clean = rng.uniform(-25, 25, (points_per_scene, 2))
        noisy = clean + np.random.normal(0, noise_std, clean.shape)

        dp_all, dg_all = [], []
        for c, n in zip(clean, noisy):
            # DUNE prediction on noisy point
            pt = np_to_tensor(n.reshape(2, 1))
            with torch.no_grad():
                mu = model(pt.T).T
            dp = float(torch.squeeze(mu.T @ (G_t @ pt - h_t)))

            # LP ground truth on clean point
            mu_var = cp.Variable((G_np.shape[0], 1), nonneg=True)
            p = cp.Parameter((2, 1))
            p.value = c.reshape(2, 1)
            prob = cp.Problem(
                cp.Maximize(mu_var.T @ (G_np @ p - h_np)),
                [cp.norm(G_np.T @ mu_var) <= 1]
            )
            prob.solve(solver=cp.ECOS)
            dg = prob.value if prob.value is not None else dp

            dp_all.append(dp)
            dg_all.append(dg)

        dp_arr = np.array(dp_all)
        dg_arr = np.array(dg_all)

        # Overestimation risk (d_pred > d_gt = dangerous)
        over = np.maximum(0, dp_arr - dg_arr)
        scene_risks_asy.append(float(np.max(over)))

        # Underestimation (d_gt > d_pred = false positive direction)
        under = np.maximum(0, dg_arr - dp_arr)
        scene_risks_under.append(float(np.max(under)))

        # Symmetric
        scene_risks_sym.append(float(np.max(np.abs(dp_arr - dg_arr))))

        if (si + 1) % 500 == 0:
            print(f"  {si+1}/{n_scenes} ({time.time()-start:.0f}s)")

    risks_asy = np.array(scene_risks_asy)
    risks_under = np.array(scene_risks_under)
    risks_sym = np.array(scene_risks_sym)

    results = {}
    for eps in [0.01, 0.05, 0.10]:
        # Overestimation q_hat (for d_min tightening)
        cal = RiskCalibrator()
        cal.fit(risks_asy)
        q_over = cal.compute_q_hat(eps)

        # Underestimation q_hat (for collision threshold adjustment)
        cal_u = RiskCalibrator()
        cal_u.fit(risks_under)
        q_under = cal_u.compute_q_hat(eps)

        # Symmetric
        N = len(risks_sym)
        level = min(np.ceil((1 - eps) * (N + 1)) / N, 1.0)
        q_sym = float(np.quantile(np.sort(risks_sym), level))

        results[str(eps)] = {
            'q_over': q_over,
            'q_under': q_under,
            'q_sym': q_sym,
            'under_mean_cm': float(np.mean(risks_under) * 100),
            'under_p95_cm': float(np.percentile(risks_under, 95) * 100),
        }
        print(f"  ε={eps}: q_over={q_over*100:.2f}cm, q_under={q_under*100:.2f}cm, "
              f"q_sym={q_sym*100:.2f}cm")

    print(f"  Underestimation: mean={np.mean(risks_under)*100:.2f}cm, "
          f"p95={np.percentile(risks_under,95)*100:.2f}cm, "
          f"max={np.max(risks_under)*100:.2f}cm")
    print(f"  Overestimation:  mean={np.mean(risks_asy)*100:.2f}cm, "
          f"p95={np.percentile(risks_asy,95)*100:.2f}cm")

    return results


def run_navigation(
    noise_std=0.03, q_over=0.0, q_under=0.0,
    budget_strategy='global', n_episodes=30, max_steps=500,
    env_file='example/corridor/diff/env.yaml',
    planner_file='example/corridor/diff/planner.yaml',
):
    """Run navigation episodes with risk calibration and collision fix."""
    env_cfg = yaml.safe_load(open(env_file))
    if noise_std > 0:
        for rob in env_cfg['robot']:
            for s in rob.get('sensors', []):
                s['noise'] = True
                s['std'] = noise_std

    plan_cfg = yaml.safe_load(open(planner_file))
    plan_cfg['time_print'] = False

    tmp_e = f'/tmp/env_{budget_strategy}_{noise_std}_{q_over:.3f}.yaml'
    tmp_p = f'/tmp/plan_{budget_strategy}_{noise_std}_{q_over:.3f}.yaml'
    yaml.dump(env_cfg, open(tmp_e, 'w'))
    yaml.dump(plan_cfg, open(tmp_p, 'w'))

    metrics = []
    for ep in range(n_episodes):
        env = EnvBase(tmp_e, display=False, save_ani=False)
        planner = neupan.init_from_yaml(tmp_p)

        # Apply risk calibration
        if q_over > 0:
            planner.enable_risk_calibration(
                q_hat=q_over,
                budget_strategy=budget_strategy,
                collision_q_hat=q_under,
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

            if planner.min_distance < min_dist:
                md = planner.min_distance
                min_dist = float(md.item() if hasattr(md, 'item') else md)

            action, info = planner(state, points, None)
            if action is not None:
                try:
                    spd = float(np.linalg.norm(action))
                except Exception:
                    spd = float(np.linalg.norm(action.detach().cpu().numpy()))
                speeds.append(spd)

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
    spds = [m['avg_speed'] for m in metrics if m['outcome'] == 'arrived']

    summary = {
        'collisions': coll, 'collision_rate': coll / n_episodes,
        'arrivals': arrv, 'success_rate': arrv / n_episodes,
        'avg_path': float(np.mean(paths)) if paths else 0,
        'avg_min_distance': float(np.mean(min_dists)),
        'min_min_distance': float(np.min(min_dists)),
        'avg_speed': float(np.mean(spds)) if spds else 0,
    }
    return summary, metrics


def main():
    print("=" * 70)
    print("EXPERIMENT 2: Risk Budget + Collision Threshold Fix")
    print("=" * 70)

    # Step 1: Calibrate both directions
    print("\n--- Step 1: Dual-direction calibration ---")
    cal = calibrate_both_directions(
        n_scenes=500, noise_std=0.03,
    )
    eps = '0.05'
    q_over = cal[eps]['q_over']   # for d_min tightening
    q_under = cal[eps]['q_under']  # for collision threshold adjustment

    print(f"\n  q_over (risk margin)   = {q_over*100:.2f}cm")
    print(f"  q_under (collision fix) = {q_under*100:.2f}cm")
    print(f"  collision_threshold     = 10.0cm")
    print(f"  effective_threshold     = {max(10.0 - q_under*100, 0):.2f}cm")

    # Step 2: Navigation comparison
    print(f"\n--- Step 2: Navigation comparison (3cm lidar noise) ---\n")

    methods = [
        ('vanilla', 0.0, 0.0, 'global'),
        ('global_cp_no_fix', q_over, 0.0, 'global'),
        ('global_cp_with_fix', q_over, q_under, 'global'),
        ('dist_weighted_with_fix', q_over, q_under, 'distance_weighted'),
    ]

    all_results = {}
    for name, qo, qu, strategy in methods:
        label = f"{name} (q_over={qo*100:.1f}cm, q_under={qu*100:.1f}cm)"
        print(f"--- {label} ---")
        summary, _ = run_navigation(
            noise_std=0.03, q_over=qo, q_under=qu,
            budget_strategy=strategy, n_episodes=30,
        )
        all_results[name] = {
            **summary,
            'q_over_cm': qo * 100,
            'q_under_cm': qu * 100,
            'budget_strategy': strategy,
        }
        print(f"  Coll: {summary['collision_rate']*100:.0f}%, "
              f"Succ: {summary['success_rate']*100:.0f}%, "
              f"MinDist: {summary['avg_min_distance']*100:.1f}cm "
              f"(worst: {summary['min_min_distance']*100:.1f}cm), "
              f"Path: {summary['avg_path']:.1f}m, "
              f"Speed: {summary['avg_speed']:.2f}m/s\n")

    # Final comparison table
    print("=" * 90)
    print("COMPARISON TABLE")
    print("=" * 90)
    hdr = (f"{'Method':>28} | {'q_over':>7} | {'q_under':>7} | "
           f"{'Coll':>5} | {'Succ':>5} | {'MinDist':>10} | {'Worst':>8} | "
           f"{'Path':>6} | {'Speed':>6}")
    print(hdr)
    print("-" * len(hdr))
    for m, r in all_results.items():
        print(f"{m:>28} | {r['q_over_cm']:>6.2f}cm | {r['q_under_cm']:>6.2f}cm | "
              f"{r['collision_rate']*100:>4.0f}% | {r['success_rate']*100:>4.0f}% | "
              f"{r['avg_min_distance']*100:>8.1f}cm | "
              f"{r['min_min_distance']*100:>6.1f}cm | "
              f"{r['avg_path']:>5.1f}m | {r['avg_speed']:>5.2f}m/s")

    # Save
    os.makedirs('experiments/exp2_output', exist_ok=True)
    output = {
        'calibration': cal,
        'navigation': all_results,
    }
    with open('experiments/exp2_output/exp2_risk_budget.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to experiments/exp2_output/exp2_risk_budget.json")


if __name__ == "__main__":
    main()
