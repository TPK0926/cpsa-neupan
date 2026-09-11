"""
Experiment 3: Multi-noise-level navigation with collision fix + risk budget.

Tests: 0cm (sanity), 1cm, 2cm, 3cm lidar noise
Methods: vanilla, global CP + fix, dist_weighted CP + fix
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


def calibrate_noise(noise_std, n_scenes=500, points_per_scene=50,
                    checkpoint='example/model/diff_robot_default/model_5000.pth'):
    r = robot(receding=10, step_time=0.1, kinematics='diff', length=1.6, width=2.0)
    G_np, h_np = r.G, r.h
    G_t, h_t = np_to_tensor(G_np), np_to_tensor(h_np)
    model = to_device(ObsPointNet(2, G_np.shape[0]))
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model.eval()

    scene_over, scene_under, scene_sym = [], [], []
    start = time.time()
    for si in range(n_scenes):
        rng = np.random.RandomState(si)
        clean = rng.uniform(-25, 25, (points_per_scene, 2))
        noisy = clean + np.random.normal(0, noise_std, clean.shape) if noise_std > 0 else clean.copy()

        dp_all, dg_all = [], []
        for c, n in zip(clean, noisy):
            pt = np_to_tensor(n.reshape(2, 1))
            with torch.no_grad():
                mu = model(pt.T).T
            dp = float(torch.squeeze(mu.T @ (G_t @ pt - h_t)))
            mu_var = cp.Variable((G_np.shape[0], 1), nonneg=True)
            p = cp.Parameter((2, 1)); p.value = c.reshape(2, 1)
            prob = cp.Problem(cp.Maximize(mu_var.T @ (G_np @ p - h_np)), [cp.norm(G_np.T @ mu_var) <= 1])
            prob.solve(solver=cp.ECOS)
            dg = prob.value if prob.value is not None else dp
            dp_all.append(dp); dg_all.append(dg)

        dp_arr, dg_arr = np.array(dp_all), np.array(dg_all)
        scene_over.append(float(np.max(np.maximum(0, dp_arr - dg_arr))))
        scene_under.append(float(np.max(np.maximum(0, dg_arr - dp_arr))))
        scene_sym.append(float(np.max(np.abs(dp_arr - dg_arr))))

        if (si + 1) % 200 == 0:
            print(f"  Calibration {si+1}/{n_scenes} ({time.time()-start:.0f}s)")

    over, under, sym = np.array(scene_over), np.array(scene_under), np.array(scene_sym)
    eps = 0.05
    cal_o = RiskCalibrator(); cal_o.fit(over)
    q_over = cal_o.compute_q_hat(eps)
    cal_u = RiskCalibrator(); cal_u.fit(under)
    q_under = cal_u.compute_q_hat(eps)
    N = len(sym)
    level = min(np.ceil((1 - eps) * (N + 1)) / N, 1.0)
    q_sym = float(np.quantile(np.sort(sym), level))

    print(f"  noise={noise_std*100:.0f}cm: q_over={q_over*100:.2f}cm, q_under={q_under*100:.2f}cm, "
          f"q_sym={q_sym*100:.2f}cm, eff_threshold={max(10-q_under*100,0):.2f}cm")
    return {'q_over': q_over, 'q_under': q_under, 'q_sym': q_sym,
            'over_mean': float(np.mean(over)*100), 'under_mean': float(np.mean(under)*100)}


def run_nav(noise_std, q_over=0.0, q_under=0.0, budget_strategy='global',
            n_episodes=30, max_steps=500):
    with open('example/corridor/diff/env.yaml') as f:
        env_yaml = f.read()
    env_cfg = yaml.safe_load(env_yaml)
    # IR-SIM lidar noise: noise=True, std=std. [0, std] is WRONG (sets noise=[list], std=default 0.2)
    if noise_std > 0:
        for rob in env_cfg['robot']:
            for s in rob.get('sensors', []):
                s['noise'] = True
                s['std'] = noise_std
    plan_cfg = yaml.safe_load(open('example/corridor/diff/planner.yaml'))
    plan_cfg['time_print'] = False

    tag = f"{budget_strategy}_{noise_std:.2f}"
    tmp_e = f'/tmp/env_{tag}.yaml'
    tmp_p = f'/tmp/plan_{tag}.yaml'
    yaml.dump(env_cfg, open(tmp_e, 'w'))
    yaml.dump(plan_cfg, open(tmp_p, 'w'))

    metrics = []
    for ep in range(n_episodes):
        env = EnvBase(tmp_e, display=False, save_ani=False)
        planner = neupan.init_from_yaml(tmp_p)
        if q_over > 0:
            planner.enable_risk_calibration(q_hat=q_over, budget_strategy=budget_strategy,
                                            collision_q_hat=q_under)
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
                try: spd = float(np.linalg.norm(action))
                except: spd = float(np.linalg.norm(action.detach().cpu().numpy()))
                speeds.append(spd)
            env.step(action)
            cur = state.flatten()[:2]
            path_len += np.linalg.norm(cur - prev)
            prev = cur
            if info.get('stop'): outcome = 'collision'; break
            if info.get('arrive'): outcome = 'arrived'; break
            if env.done(): outcome = 'done'; break
        env.end(0)
        metrics.append({'outcome': outcome, 'steps': step+1, 'path_length': path_len,
                        'min_distance': min_dist, 'avg_speed': float(np.mean(speeds)) if speeds else 0})

    coll = sum(1 for m in metrics if m['outcome']=='collision')
    arrv = sum(1 for m in metrics if m['outcome']=='arrived')
    min_dists = [m['min_distance'] for m in metrics]
    paths = [m['path_length'] for m in metrics if m['outcome']=='arrived']
    spds = [m['avg_speed'] for m in metrics if m['outcome']=='arrived']
    return {
        'collisions': coll, 'collision_rate': coll/n_episodes,
        'arrivals': arrv, 'success_rate': arrv/n_episodes,
        'avg_path': float(np.mean(paths)) if paths else 0,
        'avg_min_distance': float(np.mean(min_dists)),
        'min_min_distance': float(np.min(min_dists)),
        'avg_speed': float(np.mean(spds)) if spds else 0,
    }


def main():
    print("=" * 70)
    print("EXPERIMENT 3: Multi-Noise Navigation Comparison")
    print("=" * 70)

    noise_levels = [0.0, 0.01, 0.02, 0.03]

    # Step 1: Calibrate each noise level
    print("\n--- Step 1: Calibration ---")
    cal_data = {}
    for ns in noise_levels:
        print(f"\nNoise = {ns*100:.0f}cm")
        cal_data[ns] = calibrate_noise(ns, n_scenes=500)

    # Step 2: Navigation
    print("\n\n--- Step 2: Navigation ---")
    all_results = {}
    for ns in noise_levels:
        q_over = cal_data[ns]['q_over']
        q_under = cal_data[ns]['q_under']
        print(f"\n{'='*60}")
        print(f"Noise = {ns*100:.0f}cm | q_over={q_over*100:.2f}cm, q_under={q_under*100:.2f}cm")
        print(f"{'='*60}")

        for method, strategy in [('vanilla', 'global'), ('global_fix', 'global'), ('dw_fix', 'distance_weighted')]:
            qo = 0 if method == 'vanilla' else q_over
            qu = 0 if method == 'vanilla' else q_under
            print(f"  {method}...", end=' ', flush=True)
            r = run_nav(ns, q_over=qo, q_under=qu, budget_strategy=strategy, n_episodes=30)
            key = f"{ns*100:.0f}cm_{method}"
            all_results[key] = {**r, 'noise_cm': ns*100, 'method': method,
                                'q_over_cm': qo*100, 'q_under_cm': qu*100}
            print(f"Coll={r['collision_rate']*100:.0f}% Succ={r['success_rate']*100:.0f}% "
                  f"MinDist={r['avg_min_distance']*100:.1f}cm Path={r['avg_path']:.1f}m")

    # Summary table
    print(f"\n{'='*90}")
    print("SUMMARY TABLE")
    print(f"{'='*90}")
    print(f"{'Noise':>6} | {'Method':>10} | {'q_over':>7} | {'q_under':>7} | {'Coll':>5} | "
          f"{'Succ':>5} | {'MinDist':>8} | {'Worst':>7} | {'Path':>6}")
    print("-" * 90)
    for ns in noise_levels:
        for method in ['vanilla', 'global_fix', 'dw_fix']:
            key = f"{ns*100:.0f}cm_{method}"
            r = all_results[key]
            print(f"{r['noise_cm']:>5.0f}cm | {method:>10} | {r['q_over_cm']:>6.2f}cm | "
                  f"{r['q_under_cm']:>6.2f}cm | {r['collision_rate']*100:>4.0f}% | "
                  f"{r['success_rate']*100:>4.0f}% | {r['avg_min_distance']*100:>6.1f}cm | "
                  f"{r['min_min_distance']*100:>5.1f}cm | {r['avg_path']:>5.1f}m")
        print()

    # Save
    os.makedirs('experiments/exp3_output', exist_ok=True)
    output = {'calibration': {str(k): v for k, v in cal_data.items()}, 'navigation': all_results}
    with open('experiments/exp3_output/exp3_multi_noise.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to experiments/exp3_output/exp3_multi_noise.json")


if __name__ == "__main__":
    main()
