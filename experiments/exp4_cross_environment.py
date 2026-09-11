"""
Experiment 4: Cross-Environment Calibration Transfer + Multi-Environment Navigation.

Core experiments for Route 2 paper positioning:
  E3: Calibration transfer — calibrate once on random points, test in 3 environments
  E4: Multi-env navigation — vanilla vs CP methods across environments
  E5: In-the-loop vs post-hoc CP comparison

Environments: corridor, convex_obs, narrow_corridor
Noise levels: 0cm, 2cm
Methods: vanilla, global_cp, dist_weighted_cp, posthoc_cp (planning-level baseline)
"""
import sys, os, json, time, yaml, copy
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

# ---------------------------------------------------------------------------
# Environment configs (base_dir relative to NeuPAN-main)
# ---------------------------------------------------------------------------
ENVIRONMENTS = {
    'corridor': {
        'env': 'example/corridor/diff/env.yaml',
        'planner': 'example/corridor/diff/planner.yaml',
        'difficulty': 'medium',
        'clearance': '~3m',
    },
    'convex_obs': {
        'env': 'example/convex_obs/diff/env.yaml',
        'planner': 'example/convex_obs/diff/planner.yaml',
        'difficulty': 'medium',
        'clearance': '~4-6m',
    },
    'narrow_corridor': {
        'env': 'example/narrow_corridor/diff/env.yaml',
        'planner': 'example/narrow_corridor/diff/planner.yaml',
        'difficulty': 'hard',
        'clearance': '0.2m/side',
    },
}


def calibrate_for_noise(noise_std, n_scenes=500, points_per_scene=50,
                         checkpoint='example/model/diff_robot_default/model_5000.pth'):
    """Calibrate DUNE predictions: get q_over and q_under for given noise level."""
    r = robot(receding=10, step_time=0.1, kinematics='diff', length=1.6, width=2.0)
    G_np, h_np = r.G, r.h
    G_t, h_t = np_to_tensor(G_np), np_to_tensor(h_np)
    model = to_device(ObsPointNet(2, G_np.shape[0]))
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model.eval()

    scene_over, scene_under = [], []
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
            p = cp.Parameter((2, 1))
            p.value = c.reshape(2, 1)
            prob = cp.Problem(cp.Maximize(mu_var.T @ (G_np @ p - h_np)),
                              [cp.norm(G_np.T @ mu_var) <= 1])
            prob.solve(solver=cp.ECOS)
            dg = prob.value if prob.value is not None else dp
            dp_all.append(dp)
            dg_all.append(dg)

        dp_arr, dg_arr = np.array(dp_all), np.array(dg_all)
        scene_over.append(float(np.max(np.maximum(0, dp_arr - dg_arr))))
        scene_under.append(float(np.max(np.maximum(0, dg_arr - dp_arr))))

        if (si + 1) % 200 == 0:
            print(f"    Calibration {si+1}/{n_scenes} ({time.time()-start:.0f}s)")

    over, under = np.array(scene_over), np.array(scene_under)
    eps = 0.05
    cal_o = RiskCalibrator()
    cal_o.fit(over)
    q_over = cal_o.compute_q_hat(eps)
    cal_u = RiskCalibrator()
    cal_u.fit(under)
    q_under = cal_u.compute_q_hat(eps)

    print(f"    noise={noise_std*100:.0f}cm: q_over={q_over*100:.2f}cm, q_under={q_under*100:.2f}cm")
    return {'q_over': q_over, 'q_under': q_under,
            'over_mean_cm': float(np.mean(over)*100),
            'under_mean_cm': float(np.mean(under)*100)}


def make_env_yaml(env_name, noise_std):
    """Load env YAML and optionally add lidar noise (correct format)."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(base_dir, ENVIRONMENTS[env_name]['env'])
    with open(env_path) as f:
        cfg = yaml.safe_load(f)
    # Fix existing incorrect noise format and set desired noise
    if noise_std > 0:
        for rob in cfg.get('robot', []):
            for s in rob.get('sensors', []):
                s['noise'] = True
                s['std'] = noise_std
    else:
        # Ensure noise is off
        for rob in cfg.get('robot', []):
            for s in rob.get('sensors', []):
                if 'noise' in s:
                    # Remove noise entries cleanly
                    s.pop('noise', None)
                    s.pop('std', None)
                    s['noise'] = False
    return cfg


def make_planner_yaml(env_name, model_checkpoint=None):
    """Load planner YAML, optionally override model checkpoint."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plan_path = os.path.join(base_dir, ENVIRONMENTS[env_name]['planner'])
    with open(plan_path) as f:
        cfg = yaml.safe_load(f)
    cfg['time_print'] = False
    if model_checkpoint:
        cfg['pan']['dune_checkpoint'] = model_checkpoint
    # Ensure we use model_5000 for all environments
    if 'pan' in cfg and 'dune_checkpoint' in cfg['pan']:
        if 'model_500.pth' in cfg['pan']['dune_checkpoint']:
            cfg['pan']['dune_checkpoint'] = 'example/model/diff_robot_default/model_5000.pth'
    return cfg


class PosthocCPWrapper:
    """Planning-level CP baseline: apply margin to planned trajectory post-hoc.

    After NeuPAN computes an action, check if the minimum predicted distance
    along the trajectory minus q_hat is below threshold. If so, reduce speed
    proportionally or stop. This mimics CPSA's approach of calibrating planning
    outputs rather than embedding calibration in the optimizer.
    """
    def __init__(self, q_hat, collision_threshold=0.1):
        self.q_hat = q_hat
        self.collision_threshold = collision_threshold

    def filter_action(self, action, min_distance, info):
        """Post-hoc safety filter on the planned action.

        If the predicted min distance minus the CP margin is below collision
        threshold, scale down the action (conservative slowdown).
        """
        effective_min = min_distance - self.q_hat
        if effective_min < self.collision_threshold:
            # Conservative: scale action by how much margin we're missing
            if effective_min <= 0:
                return np.zeros_like(action), True  # full stop
            scale = effective_min / self.collision_threshold
            scale = max(0.0, min(1.0, scale))
            return action * scale * 0.3, False  # significant slowdown
        return action, False


def run_navigation(env_name, noise_std, q_over=0.0, q_under=0.0,
                   budget_strategy='global', method='cp_inloop',
                   n_episodes=30, max_steps=500):
    """Run navigation in a specific environment.

    Args:
        method: 'vanilla', 'cp_inloop' (our approach), or 'cp_posthoc' (CPSA-style)
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)  # IR-SIM needs relative paths from project root

    env_cfg = make_env_yaml(env_name, noise_std)
    plan_cfg = make_planner_yaml(env_name)

    tag = f"{env_name}_{method}_{noise_std:.2f}_{budget_strategy}"
    tmp_e = f'/tmp/exp4_env_{tag}.yaml'
    tmp_p = f'/tmp/exp4_plan_{tag}.yaml'
    yaml.dump(env_cfg, open(tmp_e, 'w'))
    yaml.dump(plan_cfg, open(tmp_p, 'w'))

    posthoc = None
    if method == 'cp_posthoc':
        posthoc = PosthocCPWrapper(q_hat=q_over, collision_threshold=0.1)

    metrics = []
    for ep in range(n_episodes):
        env = EnvBase(tmp_e, display=False, save_ani=False)
        planner = neupan.init_from_yaml(tmp_p)

        if method == 'cp_inloop':
            if q_over > 0:
                planner.enable_risk_calibration(
                    q_hat=q_over, budget_strategy=budget_strategy,
                    collision_q_hat=q_under,
                )
        elif method == 'cp_posthoc':
            # For posthoc: enable collision threshold fix only (underestimation)
            # but NOT the in-the-loop margin on NRMP
            if q_under > 0:
                planner._collision_threshold_adjustment = q_under

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

            # Always call planner — it handles None points (drives toward goal)
            action, info = planner(state, points, None)
            if action is not None:
                # Apply posthoc filter if applicable
                if posthoc is not None and method == 'cp_posthoc':
                    cur_min = planner.min_distance
                    cur_min_f = float(cur_min.item() if hasattr(cur_min, 'item') else cur_min)
                    action, force_stop = posthoc.filter_action(action, cur_min_f, info)
                    if force_stop:
                        info['stop'] = True
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

    return {
        'collisions': coll, 'collision_rate': coll / n_episodes,
        'arrivals': arrv, 'success_rate': arrv / n_episodes,
        'avg_path': float(np.mean(paths)) if paths else 0,
        'avg_min_distance': float(np.mean(min_dists)),
        'min_min_distance': float(np.min(min_dists)),
        'avg_speed': float(np.mean(spds)) if spds else 0,
    }


def main():
    print("=" * 80)
    print("EXPERIMENT 4: Cross-Environment Calibration Transfer")
    print("Route 2: Conformal Calibration in Differentiable Optimization-Based Navigation")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Step 1: Calibrate (environment-agnostic, on random points)
    # -----------------------------------------------------------------------
    print("\n--- Step 1: Environment-Agnostic Calibration ---")
    cal_data = {}
    for ns in [0.0, 0.02]:
        print(f"\nNoise = {ns*100:.0f}cm")
        cal_data[ns] = calibrate_for_noise(ns, n_scenes=500)

    # Save calibration results
    os.makedirs('experiments/exp4_output', exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 2: Cross-environment navigation
    # -----------------------------------------------------------------------
    envs = ['corridor', 'convex_obs', 'narrow_corridor']
    noise_levels = [0.0, 0.02]

    methods = [
        # (label, method, budget_strategy, use_q)
        ('vanilla', 'vanilla', 'global', False),
        ('cp_inloop_global', 'cp_inloop', 'global', True),
        ('cp_inloop_dw', 'cp_inloop', 'distance_weighted', True),
        ('cp_posthoc', 'cp_posthoc', 'global', True),
    ]

    all_results = {}
    for ns in noise_levels:
        q_over = cal_data[ns]['q_over']
        q_under = cal_data[ns]['q_under']
        print(f"\n{'='*80}")
        print(f"NOISE = {ns*100:.0f}cm | q_over={q_over*100:.2f}cm, q_under={q_under*100:.2f}cm")
        print(f"{'='*80}")

        for env_name in envs:
            print(f"\n  Environment: {env_name} ({ENVIRONMENTS[env_name]['difficulty']}, "
                  f"clearance: {ENVIRONMENTS[env_name]['clearance']})")

            for label, method, strategy, use_q in methods:
                qo = q_over if use_q else 0.0
                qu = q_under if use_q else 0.0
                key = f"{ns*100:.0f}cm_{env_name}_{label}"
                print(f"    {label}...", end=' ', flush=True)
                try:
                    r = run_navigation(
                        env_name, ns, q_over=qo, q_under=qu,
                        budget_strategy=strategy, method=method,
                        n_episodes=30,
                    )
                    all_results[key] = {**r, 'noise_cm': ns*100, 'env': env_name,
                                        'method': label, 'q_over_cm': qo*100,
                                        'q_under_cm': qu*100, 'budget': strategy}
                    print(f"Coll={r['collision_rate']*100:.0f}% "
                          f"Succ={r['success_rate']*100:.0f}% "
                          f"MinDist={r['avg_min_distance']*100:.1f}cm "
                          f"Path={r['avg_path']:.1f}m")
                except Exception as e:
                    print(f"ERROR: {e}")
                    all_results[key] = {'error': str(e), 'noise_cm': ns*100,
                                        'env': env_name, 'method': label}

    # -----------------------------------------------------------------------
    # Step 3: Summary tables
    # -----------------------------------------------------------------------
    print(f"\n{'='*120}")
    print("SUMMARY: CROSS-ENVIRONMENT RESULTS")
    print(f"{'='*120}")
    print(f"{'Noise':>6} | {'Env':>16} | {'Method':>16} | {'q_over':>7} | "
          f"{'Coll':>5} | {'Succ':>5} | {'MinDist':>8} | {'Worst':>7} | {'Path':>6} | {'Speed':>6}")
    print("-" * 120)
    for ns in noise_levels:
        for env_name in envs:
            for label, _, _, _ in methods:
                key = f"{ns*100:.0f}cm_{env_name}_{label}"
                if key in all_results and 'error' not in all_results[key]:
                    r = all_results[key]
                    print(f"{r['noise_cm']:>5.0f}cm | {env_name:>16} | {label:>16} | "
                          f"{r['q_over_cm']:>6.2f}cm | "
                          f"{r['collision_rate']*100:>4.0f}% | "
                          f"{r['success_rate']*100:>4.0f}% | "
                          f"{r['avg_min_distance']*100:>6.1f}cm | "
                          f"{r['min_min_distance']*100:>5.1f}cm | "
                          f"{r['avg_path']:>5.1f}m | {r['avg_speed']:>5.2f}m/s")
            print()

    # Key comparison: in-the-loop vs post-hoc
    print(f"\n{'='*80}")
    print("KEY COMPARISON: In-the-Loop CP vs Post-hoc CP (2cm noise)")
    print(f"{'='*80}")
    for env_name in envs:
        inloop_key = f"2cm_{env_name}_cp_inloop_global"
        posthoc_key = f"2cm_{env_name}_cp_posthoc"
        if inloop_key in all_results and posthoc_key in all_results:
            il = all_results[inloop_key]
            ph = all_results[posthoc_key]
            if 'error' not in il and 'error' not in ph:
                print(f"\n  {env_name}:")
                print(f"    In-the-loop: Coll={il['collision_rate']*100:.0f}%, "
                      f"Succ={il['success_rate']*100:.0f}%, "
                      f"MinDist={il['avg_min_distance']*100:.1f}cm, "
                      f"Path={il['avg_path']:.1f}m")
                print(f"    Post-hoc:    Coll={ph['collision_rate']*100:.0f}%, "
                      f"Succ={ph['success_rate']*100:.0f}%, "
                      f"MinDist={ph['avg_min_distance']*100:.1f}cm, "
                      f"Path={ph['avg_path']:.1f}m")

    # Save
    output = {
        'calibration': {str(k): v for k, v in cal_data.items()},
        'navigation': all_results,
    }
    with open('experiments/exp4_output/exp4_cross_environment.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to experiments/exp4_output/exp4_cross_environment.json")


if __name__ == "__main__":
    main()
