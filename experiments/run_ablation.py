"""
Ablation study runner for CPSA-v4 components.

Runs all CPSA variants on corridor 2cm and dyna_obs 5cm (30 episodes each).
Saves results to experiments/ablation_output/ablation_v4.json

Usage:
    python experiments/run_ablation.py
"""

import sys, os, json, time, yaml
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from neupan import neupan
from irsim.env import EnvBase

PYTHON = os.environ.get('PYTHON', sys.executable)

ENVIRONMENTS = {
    'corridor_2cm': {
        'env': 'corridor', 'noise': 0.02, 'env_type': 0,
        'env_file': 'example/corridor/diff/env.yaml',
        'plan_file': 'example/corridor/diff/planner.yaml',
    },
    'dyna_obs_5cm': {
        'env': 'dyna_obs', 'noise': 0.05, 'env_type': 3,
        'env_file': 'example/dyna_obs/diff/env.yaml',
        'plan_file': 'example/dyna_obs/diff/planner.yaml',
    },
}

# Ablation variants - inference-only changes (no retraining needed)
ABLATION_VARIANTS = {
    'A0_full': {
        'desc': 'Full CPSA v4',
        'cpsa_config': {},
    },
    'A1_no_global_ctx': {
        'desc': 'No global context (skip max-pool)',
        'cpsa_config': {'disable_global_ctx': True},
    },
    'A2_no_temporal': {
        'desc': 'No temporal features',
        'cpsa_config': {'disable_temporal': True},
    },
    'A3_no_passage': {
        'desc': 'No passage features',
        'cpsa_config': {'disable_passage': True},
    },
    'A4_no_dynamic_taumax': {
        'desc': 'Static tau_max=8cm (no dynamic)',
        'cpsa_config': {'static_tau_max': True},
    },
    'A5_no_dynamic_qstar': {
        'desc': 'Static q_star (no dynamic adjustment)',
        'cpsa_config': {'static_q_star': True},
    },
}

# Static CP baselines for comparison
STATIC_BASELINES = {
    'vanilla': {'desc': 'Vanilla (no CP)'},
    'cp_global': {'desc': 'CP-Global (static asym)'},
    'cp_dw': {'desc': 'CP-DW (distance-weighted)'},
}


def make_noisy_env(env_file, noise_std):
    with open(os.path.join(os.path.dirname(__file__), '..', env_file)) as f:
        cfg = yaml.safe_load(f.read())
    if noise_std > 0:
        for rob in cfg.get('robot', []):
            for s in rob.get('sensors', []):
                s['noise'] = True
                s['std'] = noise_std
    return cfg


def run_config(env_name, noise_std, env_cfg, plan_file, n_ep=30, max_steps=400,
               method='vanilla', cpsa_config=None, cp_checkpoint=None):
    """Run navigation episodes and return summary."""
    import tempfile
    base_dir = os.path.join(os.path.dirname(__file__), '..')

    plan_path = os.path.join(base_dir, plan_file)
    plan_cfg = yaml.safe_load(open(plan_path))
    plan_cfg['time_print'] = False

    tf_e = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(env_cfg, tf_e); tf_e.close()
    tf_p = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(plan_cfg, tf_p); tf_p.close()

    metrics = []
    for ep in range(n_ep):
        env = EnvBase(tf_e.name, display=False, save_ani=False)
        planner = neupan.init_from_yaml(tf_p.name)

        # Load calibration data
        import json
        cal_path = os.path.join(base_dir, 'experiments', 'calibration_output', 'calibration.json')
        with open(cal_path) as f:
            cal_data = json.load(f)

        if method == 'cpsa_v4' and cp_checkpoint:
            # Map noise_std to noise_cm for q_under lookup
            noise_cm = int(noise_std * 100)
            # Default q_under values from calibration
            q_under_map = {0: 1.37, 2: 6.44, 3: 9.44, 5: 15.14}
            collision_q = q_under_map.get(noise_cm, 6.44) / 100.0  # convert cm to m

            planner.enable_risk_calibration(
                budget_strategy='cpsa',
                cp_checkpoint=cp_checkpoint,
                collision_q_hat=collision_q,
                noise_std=noise_std,
                env_type=ENVIRONMENTS.get(f'{env_name}_{int(noise_std*100)}cm', {}).get('env_type', 0),
                cpsa_config=cpsa_config,
            )
        elif method == 'cp_global':
            # Static asymmetric CP
            cal_scores = np.array(cal_data['calibration_scores'])
            from neupan.risk_calibration import RiskCalibrator
            cal = RiskCalibrator()
            cal.fit(cal_scores)
            q_asy = cal.compute_q_hat(0.05)
            planner.enable_risk_calibration(
                q_hat=q_asy, budget_strategy='global',
                collision_q_hat=0.05)
        elif method == 'cp_dw':
            cal_scores = np.array(cal_data['calibration_scores'])
            from neupan.risk_calibration import RiskCalibrator
            cal = RiskCalibrator()
            cal.fit(cal_scores)
            q_asy = cal.compute_q_hat(0.05)
            planner.enable_risk_calibration(
                q_hat=q_asy, budget_strategy='distance_weighted',
                collision_q_hat=0.05)

        ep_min_dist = float('inf')
        outcome = 'timeout'
        for step in range(max_steps):
            state = env.get_robot_state()
            scan = env.get_lidar_scan()
            pts = planner.scan_to_point(state, scan)

            md_val = planner.min_distance
            try:
                md_val = float(md_val.item() if hasattr(md_val, 'item') else md_val)
            except:
                pass
            if md_val < ep_min_dist:
                ep_min_dist = md_val

            action, info = planner(state, pts, None)

            if info.get('stop'):
                outcome = 'collision'
                break
            if info.get('arrive'):
                outcome = 'arrived'
                break
            if env.done():
                outcome = 'done'
                break
            env.step(action)

        metrics.append({
            'outcome': outcome, 'steps': step + 1,
            'min_distance': ep_min_dist,
        })
        env.end(0)

    os.unlink(tf_e.name)
    os.unlink(tf_p.name)

    n = len(metrics)
    coll = sum(1 for m in metrics if m['outcome'] == 'collision')
    arrv = sum(1 for m in metrics if m['outcome'] == 'arrived')
    min_dists = [m['min_distance'] for m in metrics
                 if m['min_distance'] < 999]
    return {
        'collisions': coll, 'collision_rate': coll / n,
        'arrivals': arrv, 'success_rate': arrv / n,
        'avg_min_distance': float(np.mean(min_dists)) if min_dists else 0,
        'n_episodes': n,
    }


def main():
    print("=" * 70)
    print("CPSA v4 ABLATION STUDY")
    print("=" * 70)

    base_dir = os.path.join(os.path.dirname(__file__), '..')
    cp_ckpt = os.path.join(base_dir, 'experiments/cp_head_output/cpsa_v4_universal.pth')
    n_ep = 30

    all_results = {}

    for env_key, env_cfg in ENVIRONMENTS.items():
        print(f"\n{'='*60}")
        print(f"Environment: {env_key} (noise={env_cfg['noise']*100:.0f}cm)")
        print(f"{'='*60}")

        env_yaml = make_noisy_env(env_cfg['env_file'], env_cfg['noise'])

        # Run static baselines
        for method, cfg in STATIC_BASELINES.items():
            print(f"  {cfg['desc']}...", end=' ', flush=True)
            t0 = time.time()
            r = run_config(
                env_cfg['env'], env_cfg['noise'], env_yaml,
                env_cfg['plan_file'], n_ep=n_ep, method=method)
            key = f"{env_key}_{method}"
            all_results[key] = r
            print(f"SR={r['success_rate']*100:.0f}% CR={r['collision_rate']*100:.1f}% "
                  f"MinD={r['avg_min_distance']*100:.1f}cm ({time.time()-t0:.0f}s)")

        # Run CPSA variants
        for variant, cfg in ABLATION_VARIANTS.items():
            print(f"  {cfg['desc']}...", end=' ', flush=True)
            t0 = time.time()
            r = run_config(
                env_cfg['env'], env_cfg['noise'], env_yaml,
                env_cfg['plan_file'], n_ep=n_ep,
                method='cpsa_v4', cp_checkpoint=cp_ckpt,
                cpsa_config=cfg.get('cpsa_config'))
            key = f"{env_key}_{variant}"
            all_results[key] = r
            print(f"SR={r['success_rate']*100:.0f}% CR={r['collision_rate']*100:.1f}% "
                  f"MinD={r['avg_min_distance']*100:.1f}cm ({time.time()-t0:.0f}s)")

    # Summary
    print(f"\n{'='*80}")
    print("ABLATION SUMMARY")
    print(f"{'='*80}")
    for env_key in ENVIRONMENTS:
        print(f"\n--- {env_key} ---")
        print(f"{'Method':<30} {'SR %':>6} {'CR %':>6} {'MinD cm':>8}")
        for method in list(STATIC_BASELINES.keys()) + list(ABLATION_VARIANTS.keys()):
            key = f"{env_key}_{method}"
            if key in all_results:
                r = all_results[key]
                name = STATIC_BASELINES.get(method, ABLATION_VARIANTS.get(method, {}))
                desc = name.get('desc', method) if isinstance(name, dict) else method
                print(f"{desc:<30} {r['success_rate']*100:>5.1f}  {r['collision_rate']*100:>5.1f}  "
                      f"{r['avg_min_distance']*100:>7.1f}")

    # Save
    out_dir = os.path.join(base_dir, 'experiments', 'ablation_output')
    os.makedirs(out_dir, exist_ok=True)
    output = {
        'config': {
            'environments': list(ENVIRONMENTS.keys()),
            'n_episodes': n_ep,
            'cp_checkpoint': cp_ckpt,
        },
        'variants': {
            **{k: v['desc'] for k, v in ABLATION_VARIANTS.items()},
            **{k: v['desc'] for k, v in STATIC_BASELINES.items()},
        },
        'results': all_results,
    }
    out_path = os.path.join(out_dir, 'ablation_v4.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
