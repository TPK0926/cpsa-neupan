"""
Full-matrix experiment with CPSA v4 ablation variants.

Methods (9):
  vanilla, cp_global, cp_dw, cpsa_v3, cpsa_v4_full,
  cpsa_v4_noTemporal, cpsa_v4_noPassage,
  cpsa_v4_staticTau, cpsa_v4_staticQstar

Environments (5): corridor, convex_obs, pf_obs, dyna_obs, non_obs
Noise (4): 0, 2, 3, 5 cm
Episodes: 30 per config

Total: 9 × 5 × 4 × 30 = 5400 episodes. Est. 50-60 hours on RTX 4090.

Saves incrementally to experiments/exp_full_matrix_output/results.json
"""

import sys, os, json, time, yaml
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neupan import neupan
from neupan.risk_calibration import RiskCalibrator
from irsim.env import EnvBase

PYTHON = os.environ.get('PYTHON', sys.executable)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE_DIR, '..')
OUT_DIR = os.path.join(BASE_DIR, 'exp_full_matrix_output')


# ── Config ──────────────────────────────────────────────────────────

ENVIRONMENTS = {
    'corridor':    {'env': 'example/corridor/diff/env.yaml',    'plan': 'example/corridor/diff/planner.yaml',    'env_type': 0},
    'convex_obs':  {'env': 'example/convex_obs/diff/env.yaml',  'plan': 'example/convex_obs/diff/planner.yaml',  'env_type': 1},
    'pf_obs':      {'env': 'example/pf_obs/diff/env.yaml',      'plan': 'example/pf_obs/diff/planner.yaml',      'env_type': 2},
    'dyna_obs':    {'env': 'example/dyna_obs/diff/env.yaml',    'plan': 'example/dyna_obs/diff/planner.yaml',    'env_type': 3},
    'non_obs':     {'env': 'example/non_obs/diff/env.yaml',     'plan': 'example/non_obs/diff/planner.yaml',     'env_type': 4},
}

NOISE_LEVELS = [0.0, 0.02, 0.03, 0.05]  # meters

# Methods: (key, display_name, method_type, extra_config)
# method_type: 'static' | 'cpsa_v3' | 'cpsa_v4'
# extra_config: cpsa_config dict or None
METHODS = [
    ('vanilla',            'Vanilla',             'static', None),
    ('cp_global',          'CP-Global',           'static', None),
    ('cp_dw',              'CP-DW',               'static', None),
    ('cpsa_v4_full',        'CPSA-v4 (full)',       'cpsa_v4', {}),
    ('cpsa_v4_noTemporal',  'CPSA-v4 (-temporal)',  'cpsa_v4', {'disable_temporal': True}),
    ('cpsa_v4_noPassage',   'CPSA-v4 (-passage)',   'cpsa_v4', {'disable_passage': True}),
    ('cpsa_v4_staticTau',   'CPSA-v4 (-dyn tau)',   'cpsa_v4', {'static_tau_max': True}),
    ('cpsa_v4_staticQstar', 'CPSA-v4 (-dyn q*)',    'cpsa_v4', {'static_q_star': True}),
]

N_EPISODES = 30
MAX_STEPS = 500

# CPSA checkpoint paths
CPSA_V3_CKPT = 'experiments/cp_head_output/cpsa_v3.pth'  # may not exist
CPSA_V4_CKPT = 'experiments/cp_head_output/cpsa_v4_universal.pth'

# Noise-specific q_under from calibration (cm → m)
Q_UNDER_MAP = {0: 0.0137, 2: 0.0644, 3: 0.0944, 5: 0.1514}


# ── Helpers ─────────────────────────────────────────────────────────

def load_calibration():
    cal_path = os.path.join(BASE_DIR, 'calibration_output', 'calibration.json')
    with open(cal_path) as f:
        data = json.load(f)
    cal = RiskCalibrator()
    cal.fit(np.array(data['calibration_scores']))
    return cal.compute_q_hat(0.05)


def make_env_config(env_name, noise_std):
    cfg_path = os.path.join(ROOT, ENVIRONMENTS[env_name]['env'])
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f.read())
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


def load_results():
    """Load existing partial results for incremental saving."""
    os.makedirs(OUT_DIR, exist_ok=True)
    result_path = os.path.join(OUT_DIR, 'results.json')
    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)
    return {}


def save_results(data):
    os.makedirs(OUT_DIR, exist_ok=True)
    result_path = os.path.join(OUT_DIR, 'results.json')
    with open(result_path, 'w') as f:
        json.dump(data, f, indent=2)


# ── Episode Runner ───────────────────────────────────────────────────

def run_one_config(env_name, noise_std, method_key, method_type, cpsa_config,
                   q_asy, n_ep=N_EPISODES, max_steps=MAX_STEPS):
    """Run one (env, noise, method) configuration. Returns metrics dict."""
    env_cfg = make_env_config(env_name, noise_std)
    plan_path = os.path.join(ROOT, ENVIRONMENTS[env_name]['plan'])
    with open(plan_path) as f:
        plan_cfg = yaml.safe_load(f.read())
    plan_cfg['time_print'] = False

    import tempfile
    tf_e = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(env_cfg, tf_e); tf_e.close()
    tf_p = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(plan_cfg, tf_p); tf_p.close()

    noise_cm = int(noise_std * 100)
    collision_q = Q_UNDER_MAP.get(noise_cm, 0.05)

    metrics = []
    for ep in range(n_ep):
        try:
            env = EnvBase(tf_e.name, display=False, save_ani=False)
            planner = neupan.init_from_yaml(tf_p.name)

            # Configure method
            if method_type == 'static':
                if method_key == 'cp_global':
                    planner.enable_risk_calibration(
                        q_hat=q_asy, budget_strategy='global',
                        collision_q_hat=collision_q)
                elif method_key == 'cp_dw':
                    planner.enable_risk_calibration(
                        q_hat=q_asy, budget_strategy='distance_weighted',
                        collision_q_hat=collision_q)
            elif method_type == 'cpsa_v3':
                # v3 checkpoint
                v3_ckpt = os.path.join(ROOT, CPSA_V3_CKPT)
                if not os.path.exists(v3_ckpt):
                    raise FileNotFoundError(f"CPSA v3 checkpoint not found: {v3_ckpt}")
                planner.enable_risk_calibration(
                    budget_strategy='cpsa',
                    cp_checkpoint=v3_ckpt,
                    collision_q_hat=collision_q,
                    noise_std=noise_std,
                    env_type=ENVIRONMENTS[env_name]['env_type'])
            elif method_type == 'cpsa_v4':
                v4_ckpt = os.path.join(ROOT, CPSA_V4_CKPT)
                if not os.path.exists(v4_ckpt):
                    raise FileNotFoundError(f"CPSA v4 checkpoint not found: {v4_ckpt}")
                planner.enable_risk_calibration(
                    budget_strategy='cpsa',
                    cp_checkpoint=v4_ckpt,
                    collision_q_hat=collision_q,
                    noise_std=noise_std,
                    env_type=ENVIRONMENTS[env_name]['env_type'],
                    cpsa_config=cpsa_config or {})

            ep_min_dist = float('inf')
            prev = env.get_robot_state().flatten()[:2]
            path_len = 0.0
            speeds = []
            outcome = 'timeout'
            steps_taken = 0

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

                if action is not None:
                    try:
                        spd = float(np.linalg.norm(action))
                    except:
                        try:
                            spd = float(np.linalg.norm(action.detach().cpu().numpy()))
                        except:
                            spd = 0.0
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
                steps_taken = step + 1

            env.end(0)
            metrics.append({
                'outcome': outcome, 'steps': steps_taken + 1,
                'path_length': float(path_len), 'min_distance': float(ep_min_dist),
                'avg_speed': float(np.mean(speeds)) if speeds else 0,
            })

        except Exception as e:
            print(f"  ERROR ep {ep}: {e}", flush=True)
            try:
                env.end(0)
            except:
                pass
            metrics.append({
                'outcome': 'error', 'steps': 0,
                'path_length': 0, 'min_distance': float('inf'),
                'avg_speed': 0, 'error': str(e)[:100],
            })

    os.unlink(tf_e.name)
    os.unlink(tf_p.name)

    # Summarize
    valid = [m for m in metrics if m['outcome'] != 'error']
    n = len(valid)
    n_err = len(metrics) - n
    coll = sum(1 for m in valid if m['outcome'] == 'collision')
    arrv = sum(1 for m in valid if m['outcome'] == 'arrived')
    min_dists = [m['min_distance'] for m in valid
                 if not (np.isnan(m['min_distance']) or np.isinf(m['min_distance']))]
    paths = [m['path_length'] for m in valid if m['outcome'] == 'arrived']
    spds = [m['avg_speed'] for m in valid if m['outcome'] == 'arrived']

    return {
        'collisions': coll, 'collision_rate': coll / max(n, 1),
        'arrivals': arrv, 'success_rate': arrv / max(n, 1),
        'avg_path': float(np.mean(paths)) if paths else 0,
        'avg_min_distance': float(np.mean(min_dists)) if min_dists else 0,
        'min_min_distance': float(np.min(min_dists)) if min_dists else 0,
        'avg_speed': float(np.mean(spds)) if spds else 0,
        'n_valid': n, 'n_errors': n_err,
        'n_episodes': n_ep,
    }


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("FULL MATRIX EXPERIMENT + CPSA ABLATION")
    print(f"Methods: {len(METHODS)} | Envs: {len(ENVIRONMENTS)} | Noise: {len(NOISE_LEVELS)}")
    print(f"Total configs: {len(METHODS) * len(ENVIRONMENTS) * len(NOISE_LEVELS)}")
    print(f"Episodes: {len(METHODS) * len(ENVIRONMENTS) * len(NOISE_LEVELS) * N_EPISODES}")
    print("=" * 70)

    # Load calibration
    q_asy = load_calibration()
    print(f"\nq_asym(0.05) = {q_asy*100:.2f}cm")

    # Load existing results for incremental resume
    all_results = load_results()

    # Check which CPSA checkpoints exist
    v3_ckpt = os.path.join(ROOT, CPSA_V3_CKPT)
    v4_ckpt = os.path.join(ROOT, CPSA_V4_CKPT)
    print(f"CPSA v3 checkpoint: {'EXISTS' if os.path.exists(v3_ckpt) else 'NOT FOUND (will skip)'}")
    print(f"CPSA v4 checkpoint: {'EXISTS' if os.path.exists(v4_ckpt) else 'NOT FOUND (will skip)'}")

    total_configs = len(METHODS) * len(ENVIRONMENTS) * len(NOISE_LEVELS)
    completed = 0
    t_start = time.time()

    for method_key, method_name, method_type, cpsa_config in METHODS:
        # Skip v3 if checkpoint missing
        if method_type == 'cpsa_v3' and not os.path.exists(v3_ckpt):
            print(f"\nSKIP {method_name}: v3 checkpoint not found")
            continue
        if method_type == 'cpsa_v4' and not os.path.exists(v4_ckpt):
            print(f"\nSKIP {method_name}: v4 checkpoint not found")
            continue

        for env_name, env_info in ENVIRONMENTS.items():
            for noise_std in NOISE_LEVELS:
                config_key = f"{env_name}_{int(noise_std*100)}cm_{method_key}"
                completed += 1

                # Skip already completed configs
                if config_key in all_results:
                    r = all_results[config_key]
                    print(f"[{completed}/{total_configs}] SKIP {config_key} "
                          f"(done: SR={r.get('success_rate',0)*100:.0f}%)", flush=True)
                    continue

                elapsed = time.time() - t_start
                eta = (elapsed / max(completed - len(all_results), 1)) * (total_configs - completed) if completed > len(all_results) else 0
                print(f"[{completed}/{total_configs}] {config_key} "
                      f"(elapsed={elapsed/3600:.1f}h ETA={eta/3600:.1f}h)...",
                      end=' ', flush=True)

                t0 = time.time()
                try:
                    r = run_one_config(env_name, noise_std, method_key,
                                      method_type, cpsa_config, q_asy)
                    dt = time.time() - t0
                    all_results[config_key] = r
                    save_results(all_results)
                    print(f"SR={r['success_rate']*100:.0f}% "
                          f"CR={r['collision_rate']*100:.1f}% "
                          f"MinD={r['avg_min_distance']*100:.1f}cm "
                          f"({dt:.0f}s)", flush=True)
                except Exception as e:
                    print(f"FAILED: {e}", flush=True)
                    all_results[config_key] = {'error': str(e)[:200]}
                    save_results(all_results)
                    import traceback
                    traceback.print_exc()

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"DONE. {len(all_results)} configs completed in {elapsed/3600:.1f}h")
    print(f"Results saved to {OUT_DIR}/results.json")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
