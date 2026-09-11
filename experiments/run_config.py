"""
Run a single (env, noise, method) configuration and print JSON result line.
Usage: python experiments/run_config.py <env> <noise_cm> <method>

Methods:
  Static:        vanilla, cp_global, cp_dw
  CPSA v4:        cpsa_v4_full, cpsa_v4_noTemporal, cpsa_v4_noPassage,
                 cpsa_v4_staticTau, cpsa_v4_staticQstar
  Baselines:     input_perturbation, cbf, aci
"""

import sys, os, json, time, yaml, tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neupan import neupan
from neupan.risk_calibration import RiskCalibrator
from neupan.baselines.input_perturbation import InputPerturbationWrapper
from neupan.baselines.cbf_filter import CBFSafetyFilter
from neupan.baselines.aci import AdaptiveConformalInference
from irsim.env import EnvBase

ENV_MAP = {
    'corridor':       ('example/corridor/diff/env.yaml',       'example/corridor/diff/planner.yaml',       0),
    'convex_obs':     ('example/convex_obs/diff/env.yaml',     'example/convex_obs/diff/planner.yaml',     1),
    'pf_obs':         ('example/pf_obs/diff/env.yaml',         'example/pf_obs/diff/planner.yaml',         2),
    'dyna_obs':       ('example/dyna_obs/diff/env.yaml',       'example/dyna_obs/diff/planner.yaml',       3),
    'non_obs':        ('example/non_obs/diff/env.yaml',        'example/non_obs/diff/planner.yaml',        4),
    'mixed_corridor': ('example/mixed_corridor/diff/env.yaml', 'example/mixed_corridor/diff/planner.yaml', 5),
}

CPSA_METHODS = {
    'cpsa_v4_full':         {},
    'cpsa_v4_noTemporal':   {'disable_temporal': True},
    'cpsa_v4_noPassage':    {'disable_passage': True},
    'cpsa_v4_noGlobalCtx':  {'disable_global_ctx': True},
    'cpsa_v4_staticTau':    {'static_tau_max': True},
    'cpsa_v4_staticQstar':  {'static_q_star': True},
    'cpsa_v4_online':       {'online': True},
}

BASELINE_METHODS = {'input_perturbation', 'cbf', 'aci'}

CKPT_V4 = 'experiments/cp_head_output/cpsa_v4_universal.pth'
N_EP = 30
MAX_STEPS = 500


def main():
    if len(sys.argv) < 4:
        print("Usage: run_config.py <env> <noise_cm> <method> [n_episodes]")
        sys.exit(1)

    env_name = sys.argv[1]
    noise_cm = int(sys.argv[2])
    method = sys.argv[3]
    n_ep = int(sys.argv[4]) if len(sys.argv) >= 5 else N_EP
    noise_std = noise_cm / 100.0

    base_dir = os.path.join(os.path.dirname(__file__), '..')
    env_file, plan_file, env_type = ENV_MAP[env_name]

    # Load calibration
    cal_path = os.path.join(base_dir, 'experiments', 'calibration_output', 'calibration.json')
    with open(cal_path) as f:
        cal_data = json.load(f)
    cal = RiskCalibrator()
    cal.fit(np.array(cal_data['calibration_scores']))
    q_asy = cal.compute_q_hat(0.05)

    # Load env config
    with open(os.path.join(base_dir, env_file)) as f:
        env_cfg = yaml.safe_load(f.read())
    if noise_std > 0:
        for rob in env_cfg.get('robot', []):
            for s in rob.get('sensors', []):
                s['noise'] = True
                s['std'] = noise_std
    else:
        for rob in env_cfg.get('robot', []):
            for s in rob.get('sensors', []):
                s['noise'] = False

    # Load plan config
    with open(os.path.join(base_dir, plan_file)) as f:
        plan_cfg = yaml.safe_load(f.read())
    plan_cfg['time_print'] = False

    # Write temp files
    tf_e = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(env_cfg, tf_e); tf_e.close()
    tf_p = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(plan_cfg, tf_p); tf_p.close()

    coll = 0; arrv = 0
    min_dists = []; paths = []; speeds = []
    episodes = []  # per-episode data for distribution plots

    # For online methods, create planner once to preserve adapter state
    is_online = method in CPSA_METHODS and CPSA_METHODS[method].get('online')
    if is_online:
        planner = neupan.init_from_yaml(tf_p.name)
        ckpt = os.path.join(base_dir, CKPT_V4)
        cpsa_cfg = CPSA_METHODS[method]
        planner.enable_risk_calibration(
            budget_strategy='cpsa', cp_checkpoint=ckpt,
            q_hat=q_asy, noise_std=noise_std,
            env_type=env_type, cpsa_config=cpsa_cfg)
        planner.enable_cpsa_online(lr=1e-4, buffer_size=2000)

    for ep in range(n_ep):
        try:
            env = EnvBase(tf_e.name, display=False, save_ani=False)

            if is_online:
                planner.reset()
            else:
                planner = neupan.init_from_yaml(tf_p.name)

                if method == 'cp_global':
                    planner.enable_risk_calibration(
                        q_hat=q_asy, budget_strategy='global')
                elif method == 'cp_dw':
                    planner.enable_risk_calibration(
                        q_hat=q_asy, budget_strategy='distance_weighted')
                elif method in CPSA_METHODS:
                    ckpt = os.path.join(base_dir, CKPT_V4)
                    cpsa_cfg = CPSA_METHODS[method]
                    planner.enable_risk_calibration(
                        budget_strategy='cpsa', cp_checkpoint=ckpt,
                        q_hat=q_asy, noise_std=noise_std,
                        env_type=env_type, cpsa_config=cpsa_cfg)

            # Baseline initialization
            ip_wrapper = None
            cbf_filter = None
            aci = None
            if method == 'input_perturbation':
                ip_wrapper = InputPerturbationWrapper(
                    n_samples=10, noise_std=0.01, safety_factor=2.0)
            elif method == 'cbf':
                cbf_filter = CBFSafetyFilter(d_safe=0.05, alpha=10.0, eta=0.3)
            elif method == 'aci':
                aci = AdaptiveConformalInference(
                    epsilon=0.05, gamma=0.005, q_hat_initial=q_asy)
                aci.set_calibration(np.array(cal_data['calibration_scores']))

            ep_min_dist = float('inf')
            prev = env.get_robot_state().flatten()[:2]
            path_len = 0.0
            ep_speeds = []
            outcome = 'timeout'
            trajectory = []

            for step in range(MAX_STEPS):
                state = env.get_robot_state()
                trajectory.append(state.flatten()[:3].tolist())
                scan = env.get_lidar_scan()
                pts = planner.scan_to_point(state, scan)

                # --- Input Perturbation: set per-step margin ---
                if method == 'input_perturbation' and ip_wrapper is not None:
                    pts_t = [np_to_tensor(p) if not isinstance(p, torch.Tensor) else p
                             for p in (pts if isinstance(pts, list) else [pts])]
                    R_t = [torch.eye(2) for _ in pts_t]
                    with torch.no_grad():
                        margin = ip_wrapper.estimate(pts_t, R_t, pts_t, planner.pan.dune_layer)
                    planner.pan.nrmp_layer.set_risk_margin(margin)

                # --- ACI: set dynamic q_hat ---
                if method == 'aci' and aci is not None:
                    planner.pan.nrmp_layer.set_risk_margin(aci.get_q_hat())

                md = planner.min_distance
                try:
                    md = float(md.item() if hasattr(md, 'item') else md)
                except:
                    pass
                if md < ep_min_dist:
                    ep_min_dist = md

                action, info = planner(state, pts, None)

                # --- CBF: post-hoc action filtering ---
                if method == 'cbf' and cbf_filter is not None and action is not None:
                    try:
                        if hasattr(planner.pan.dune_layer, 'distances_0'):
                            d_pred = planner.pan.dune_layer.distances_0.cpu().numpy()
                        else:
                            d_pred = np.array([])
                        if isinstance(pts, list) and len(pts) > 0:
                            pts_arr = pts[0]
                        elif pts is not None:
                            pts_arr = pts
                        else:
                            pts_arr = None
                        if pts_arr is not None:
                            if isinstance(pts_arr, torch.Tensor):
                                pts_arr = pts_arr.cpu().numpy()
                            if hasattr(pts_arr, 'ndim') and pts_arr.ndim == 2 and pts_arr.shape[0] >= 2 and len(d_pred) > 0:
                                phi = np.arctan2(pts_arr[1, :], pts_arr[0, :])
                            else:
                                phi = np.zeros(len(d_pred))
                        else:
                            phi = np.zeros(len(d_pred))
                        act_np = action.detach().cpu().numpy().flatten() if hasattr(action, 'detach') else np.array(action).flatten()
                        action = cbf_filter.filter_action(act_np, d_pred, phi)
                    except Exception:
                        pass  # CBF filter fails safe — leave action unchanged

                # --- ACI: track violation ---
                aci_violation = False
                if method == 'aci' and aci is not None:
                    if md < aci.get_q_hat() + 0.02:
                        aci_violation = True

                if action is not None:
                    try:
                        spd = float(np.linalg.norm(action))
                    except:
                        try: spd = float(np.linalg.norm(action.detach().cpu().numpy()))
                        except: spd = 0.0
                    ep_speeds.append(spd)

                env.step(action)
                cur = state.flatten()[:2]
                path_len += np.linalg.norm(cur - prev)
                prev = cur

                if info.get('stop'): outcome = 'collision'; break
                if info.get('arrive'): outcome = 'arrived'; break
                if env.done(): outcome = 'done'; break

            env.end(0)

            # --- Per-episode data ---
            episodes.append({
                'ep': ep,
                'outcome': outcome,
                'steps': step + 1,
                'min_dist': float(ep_min_dist) if ep_min_dist < float('inf') else 0,
                'path_len': float(path_len),
                'speed': float(np.mean(ep_speeds)) if ep_speeds else 0,
                'trajectory': trajectory,
            })

            # --- ACI: per-episode update ---
            if method == 'aci' and aci is not None:
                violation = (outcome == 'collision') or aci_violation
                aci.update(violation)

            # --- CPSA Online: ACI-style asymmetric per-episode finetuning ---
            # Safe episodes: tiny update (loss_scale=0.00025) to maintain tau
            # Collision: full update (loss_scale=1.0) to push tau up
            if method == 'cpsa_v4_online' and hasattr(planner, '_cpsa_online_adapter') and planner._cpsa_online_adapter is not None:
                buf = planner._cpsa_episode_buffer
                if buf:
                    all_d_pred = torch.cat([d for _, d in buf], dim=0)
                    if outcome == 'collision':
                        d_gt_proxy = all_d_pred * 0.3  # force tau up
                    else:
                        d_gt_proxy = all_d_pred.clone()  # maintain tau
                    planner.cpsa_episode_feedback(
                        d_gt_proxy, outcome=outcome, noise_std=noise_std)

            if outcome == 'collision': coll += 1
            if outcome == 'arrived': arrv += 1
            min_dists.append(ep_min_dist if ep_min_dist < float('inf') else 0)
            if outcome == 'arrived':
                paths.append(path_len)
                ep_speeds and speeds.append(float(np.mean(ep_speeds)))

        except Exception as e:
            print(f"EPISODE_ERROR: ep={ep} {e}", flush=True)
            try: env.end(0)
            except: pass

    os.unlink(tf_e.name)
    os.unlink(tf_p.name)

    n = n_ep
    result = {
        'env': env_name, 'noise_cm': noise_cm, 'method': method,
        'collisions': coll, 'collision_rate': coll / n,
        'arrivals': arrv, 'success_rate': arrv / n,
        'avg_path': float(np.mean(paths)) if paths else 0,
        'avg_min_distance': float(np.mean(min_dists)) if min_dists else 0,
        'min_min_distance': float(np.min(min_dists)) if min_dists else 0,
        'avg_speed': float(np.mean(speeds)) if speeds else 0,
        'episodes': episodes,
    }
    if method == 'aci' and aci is not None:
        result['aci_alpha_final'] = float(aci.get_alpha())
        result['aci_q_hat_final_cm'] = float(aci.get_q_hat() * 100)
    if method == 'cbf' and cbf_filter is not None:
        result['cbf_intervention_rate'] = cbf_filter.intervention_rate
    if method == 'cpsa_v4_online' and hasattr(planner, '_cpsa_online_adapter') and planner._cpsa_online_adapter is not None:
        stats = planner._cpsa_online_adapter.get_stats()
        result['cpsa_online_n_updates'] = stats['n_updates']
        result['cpsa_online_buffer'] = stats['buffer_size']
        result['cpsa_online_tau_mean'] = stats.get('recent_tau_mean', 0)

    print("RESULT_JSON:" + json.dumps(result), flush=True)


def np_to_tensor(arr):
    if isinstance(arr, torch.Tensor):
        return arr
    return torch.from_numpy(np.asarray(arr, dtype=np.float32))


if __name__ == '__main__':
    main()
