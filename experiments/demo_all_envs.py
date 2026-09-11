"""Demo: 逐个环境跑一集, 实时可视化。Enter 开始, q 跳过, ctrl+c 退出。
支持 --method cpsa_v4 / cp_global / vanilla
"""
import sys, os, yaml, tempfile, time, json, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from neupan import neupan
from irsim.env import EnvBase
from neupan.risk_calibration import RiskCalibrator

BASE = os.path.join(os.path.dirname(__file__), '..')

ENVS = [
    ('1_corridor',       'example/corridor/diff/env.yaml',       'example/corridor/diff/planner.yaml',       0),
    ('2_convex_obs',     'example/convex_obs/diff/env.yaml',     'example/convex_obs/diff/planner.yaml',     1),
    ('3_pf_obs',         'example/pf_obs/diff/env.yaml',         'example/pf_obs/diff/planner.yaml',         2),
    ('4_dyna_obs',       'example/dyna_obs/diff/env.yaml',       'example/dyna_obs/diff/planner.yaml',       3),
    ('5_non_obs',        'example/non_obs/diff/env.yaml',        'example/non_obs/diff/planner.yaml',        4),
    ('6_mixed_corridor', 'example/mixed_corridor/diff/env.yaml', 'example/mixed_corridor/diff/planner.yaml', 5),
]

parser = argparse.ArgumentParser()
parser.add_argument('--method', default='cpsa_v4', choices=['vanilla', 'cp_global', 'cpsa_v4'])
args = parser.parse_args()
method = args.method

# Load calibration (needed for cp_global and cpsa_v4)
cal = None
q_asy = None
if method in ('cp_global', 'cpsa_v4'):
    cal_path = os.path.join(BASE, 'experiments', 'calibration_output', 'calibration.json')
    with open(cal_path) as f:
        cal_data = json.load(f)
    cal = RiskCalibrator()
    cal.fit(np.array(cal_data['calibration_scores']))
    q_asy = cal.compute_q_hat(0.05)

CKPT = os.path.join(BASE, 'experiments', 'cp_head_output', 'cpsa_v4_universal.pth')

for name, env_f, plan_f, env_type in ENVS:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {name}  [{method}]")
    print(f"  ENTER=开始  q=跳过  ctrl+c=退出")
    print(f"{bar}")

    try:
        inp = input('> ').strip().lower()
    except (EOFError, KeyboardInterrupt):
        break
    if inp == 'q':
        continue

    with open(os.path.join(BASE, env_f)) as f:
        ec = yaml.safe_load(f)
    for rob in ec.get('robot', []):
        for s in rob.get('sensors', []):
            s['noise'] = True
            s['std'] = 0.01

    with open(os.path.join(BASE, plan_f)) as f:
        pc = yaml.safe_load(f)
    pc['time_print'] = False

    tf_e = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(ec, tf_e); tf_e.close()
    tf_p = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(pc, tf_p); tf_p.close()

    env = EnvBase(tf_e.name, display=True, save_ani=False)
    planner = neupan.init_from_yaml(tf_p.name)

    if method == 'cpsa_v4':
        planner.enable_risk_calibration(
            budget_strategy='cpsa', cp_checkpoint=CKPT,
            q_hat=q_asy, noise_std=0.01, env_type=env_type)
    elif method == 'cp_global':
        planner.enable_risk_calibration(q_hat=q_asy, budget_strategy='global')

    outcome = 'timeout'
    min_dist = float('inf')
    for step in range(500):
        state = env.get_robot_state()
        scan = env.get_lidar_scan()
        pts = planner.scan_to_point(state, scan)

        md = planner.min_distance
        try:
            md = float(md.item() if hasattr(md, 'item') else md)
        except Exception:
            pass
        if md < min_dist:
            min_dist = md

        action, info = planner(state, pts, None)

        if info.get('stop'):
            outcome = 'collision'
        if info.get('arrive'):
            outcome = 'arrived'

        env.draw_points(planner.dune_points, s=25, c="g", refresh=True)
        env.draw_points(planner.nrmp_points, s=13, c="r", refresh=True)
        env.draw_trajectory(info["opt_state_list"], "r", refresh=True)
        env.draw_trajectory(info["ref_state_list"], "b", refresh=True)

        env.step(action)
        env.render()

        if step == 0:
            env.draw_trajectory(planner.initial_path, traj_type="-k")

        if info.get('stop'):
            break
        if info.get('arrive'):
            break
        if env.done():
            outcome = 'done'
            break

        time.sleep(0.02)

    env.end(3)
    os.unlink(tf_e.name)
    os.unlink(tf_p.name)
    print(f"  -> {outcome} | steps={step+1} | min_dist={min_dist*100:.1f}cm")

print("\nDone.")
