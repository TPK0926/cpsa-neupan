"""
Single-episode visualization test for NeuPAN navigation.
Usage:
    python experiments/visual_test.py
    python experiments/visual_test.py --env corridor --noise 2 --method vanilla
    python experiments/visual_test.py --env dyna_obs --noise 1 --method cpsa_v4
"""

import sys, os, argparse, yaml, tempfile, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neupan import neupan
from irsim.env import EnvBase

ENV_MAP = {
    'corridor':    ('example/corridor/diff/env.yaml',    'example/corridor/diff/planner.yaml',    0),
    'convex_obs':  ('example/convex_obs/diff/env.yaml',  'example/convex_obs/diff/planner.yaml',  1),
    'pf_obs':      ('example/pf_obs/diff/env.yaml',      'example/pf_obs/diff/planner.yaml',      2),
    'dyna_obs':    ('example/dyna_obs/diff/env.yaml',    'example/dyna_obs/diff/planner.yaml',    3),
    'non_obs':     ('example/non_obs/diff/env.yaml',     'example/non_obs/diff/planner.yaml',     4),
}
CKPT_V4 = 'experiments/cp_head_output/cpsa_v4_universal.pth'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='corridor', choices=list(ENV_MAP))
    parser.add_argument('--noise', type=int, default=1, help='noise in cm')
    parser.add_argument('--method', default='vanilla',
                        choices=['vanilla', 'cp_global', 'cp_dw', 'cpsa_v4', 'cbf', 'aci'])
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--speed', type=float, default=0.02, help='delay between steps (s)')
    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(__file__), '..')
    env_file, plan_file, env_type = ENV_MAP[args.env]
    noise_std = args.noise / 100.0

    # --- Load calibration ---
    cal_path = os.path.join(base_dir, 'experiments/calibration_output/calibration.json')
    import json
    with open(cal_path) as f:
        cal_data = json.load(f)
    from neupan.risk_calibration import RiskCalibrator
    cal = RiskCalibrator()
    cal.fit(np.array(cal_data['calibration_scores']))
    q_asy = cal.compute_q_hat(0.05)

    # --- Env config ---
    with open(os.path.join(base_dir, env_file)) as f:
        ec = yaml.safe_load(f.read())
    for rob in ec.get('robot', []):
        for s in rob.get('sensors', []):
            if noise_std > 0:
                s['noise'] = True
                s['std'] = noise_std
            else:
                s['noise'] = False

    # --- Plan config ---
    with open(os.path.join(base_dir, plan_file)) as f:
        pc = yaml.safe_load(f.read())
    pc['time_print'] = False

    tf_e = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(ec, tf_e); tf_e.close()
    tf_p = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(pc, tf_p); tf_p.close()

    print(f"Env: {args.env} | Noise: {args.noise}cm | Method: {args.method}")
    print("Starting visualization...")

    env = EnvBase(tf_e.name, display=True, save_ani=False)
    planner = neupan.init_from_yaml(tf_p.name)

    # --- Enable method ---
    if args.method == 'cp_global':
        planner.enable_risk_calibration(q_hat=q_asy, budget_strategy='global')
    elif args.method == 'cp_dw':
        planner.enable_risk_calibration(q_hat=q_asy, budget_strategy='distance_weighted')
    elif args.method == 'cpsa_v4':
        ckpt = os.path.join(base_dir, CKPT_V4)
        planner.enable_risk_calibration(
            budget_strategy='cpsa', cp_checkpoint=ckpt,
            q_hat=q_asy, noise_std=noise_std, env_type=env_type)

    outcome = 'timeout'
    min_dist = float('inf')
    t0 = time.time()

    for step in range(args.max_steps):
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
        env.step(action)
        env.render()            # 必须手动调 render，否则画面不更新

        if info.get('stop'):
            outcome = 'collision'
            break
        if info.get('arrive'):
            outcome = 'arrived'
            break
        if env.done():
            outcome = 'done'
            break

        time.sleep(args.speed)

    elapsed = time.time() - t0
    env.end(0)
    os.unlink(tf_e.name)
    os.unlink(tf_p.name)

    print(f"Outcome: {outcome} | Steps: {step+1} | MinDist: {min_dist*100:.1f}cm | Time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
