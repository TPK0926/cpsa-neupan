"""Run NeuPAN + CPSA in IR-SIM with visualization.

Usage:
    python run_cpsa.py corridor
    python run_cpsa.py convex_obs
    python run_cpsa.py pf_obs
    python run_cpsa.py dyna_obs
    python run_cpsa.py non_obs
"""
import sys, os, json
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from neupan import neupan
from irsim.env import EnvBase
from neupan.risk_calibration import RiskCalibrator

ENV_MAP = {
    'corridor':   ('example/corridor/diff/env.yaml',   'example/corridor/diff/planner.yaml',   0),
    'convex_obs': ('example/convex_obs/diff/env.yaml',  'example/convex_obs/diff/planner.yaml', 1),
    'pf_obs':     ('example/pf_obs/diff/env.yaml',      'example/pf_obs/diff/planner.yaml',     2),
    'dyna_obs':   ('example/dyna_obs/diff/env.yaml',    'example/dyna_obs/diff/planner.yaml',   3),
    'non_obs':    ('example/non_obs/diff/env.yaml',     'example/non_obs/diff/planner.yaml',    4),
}
BASE = os.path.dirname(__file__)
CKPT = 'experiments/cp_head_output/cpsa_v4_universal.pth'

env_name = sys.argv[1] if len(sys.argv) > 1 else 'corridor'
assert env_name in ENV_MAP, f"Unknown env: {env_name}, choose from {list(ENV_MAP.keys())}"

env_file, plan_file, env_type = ENV_MAP[env_name]

# Calibration
with open(os.path.join(BASE, 'experiments/calibration_output/calibration.json')) as f:
    cal_data = json.load(f)
cal = RiskCalibrator()
cal.fit(np.array(cal_data['calibration_scores']))
q_asy = cal.compute_q_hat(0.05)

# Env
env = EnvBase(os.path.join(BASE, env_file), display=True, save_ani=False)

# Planner + CPSA
planner = neupan.init_from_yaml(os.path.join(BASE, plan_file))
planner.enable_risk_calibration(
    budget_strategy='cpsa',
    cp_checkpoint=os.path.join(BASE, CKPT),
    q_hat=q_asy, noise_std=0.01, env_type=env_type
)

for i in range(500):
    state = env.get_robot_state()
    scan = env.get_lidar_scan()
    pts = planner.scan_to_point(state, scan)

    action, info = planner(state, pts, None)

    if info["stop"]:
        print(f"Step {i}: collision stop")
    if info["arrive"]:
        print(f"Step {i}: arrived!")
        break

    env.draw_points(planner.dune_points, s=25, c="g", refresh=True)
    env.draw_points(planner.nrmp_points, s=13, c="r", refresh=True)
    env.draw_trajectory(info["opt_state_list"], "r", refresh=True)
    env.draw_trajectory(info["ref_state_list"], "b", refresh=True)
    env.step(action)
    env.render()

    if env.done():
        break

env.end(3)
