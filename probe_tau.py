"""Probe tau diversity across corridor steps."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from neupan import neupan
from neupan.risk_calibration import RiskCalibrator
from irsim.env import EnvBase

BASE = os.path.dirname(__file__)
CKPT = 'experiments/cp_head_output/cpsa_v4_universal.pth'

with open(os.path.join(BASE, 'experiments/calibration_output/calibration.json')) as f:
    cal_data = json.load(f)
cal = RiskCalibrator()
cal.fit(np.array(cal_data['calibration_scores']))
q_asy = cal.compute_q_hat(0.05)

env_file = 'example/corridor/diff/env.yaml'
plan_file = 'example/corridor/diff/planner.yaml'

env = EnvBase(os.path.join(BASE, env_file), display=False, save_ani=False)
planner = neupan.init_from_yaml(os.path.join(BASE, plan_file))
planner.enable_risk_calibration(
    budget_strategy='cpsa', cp_checkpoint=os.path.join(BASE, CKPT),
    q_hat=q_asy, noise_std=0.01, env_type=0
)

print("Probing tau diversity at different steps...")
for i in range(100):
    state = env.get_robot_state()
    scan = env.get_lidar_scan()
    pts = planner.scan_to_point(state, scan)
    action, info = planner(state, pts, None)

    # Every 10 steps, print tau stats
    if i % 10 == 5 and info.get("cpsa_margins") is not None:
        m = info["cpsa_margins"]
        pts_ = info.get("cpsa_margin_points")
        n_pts = len(pts_.T) if pts_ is not None else 0
        print(f"step {i:3d}: n={len(m):2d}, tau=[{m.min():.4f},{m.max():.4f}]m range={m.max()-m.min():.4f}m mean={m.mean():.4f}m d_pred_range_tau={m.max() - m.min():.4f}")

    env.step(action)
    env.render()

env.end(3)
