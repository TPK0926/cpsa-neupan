"""Capture one frame of CPSA τ values and generate heatmap figure."""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from neupan import neupan
from neupan.risk_calibration import RiskCalibrator
from irsim.env import EnvBase

BASE = os.path.dirname(__file__)
CKPT = 'experiments/cp_head_output/cpsa_v4_universal.pth'

env_name = sys.argv[1] if len(sys.argv) > 1 else 'corridor'
print(f"Running {env_name}...")

ENV_MAP = {
    'corridor':   ('example/corridor/diff/env.yaml',   'example/corridor/diff/planner.yaml',   0),
    'convex_obs': ('example/convex_obs/diff/env.yaml',  'example/convex_obs/diff/planner.yaml', 1),
    'pf_obs':     ('example/pf_obs/diff/env.yaml',      'example/pf_obs/diff/planner.yaml',     2),
    'non_obs':    ('example/non_obs/diff/env.yaml',     'example/non_obs/diff/planner.yaml',    4),
}

env_file, plan_file, env_type = ENV_MAP.get(env_name, ENV_MAP['corridor'])

with open(os.path.join(BASE, 'experiments/calibration_output/calibration.json')) as f:
    cal_data = json.load(f)
cal = RiskCalibrator()
cal.fit(np.array(cal_data['calibration_scores']))
q_asy = cal.compute_q_hat(0.05)

env = EnvBase(os.path.join(BASE, env_file), display=False, save_ani=False)
planner = neupan.init_from_yaml(os.path.join(BASE, plan_file))
planner.enable_risk_calibration(
    budget_strategy='cpsa',
    cp_checkpoint=os.path.join(BASE, CKPT),
    q_hat=q_asy, noise_std=0.01, env_type=env_type
)

# Choose a step that's in a narrow passage area (step 50-150 typically)
capture_step = 20
tau_data = None

for i in range(capture_step + 10):
    state = env.get_robot_state()
    scan = env.get_lidar_scan()
    pts = planner.scan_to_point(state, scan)
    action, info = planner(state, pts, None)

    if i == capture_step:
        # Capture ALL obstacle points + tau via internal access
        try:
            obs_pts = planner.pan.dune_layer.obstacle_points  # (2, N) tensor
            if obs_pts is not None and obs_pts.shape[1] > 5:
                all_margins = planner._compute_cpsa_margins(obs_pts)
                if all_margins is not None and len(all_margins) > 5:
                    tau_data = {
                        'points': obs_pts.cpu().numpy().T.tolist(),
                        'margins': all_margins.tolist(),
                    }
                    print(f"Captured ALL {len(all_margins)} points at step {i}, "
                          f"tau range=[{all_margins.min():.4f}, {all_margins.max():.4f}]m")
        except Exception as e:
            print(f"Full capture failed: {e}")
        # Fallback: use info dict (top nrmp_max_num points)
        if tau_data is None:
            margins = info.get("cpsa_margins")
            points = info.get("cpsa_margin_points")
            if margins is not None:
                tau_data = {
                    'points': np.array(points).T.tolist(),
                    'margins': np.array(margins).flatten().tolist(),
                }
                print(f"Fallback: {len(margins)} points")

    env.step(action)
    env.render()

env.end(3)

if tau_data is None:
    print("ERROR: No tau data captured!")
    sys.exit(1)

# Generate heatmap
fig, ax = plt.subplots(figsize=(6, 5))
pts = np.array(tau_data['points'])
margins = np.array(tau_data['margins'])

# Scale margins from meters to cm for display
margins_cm = margins * 100
sc = ax.scatter(pts[:, 0], pts[:, 1], c=margins_cm, cmap='jet',
                s=12, alpha=0.85, edgecolors='none',
                vmin=0, vmax=8)

cbar = plt.colorbar(sc, ax=ax, label=r'$\tau(p_i)$ (cm)', shrink=0.8)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title(f'CPSA Per-Point Safety Tightening — {env_name} (step {capture_step})')
ax.set_aspect('equal')
ax.grid(alpha=0.3)

outpath = os.path.join(BASE, 'figures', 'fig_tau_heatmap.pdf')
os.makedirs(os.path.dirname(outpath), exist_ok=True)
fig.tight_layout()
fig.savefig(outpath, dpi=200, bbox_inches='tight')
fig.savefig(outpath.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
plt.close()

print(f"Saved: {outpath}")
