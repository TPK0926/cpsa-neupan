"""Generate Figure 1 teaser: Full 360° LiDAR + τ heatmap + trajectory."""
import sys, os, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from neupan import neupan
from neupan.risk_calibration import RiskCalibrator
from irsim.env import EnvBase

BASE = os.path.dirname(__file__)
CKPT = 'experiments/cp_head_output/cpsa_v4_universal.pth'

ENV_MAP = {
    'corridor':   ('example/corridor/diff/env.yaml',   'example/corridor/diff/planner.yaml',   0),
    'pf_obs':     ('example/pf_obs/diff/env.yaml',      'example/pf_obs/diff/planner.yaml',     2),
}
env_name = sys.argv[1] if len(sys.argv) > 1 else 'pf_obs'
env_file, plan_file, env_type = ENV_MAP.get(env_name, ENV_MAP['pf_obs'])

with open(os.path.join(BASE, 'experiments/calibration_output/calibration.json')) as f:
    cal_data = json.load(f)
cal = RiskCalibrator()
cal.fit(np.array(cal_data['calibration_scores']))
q_asy = cal.compute_q_hat(0.05)

env = EnvBase(os.path.join(BASE, env_file), display=False, save_ani=False)
planner = neupan.init_from_yaml(os.path.join(BASE, plan_file))
planner.enable_risk_calibration(
    budget_strategy='cpsa', cp_checkpoint=os.path.join(BASE, CKPT),
    q_hat=q_asy, noise_std=0.01, env_type=env_type
)

capture_step = int(sys.argv[2]) if len(sys.argv) > 2 else 20
state_trace = []
all_scan_pts = None
tau_data = None

for i in range(max(capture_step + 5, 60)):
    state = env.get_robot_state()
    scan = env.get_lidar_scan()
    pts = planner.scan_to_point(state, scan)
    action, info = planner(state, pts, None)
    state_trace.append([float(state[0]), float(state[1])])

    if i == capture_step:
        raw_scan = env.get_lidar_scan()
        full_pts = planner.scan_to_point(state, raw_scan)
        if full_pts is not None and full_pts.shape[1] > 10:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(full_pts).float()
                # Zero CP correction
                save_q_star = planner._cpsa_q_star
                save_q_hat = planner._risk_calibration.get('q_hat', 0.0) if planner._risk_calibration else 0.0
                planner._cpsa_q_star = 0.0
                if planner._risk_calibration:
                    planner._risk_calibration['q_hat'] = 0.0

                # Direct network call to bypass d_pred rank smoothing
                import torch.nn as nn
                N = obs_tensor.shape[1]
                pts = obs_tensor.T  # (N, 2)
                d_pred = planner.pan.dune_layer.distances_0
                if d_pred is None or d_pred.shape[0] != N:
                    d_pred = torch.ones(N)

                # Build 17 features (same as _compute_cpsa_margins)
                eps = 1e-6
                radius = torch.sqrt(pts[:, 0]**2 + pts[:, 1]**2 + eps)
                angle = torch.arctan2(pts[:, 1], pts[:, 0])
                proximity_ratio = torch.abs(d_pred) / (radius + eps)
                # ... simplified for now

                # Just call the raw network on cpsa_net directly via compiled features
                # Instead, let's patch the internal smoothing temporarily
                import neupan.blocks.cpsa_risk_net as lrn

                tau_raw = planner._compute_cpsa_margins(obs_tensor)

                planner._cpsa_q_star = save_q_star
                if planner._risk_calibration:
                    planner._risk_calibration['q_hat'] = save_q_hat

                if tau_raw is not None and len(tau_raw) > 10:
                    all_scan_pts = full_pts
                    tau_data = {
                        'points': full_pts.T.tolist(),
                        'margins': tau_raw.tolist(),
                    }
                    print(f"Captured {len(tau_raw)} points, tau=[{tau_raw.min():.3f},{tau_raw.max():.3f}]m")

    env.step(action)
    env.render()
env.end(3)

if tau_data is None:
    print("ERROR: No tau data!")
    sys.exit(1)

# --- Generate Figure ---
pts = np.array(tau_data['points'])
margins_cm = np.array(tau_data['margins']) * 100
trace = np.array(state_trace)
xs, ys = trace[:, 0], trace[:, 1]

fig = plt.figure(figsize=(12, 3.2))

# Panel 1: Full LiDAR scan (360°, grey points)
ax1 = fig.add_subplot(1, 4, 1)
ax1.scatter(pts[:, 0], pts[:, 1], c='gray', s=3, alpha=0.6, label='All scan points')
ax1.scatter(0, 0, c='green', s=70, marker='o', edgecolors='darkgreen', linewidths=1, label='Robot')
ax1.set_xlabel('x (m)'); ax1.set_ylabel('y (m)')
ax1.set_title('(a) Full 360° LiDAR Scan', fontsize=9)
ax1.set_aspect('equal'); ax1.legend(fontsize=7); ax1.grid(alpha=0.2)

# Panel 2: CPSA τ heatmap (full scan, coolwarm colormap)
ax2 = fig.add_subplot(1, 4, 2)
sc = ax2.scatter(pts[:, 0], pts[:, 1], c=margins_cm, cmap='coolwarm',
                 s=4, alpha=0.85, edgecolors='none',
                 vmin=0, vmax=max(8, np.percentile(margins_cm, 95)))
cbar = plt.colorbar(sc, ax=ax2, shrink=0.75)
cbar.set_label(r'$\tau(p_i)$ (cm)', fontsize=8)
ax2.scatter(0, 0, c='green', s=70, marker='o', edgecolors='darkgreen', linewidths=1)
ax2.set_xlabel('x (m)'); ax2.set_ylabel('y (m)')
ax2.set_title('(b) CPSA Per-Point Safety Tightening', fontsize=9)
ax2.set_aspect('equal'); ax2.grid(alpha=0.2)

# Panel 3: τ vs distance to obstacle scatter
ax3 = fig.add_subplot(1, 4, 3)
distances = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
ax3.scatter(distances, margins_cm, c='#C73E1D', s=15, alpha=0.6, edgecolors='none')
ax3.set_xlabel('Distance from robot (m)')
ax3.set_ylabel(r'$\tau$ (cm)')
ax3.set_title('(c) τ vs. Distance', fontsize=9)
ax3.grid(alpha=0.3)

# Panel 4: τ histogram + trajectory info
ax4 = fig.add_subplot(1, 4, 4)
ax4.hist(margins_cm, bins=25, color='#2E86AB', alpha=0.7, edgecolor='white')
ax4.axvline(margins_cm.mean(), color='darkblue', linestyle='--', linewidth=1.5,
            label=f'mean={margins_cm.mean():.1f}cm')
ax4.axvline(margins_cm.min(), color='green', linestyle=':', linewidth=1)
ax4.axvline(margins_cm.max(), color='red', linestyle=':', linewidth=1)
ax4.set_xlabel(r'$\tau$ (cm)'); ax4.set_ylabel('Count')
ax4.set_title(f'(d) τ Distribution', fontsize=9)
ax4.legend(fontsize=7)

fig.suptitle(f'CPSA Safety Allocation — {env_name} (step {capture_step}, {len(pts)} points)',
             fontsize=10, fontweight='bold')
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join('figures', f'fig1_teaser.{ext}'), dpi=200, bbox_inches='tight')
plt.close()
print("Figure saved to figures/fig1_teaser.pdf")
