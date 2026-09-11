"""Generate Figure 1: Full LiDAR + τ heatmap using all obstacle points."""
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

env_name = sys.argv[1] if len(sys.argv) > 1 else 'corridor'
env_file = f'example/{env_name}/diff/env.yaml'
plan_file = f'example/{env_name}/diff/planner.yaml'
env_type = {'corridor': 0, 'pf_obs': 2}.get(env_name, 0)

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
# Zero CP correction to use raw network output for visualization
planner._cpsa_q_star = 0.0
planner._risk_calibration['cpsa_config'] = {'disable_smoothing': True}
if planner._risk_calibration:
    planner._risk_calibration['q_hat'] = 0.0

# Sweep to find step with best tau diversity
best_step = 0
best_range = 0
best_data = None

for i in range(100):
    state = env.get_robot_state()
    scan = env.get_lidar_scan()
    pts = planner.scan_to_point(state, scan)
    action, info = planner(state, pts, None)

    # Use full all_margins (all obstacle points) from patched neupan
    all_m = info.get("cpsa_all_margins")
    all_p = info.get("cpsa_all_points")
    if all_m is not None and len(all_m) > 20:
        rng = all_m.max() - all_m.min()
        if rng > best_range:
            best_range = rng
            best_step = i
            best_data = {'margins': all_m.copy(), 'points': all_p.cpu().numpy().copy() if hasattr(all_p, 'cpu') else all_p.copy()}

    env.step(action)
    env.render()

env.end(3)

if best_data is None:
    print("No data found!")
    sys.exit(1)

pts = best_data['points']  # (2, N) → transpose to (N, 2)
pts_xy = pts.T  # (N, 2)
margins_cm = best_data['margins'] * 100  # m -> cm

print(f"Best step {best_step}: {len(pts_xy)} points, tau=[{margins_cm.min():.1f},{margins_cm.max():.1f}]cm, range={margins_cm.max()-margins_cm.min():.2f}cm")

# --- 4-panel figure ---
fig = plt.figure(figsize=(13, 3.0))

# Panel 1: Raw LiDAR scan
ax1 = fig.add_subplot(1, 4, 1)
ax1.scatter(pts_xy[:, 0], pts_xy[:, 1], c='gray', s=3, alpha=0.7)
ax1.scatter(0, 0, c='green', s=80, marker='o', edgecolors='darkgreen', linewidths=1.5)
ax1.set_xlabel('x (m)', fontsize=8); ax1.set_ylabel('y (m)', fontsize=8)
ax1.set_title('(a) LiDAR Point Cloud', fontsize=9, fontweight='bold')
ax1.set_aspect('equal'); ax1.grid(alpha=0.2)

# Panel 2: CPSA τ heatmap
ax2 = fig.add_subplot(1, 4, 2)
vmin = max(0, margins_cm.min() - 0.5)
vmax = margins_cm.max() + 0.5
sc = ax2.scatter(pts_xy[:, 0], pts_xy[:, 1], c=margins_cm, cmap='coolwarm',
                 s=6, alpha=0.9, edgecolors='none', vmin=vmin, vmax=vmax)
cbar = plt.colorbar(sc, ax=ax2, shrink=0.75)
cbar.set_label(r'Learned tightening $\tau(p_i)$ [cm]', fontsize=8)
ax2.scatter(0, 0, c='green', s=80, marker='o', edgecolors='darkgreen', linewidths=1.5)
ax2.set_xlabel('x (m)', fontsize=8); ax2.set_ylabel('y (m)', fontsize=8)
ax2.set_title('(b) Learned Safety Tightening', fontsize=9, fontweight='bold')
ax2.set_aspect('equal'); ax2.grid(alpha=0.2)

# Panel 3: τ vs. Distance (closer points → higher τ)
ax3 = fig.add_subplot(1, 4, 3)
dist = np.sqrt(pts_xy[:, 0]**2 + pts_xy[:, 1]**2)
ax3.scatter(dist, margins_cm, c=margins_cm, cmap='coolwarm', s=12, alpha=0.7, edgecolors='none')
# Fit a rough trend
z = np.polyfit(dist, margins_cm, 1)
p = np.poly1d(z)
xs_line = np.linspace(dist.min(), dist.max(), 50)
ax3.plot(xs_line, p(xs_line), '--k', linewidth=1, alpha=0.5, label='trend')
ax3.set_xlabel('Distance from robot (m)', fontsize=8)
ax3.set_ylabel(r'$\tau$ (cm)', fontsize=8)
ax3.set_title('(c) Tightening vs. Distance', fontsize=9, fontweight='bold')
ax3.legend(fontsize=7); ax3.grid(alpha=0.3)

# Panel 4: τ histogram
ax4 = fig.add_subplot(1, 4, 4)
ax4.hist(margins_cm, bins=20, color='#C73E1D', alpha=0.7, edgecolor='white')
ax4.axvline(margins_cm.mean(), color='darkred', linestyle='--', linewidth=1.5,
            label=f'mean={margins_cm.mean():.1f}cm')
ax4.set_xlabel(r'$\tau$ (cm)', fontsize=8); ax4.set_ylabel('Count', fontsize=8)
ax4.set_title(f'(d) Distribution (range={margins_cm.max()-margins_cm.min():.1f}cm)', fontsize=9, fontweight='bold')
ax4.legend(fontsize=7)

fig.suptitle(f'Learned Safety Tightening Map — {env_name} (step {best_step})', fontsize=11, fontweight='bold')
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join('figures', f'fig1_cpsa_tau.{ext}'), dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: figures/fig1_cpsa_tau.pdf")
