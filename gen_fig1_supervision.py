"""Generate τ heatmap + residual vs τ correlation plot."""
import sys, os, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from neupan import neupan
from neupan.robot import robot
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

r = robot(receding=10, step_time=0.1, kinematics='diff', length=1.6, width=2.0)
G_np, h_np = r.G, r.h

import cvxpy as cp

def compute_d_gt_fast(local_pts):
    """Compute d_gt via convex opt for each point (batch CVXPY)."""
    n = len(local_pts)
    edge_dim = G_np.shape[0]
    d_gt = np.zeros(n)
    mu_var = cp.Variable((edge_dim, 1), nonneg=True)
    p = cp.Parameter((2, 1))
    prob = cp.Problem(cp.Maximize(mu_var.T @ (G_np @ p - h_np)),
                       [cp.norm(G_np.T @ mu_var) <= 1])
    for i in range(n):
        p.value = local_pts[i].reshape(2, 1)
        try:
            prob.solve(solver=cp.ECOS, warm_start=True)
            d_gt[i] = prob.value if prob.value is not None else 0.0
        except Exception:
            d_gt[i] = 0.0
    return d_gt

env = EnvBase(os.path.join(BASE, env_file), display=False, save_ani=False)
planner = neupan.init_from_yaml(os.path.join(BASE, plan_file))
planner.enable_risk_calibration(
    budget_strategy='cpsa', cp_checkpoint=os.path.join(BASE, CKPT),
    q_hat=q_asy, noise_std=0.01, env_type=env_type
)
planner._cpsa_q_star = 0.0
planner._risk_calibration['cpsa_config'] = {'disable_smoothing': True}
if planner._risk_calibration:
    planner._risk_calibration['q_hat'] = 0.0

# Sweep for best step
best_step, best_range = 0, 0
best_data = None
for i in range(100):
    state = env.get_robot_state()
    scan = env.get_lidar_scan()
    pts = planner.scan_to_point(state, scan)
    action, info = planner(state, pts, None)
    all_m = info.get("cpsa_all_margins")
    all_p = info.get("cpsa_all_points")
    if all_m is not None and len(all_m) > 20:
        rng = all_m.max() - all_m.min()
        if rng > best_range:
            best_range = rng
            best_step = i
            pts_np = all_p.cpu().numpy() if hasattr(all_p, 'cpu') else all_p
            # Convert global points to local for d_gt computation
            x, y, theta = state.flatten()[:3]
            local_pts = np.zeros_like(pts_np.T)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            for j in range(pts_np.shape[1]):
                dx = pts_np[0, j] - x
                dy = pts_np[1, j] - y
                local_pts[j, 0] = cos_t * dx + sin_t * dy
                local_pts[j, 1] = -sin_t * dx + cos_t * dy
            # Also get d_pred from DUNE
            if hasattr(planner.pan.dune_layer, 'distances_0'):
                d_pred_np = planner.pan.dune_layer.distances_0.detach().cpu().numpy()
            else:
                d_pred_np = np.zeros(len(all_m))
            best_data = {
                'margins': all_m.copy(),
                'points': pts_np,
                'local_pts': local_pts,
                'd_pred': d_pred_np,
                'step': i,
            }
    env.step(action)
    env.render()
env.end(3)

if best_data is None:
    print("No data found!"); sys.exit(1)

# Compute d_gt for the captured points
print(f"Computing d_gt for {len(best_data['local_pts'])} points at step {best_step}...")
d_gt = compute_d_gt_fast(best_data['local_pts'])
best_data['d_gt'] = d_gt

residual = best_data['d_pred'] - d_gt
margins_cm = best_data['margins'] * 100
residual_cm = residual * 100

print(f"τ range: [{margins_cm.min():.1f}, {margins_cm.max():.1f}]cm")
print(f"residual range: [{residual_cm.min():.1f}, {residual_cm.max():.1f}]cm")
corr = np.corrcoef(residual_cm, margins_cm)[0, 1]
print(f"τ vs residual correlation: r={corr:.3f}")

pts_xy = best_data['points'].T

# --- 4-panel figure ---
fig = plt.figure(figsize=(13, 3.0))

# Panel 1: LiDAR scan
ax1 = fig.add_subplot(1, 4, 1)
ax1.scatter(pts_xy[:, 0], pts_xy[:, 1], c='gray', s=3, alpha=0.7)
ax1.scatter(0, 0, c='green', s=80, marker='o', edgecolors='darkgreen', linewidths=1.5)
ax1.set_xlabel('x (m)', fontsize=8); ax1.set_ylabel('y (m)', fontsize=8)
ax1.set_title('(a) LiDAR Point Cloud', fontsize=9, fontweight='bold')
ax1.set_aspect('equal'); ax1.grid(alpha=0.2)

# Panel 2: τ heatmap
ax2 = fig.add_subplot(1, 4, 2)
vmin, vmax = max(0, margins_cm.min() - 0.5), margins_cm.max() + 0.5
sc = ax2.scatter(pts_xy[:, 0], pts_xy[:, 1], c=margins_cm, cmap='coolwarm',
                 s=6, alpha=0.9, edgecolors='none', vmin=vmin, vmax=vmax)
cbar = plt.colorbar(sc, ax=ax2, shrink=0.75)
cbar.set_label(r'Learned tightening $\tau(p_i)$ [cm]', fontsize=8)
ax2.scatter(0, 0, c='green', s=80, marker='o', edgecolors='darkgreen', linewidths=1.5)
ax2.set_xlabel('x (m)', fontsize=8); ax2.set_ylabel('y (m)', fontsize=8)
ax2.set_title('(b) Learned Safety Tightening', fontsize=9, fontweight='bold')
ax2.set_aspect('equal'); ax2.grid(alpha=0.2)

# Panel 3: τ vs residual — KEY PLOT
ax3 = fig.add_subplot(1, 4, 3)
ax3.scatter(residual_cm, margins_cm, c=margins_cm, cmap='coolwarm',
            s=15, alpha=0.75, edgecolors='none', vmin=vmin, vmax=vmax)
# Fit line
z = np.polyfit(residual_cm, margins_cm, 1)
p = np.poly1d(z)
xs_line = np.linspace(residual_cm.min(), residual_cm.max(), 50)
ax3.plot(xs_line, p(xs_line), '--k', linewidth=1, alpha=0.6,
         label=f'r={corr:.2f}')
ax3.set_xlabel(r'Residual $d_{pred} - d_{gt}$ (cm)', fontsize=8)
ax3.set_ylabel(r'$\tau$ (cm)', fontsize=8)
ax3.set_title('(c) Tightening vs. Pred. Error', fontsize=9, fontweight='bold')
ax3.legend(fontsize=7); ax3.grid(alpha=0.3)

# Panel 4: residual histogram + τ histogram (dashed)
ax4 = fig.add_subplot(1, 4, 4)
ax4.hist(residual_cm, bins=15, color='gray', alpha=0.5, edgecolor='white', label=f'Residual (μ={residual_cm.mean():.1f})')
ax4.hist(margins_cm, bins=15, color='#C73E1D', alpha=0.5, edgecolor='white', label=f'τ (μ={margins_cm.mean():.1f})')
ax4.set_xlabel('cm', fontsize=8); ax4.set_ylabel('Count', fontsize=8)
ax4.set_title('(d) Residual vs τ Distribution', fontsize=9, fontweight='bold')
ax4.legend(fontsize=7)

fig.suptitle(f'CPSA Supervision Alignment — {env_name} (step {best_step}, r={corr:.2f})',
             fontsize=10, fontweight='bold')
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join('figures', f'fig1_cpsa_supervision.{ext}'), dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: figures/fig1_cpsa_supervision.pdf (r={corr:.3f})")
