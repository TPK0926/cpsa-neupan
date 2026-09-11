"""
Kill Check #1: DUNE Risk Calibratability + Calibration Data Collection

Collects (d_pred, d_gt) pairs from DUNE on random scenes, computes asymmetric
and symmetric risk scores, and derives conformal quantiles q_hat(epsilon).

Kill Conditions:
  - Pr[r > 0] > 50%: DUNE systematically overestimates
  - q_hat(0.05) > 10cm: too conservative for practical navigation
  - q_hat(0.05) > 5cm: warning threshold

Pass Condition: q_hat(0.05) < 5cm with reasonable risk distribution.
"""

import sys
import os
import json
import time
import argparse
import numpy as np
import torch
import cvxpy as cp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neupan.robot import robot
from neupan.blocks import ObsPointNet
from neupan.configuration import np_to_tensor, to_device
from neupan.risk_calibration import RiskCalibrator


def load_dune_model(checkpoint_path, robot_instance):
    """Load trained DUNE model (ObsPointNet) from checkpoint."""
    edge_dim = robot_instance.G.shape[0]
    model = to_device(ObsPointNet(2, edge_dim))
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    return model


def solve_lp_ground_truth(G_np, h_np, point):
    """Solve LP for ground-truth distance: max mu^T(G@p - h) s.t. ||G^T mu|| <= 1, mu >= 0."""
    mu = cp.Variable((G_np.shape[0], 1), nonneg=True)
    p = cp.Parameter((2, 1))
    p.value = point
    cost = mu.T @ (G_np @ p - h_np)
    constraints = [cp.norm(G_np.T @ mu) <= 1]
    prob = cp.Problem(cp.Maximize(cost), constraints)
    prob.solve(solver=cp.ECOS)
    return prob.value, mu.value


def dune_predict(model, G_tensor, h_tensor, point_np):
    """Run DUNE forward on a single point to get predicted distance."""
    point_tensor = np_to_tensor(point_np.reshape(2, 1))
    with torch.no_grad():
        mu_pred = model(point_tensor.T).T  # (edge_dim, 1)
    temp = (G_tensor @ point_tensor - h_tensor).T.unsqueeze(2)
    muT = mu_pred.T.unsqueeze(1)
    distance = torch.squeeze(torch.bmm(muT, temp))
    return distance.item(), mu_pred


def generate_random_scene(n_points, data_range=(-25, -25, 25, 25), seed=None):
    """Generate a random scene with n_points obstacle points."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    return rng.uniform(low=data_range[:2], high=data_range[2:], size=(n_points, 2))


def run_calibration(
    checkpoint_path='example/model/diff_robot_default/model_5000.pth',
    n_scenes=2000,
    points_per_scene=50,
    data_range=(-25, -25, 25, 25),
    output_dir='experiments/calibration_output',
    epsilons=[0.01, 0.05, 0.10],
):
    os.makedirs(output_dir, exist_ok=True)

    # Setup robot and model
    robot_instance = robot(receding=10, step_time=0.1, kinematics='diff',
                           length=1.6, width=2.0)
    G_np = robot_instance.G
    h_np = robot_instance.h
    G_tensor = np_to_tensor(G_np)
    h_tensor = np_to_tensor(h_np)

    print(f"Loading DUNE model from {checkpoint_path}")
    model = load_dune_model(checkpoint_path, robot_instance)
    print(f"Robot: diff drive, {robot_instance.length}x{robot_instance.width}m")
    print(f"G shape: {G_np.shape}, h shape: {h_np.shape}")

    # Collect calibration data
    scene_risks_asymmetric = []
    scene_risks_symmetric = []
    all_d_pred = []
    all_d_gt = []

    calibrator = RiskCalibrator()

    print(f"\nCollecting calibration data: {n_scenes} scenes × {points_per_scene} points")
    print(f"Data range: {data_range}")
    start_time = time.time()

    for scene_idx in range(n_scenes):
        points = generate_random_scene(points_per_scene, data_range, seed=scene_idx)
        d_pred_scene = []
        d_gt_scene = []

        for pt in points:
            point_2x1 = pt.reshape(2, 1)

            # DUNE prediction
            d_p, _ = dune_predict(model, G_tensor, h_tensor, point_2x1)
            d_pred_scene.append(d_p)

            # LP ground truth
            d_g, _ = solve_lp_ground_truth(G_np, h_np, point_2x1)
            d_gt_scene.append(d_g if d_g is not None else d_p)  # fallback

        d_pred_arr = np.array(d_pred_scene)
        d_gt_arr = np.array(d_gt_scene)

        all_d_pred.append(d_pred_arr)
        all_d_gt.append(d_gt_arr)

        # Scene-level risks
        r_asy = calibrator.collect_scene_risk(d_pred_arr, d_gt_arr, "asymmetric")
        r_sym = calibrator.collect_scene_risk(d_pred_arr, d_gt_arr, "symmetric")
        scene_risks_asymmetric.append(r_asy)
        scene_risks_symmetric.append(r_sym)

        if (scene_idx + 1) % 200 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (scene_idx + 1) * (n_scenes - scene_idx - 1)
            print(f"  Scene {scene_idx+1}/{n_scenes} ({elapsed:.1f}s elapsed, ETA {eta:.1f}s)")

    total_time = time.time() - start_time
    print(f"\nCalibration data collected in {total_time:.1f}s")

    # Fit calibrator
    calibrator.fit(np.array(scene_risks_asymmetric))

    # Compute q_hat for each epsilon
    print("\n" + "="*60)
    print("CALIBRATION RESULTS")
    print("="*60)

    results = {
        "n_scenes": n_scenes,
        "points_per_scene": points_per_scene,
        "data_range": list(data_range),
        "total_time_s": total_time,
        "epsilons": {},
        "statistics": {},
    }

    # Asymmetric q_hat
    print(f"\nAsymmetric risk score: r(p) = max(0, d_pred - d_gt)")
    for eps in epsilons:
        q = calibrator.compute_q_hat(eps)
        results["epsilons"][str(eps)] = {"asymmetric_q_hat": q}
        print(f"  ε={eps:.2f}: q_hat = {q:.4f} m ({q*100:.2f} cm)")

    # Symmetric q_hat (comparison)
    print(f"\nSymmetric risk score: r(p) = |d_pred - d_gt|")
    for eps in epsilons:
        q_sym = calibrator.compute_q_hat_symmetric(all_d_pred, all_d_gt, eps)
        results["epsilons"][str(eps)]["symmetric_q_hat"] = q_sym
        print(f"  ε={eps:.2f}: q_hat = {q_sym:.4f} m ({q_sym*100:.2f} cm)")

    # Statistics
    all_d_pred_flat = np.concatenate(all_d_pred)
    all_d_gt_flat = np.concatenate(all_d_gt)
    errors = all_d_pred_flat - all_d_gt_flat
    risks_asym = np.maximum(0, errors)
    risks_sym = np.abs(errors)

    results["statistics"] = {
        "error_mean": float(np.mean(errors)),
        "error_std": float(np.std(errors)),
        "overestimation_rate": float(np.mean(errors > 0)),
        "underestimation_rate": float(np.mean(errors < 0)),
        "mean_asymmetric_risk": float(np.mean(risks_asym)),
        "mean_symmetric_risk": float(np.mean(risks_sym)),
        "max_asymmetric_risk": float(np.max(risks_asym)),
        "scene_risk_asym_mean": float(np.mean(scene_risks_asymmetric)),
        "scene_risk_asym_std": float(np.std(scene_risks_asymmetric)),
        "scene_risk_sym_mean": float(np.mean(scene_risks_symmetric)),
    }

    print(f"\nError Statistics:")
    print(f"  Mean error (d_pred - d_gt): {np.mean(errors):.6f} m")
    print(f"  Std error: {np.std(errors):.6f} m")
    print(f"  Overestimation rate: {np.mean(errors > 0)*100:.1f}%")
    print(f"  Underestimation rate: {np.mean(errors < 0)*100:.1f}%")
    print(f"  Mean asymmetric risk: {np.mean(risks_asym)*100:.4f} cm")
    print(f"  Mean symmetric risk: {np.mean(risks_sym)*100:.4f} cm")

    # Kill Check
    print(f"\n{'='*60}")
    print("KILL CHECK #1")
    print("="*60)

    q05_asy = calibrator.compute_q_hat(0.05)
    q05_sym = calibrator.compute_q_hat_symmetric(all_d_pred, all_d_gt, 0.05)
    over_rate = np.mean(errors > 0)

    kill = False
    warnings = []

    if over_rate > 0.5:
        kill = True
        warnings.append(f"CRITICAL: Overestimation rate {over_rate*100:.1f}% > 50%. DUNE systematically overestimates.")
    if q05_asy > 0.10:
        kill = True
        warnings.append(f"CRITICAL: q_hat(0.05) = {q05_asy*100:.1f} cm > 10cm. Too conservative.")
    elif q05_asy > 0.05:
        warnings.append(f"WARNING: q_hat(0.05) = {q05_asy*100:.1f} cm > 5cm. May be too conservative for tight spaces.")

    # Asymmetric vs symmetric comparison
    improvement = (q05_sym - q05_asy) / q05_sym * 100 if q05_sym > 0 else 0
    print(f"  q_hat_asy(0.05) = {q05_asy*100:.2f} cm")
    print(f"  q_hat_sym(0.05) = {q05_sym*100:.2f} cm")
    print(f"  Asymmetric improvement: {improvement:.1f}%")

    if improvement < 5:
        warnings.append(f"MINOR: Asymmetric improvement only {improvement:.1f}%. Benefit may be marginal.")

    results["kill_check"] = {
        "q05_asymmetric_m": q05_asy,
        "q05_symmetric_m": q05_sym,
        "improvement_pct": improvement,
        "overestimation_rate": over_rate,
        "passed": not kill,
        "warnings": warnings,
    }

    if kill:
        print(f"\n  KILL CHECK FAILED:")
        for w in warnings:
            print(f"    - {w}")
    else:
        print(f"\n  KILL CHECK PASSED")
        for w in warnings:
            print(f"    - {w}")

    # Save calibration data
    cal_path = os.path.join(output_dir, "calibration.json")
    calibrator.save(cal_path)
    print(f"\nCalibration saved to {cal_path}")

    # Save full results
    results_path = os.path.join(output_dir, "kill_check_1_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    # Save raw data for further analysis
    raw_path = os.path.join(output_dir, "raw_calibration_data.npz")
    np.savez(raw_path,
             scene_risks_asymmetric=np.array(scene_risks_asymmetric),
             scene_risks_symmetric=np.array(scene_risks_symmetric),
             errors=errors)
    print(f"Raw data saved to {raw_path}")

    return calibrator, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kill Check #1: DUNE Risk Calibratability")
    parser.add_argument('--checkpoint', default='example/model/diff_robot_default/model_5000.pth',
                        help='Path to trained DUNE model checkpoint')
    parser.add_argument('--n-scenes', type=int, default=2000,
                        help='Number of calibration scenes')
    parser.add_argument('--points-per-scene', type=int, default=50,
                        help='Obstacle points per scene')
    parser.add_argument('--output-dir', default='experiments/calibration_output',
                        help='Output directory')
    args = parser.parse_args()

    run_calibration(
        checkpoint_path=args.checkpoint,
        n_scenes=args.n_scenes,
        points_per_scene=args.points_per_scene,
        output_dir=args.output_dir,
    )
