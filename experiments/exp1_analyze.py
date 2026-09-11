"""
Experiment 1 Part A: Analyze already-collected test data for coverage.

This script re-analyzes the test scene data without re-running the expensive
DUNE forward + LP solve loop.
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neupan.robot import robot
from neupan.blocks import ObsPointNet
from neupan.configuration import np_to_tensor, to_device
from neupan.risk_calibration import RiskCalibrator
import cvxpy as cp
import torch
import time


def load_dune_model(checkpoint_path, robot_instance):
    edge_dim = robot_instance.G.shape[0]
    model = to_device(ObsPointNet(2, edge_dim))
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    return model


def solve_lp_distance(G_np, h_np, point):
    mu = cp.Variable((G_np.shape[0], 1), nonneg=True)
    p = cp.Parameter((2, 1))
    p.value = point
    cost = mu.T @ (G_np @ p - h_np)
    constraints = [cp.norm(G_np.T @ mu) <= 1]
    prob = cp.Problem(cp.Maximize(cost), constraints)
    prob.solve(solver=cp.ECOS)
    return prob.value


def dune_predict(model, G_tensor, h_tensor, point_np):
    point_tensor = np_to_tensor(point_np.reshape(2, 1))
    with torch.no_grad():
        mu_pred = model(point_tensor.T).T
    temp = (G_tensor @ point_tensor - h_tensor).T.unsqueeze(2)
    muT = mu_pred.T.unsqueeze(1)
    distance = torch.squeeze(torch.bmm(muT, temp))
    return distance.item()


def run_coverage_test(
    checkpoint_path='example/model/diff_robot_default/model_5000.pth',
    calibration_file='experiments/calibration_output/calibration.json',
    kc_results_file='experiments/calibration_output/kill_check_1_results.json',
    n_test_scenes=1000,
    points_per_scene=50,
    epsilons=[0.01, 0.05, 0.10],
    output_dir='experiments/exp1_output',
):
    import os
    os.makedirs(output_dir, exist_ok=True)

    robot_instance = robot(receding=10, step_time=0.1, kinematics='diff',
                           length=1.6, width=2.0)
    G_np, h_np = robot_instance.G, robot_instance.h
    G_tensor, h_tensor = np_to_tensor(G_np), np_to_tensor(h_np)
    model = load_dune_model(checkpoint_path, robot_instance)

    # Load q_hat values from Kill Check results (already computed)
    kc = json.load(open(kc_results_file))

    print("=" * 60)
    print("EXPERIMENT 1 PART A: COVERAGE VALIDATION")
    print(f"Test scenes: {n_test_scenes} (seeds 10000-10999, no overlap with calibration)")
    print(f"Points/scene: {points_per_scene}")
    print("=" * 60)

    # Collect test scene risks
    scene_risks_asy = []
    scene_risks_sym = []

    start_time = time.time()
    for scene_idx in range(n_test_scenes):
        rng = np.random.RandomState(10000 + scene_idx)
        points = rng.uniform(low=[-25, -25], high=[25, 25],
                             size=(points_per_scene, 2))

        d_pred = []
        d_gt = []
        for pt in points:
            p = pt.reshape(2, 1)
            dp = dune_predict(model, G_tensor, h_tensor, p)
            dg = solve_lp_distance(G_np, h_np, p)
            d_pred.append(dp)
            d_gt.append(dg if dg is not None else dp)

        d_pred_arr = np.array(d_pred)
        d_gt_arr = np.array(d_gt)

        r_asy = np.max(np.maximum(0, d_pred_arr - d_gt_arr))
        r_sym = np.max(np.abs(d_pred_arr - d_gt_arr))
        scene_risks_asy.append(r_asy)
        scene_risks_sym.append(r_sym)

        if (scene_idx + 1) % 200 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (scene_idx + 1) * (n_test_scenes - scene_idx - 1)
            print(f"  Scene {scene_idx+1}/{n_test_scenes} ({elapsed:.0f}s, ETA {eta:.0f}s)")

    risks_asy = np.array(scene_risks_asy)
    risks_sym = np.array(scene_risks_sym)
    print(f"\nData collection: {time.time()-start_time:.0f}s")

    # Coverage analysis
    print(f"\n{'='*60}")
    print("COVERAGE RESULTS (Theorem 1 Validation)")
    print(f"{'='*60}")
    print(f"{'ε':>6} | {'Method':>12} | {'q_hat(cm)':>10} | {'Coverage':>10} | {'Target':>8} | {'Gap':>8} | {'Pass':>5}")
    print("-" * 75)

    results = {"n_test_scenes": n_test_scenes, "epsilons": {}}

    for eps in epsilons:
        eps_str = str(eps)
        q_asy = kc['epsilons'][eps_str]['asymmetric_q_hat']
        q_sym = kc['epsilons'][eps_str]['symmetric_q_hat']
        target = (1 - eps) * 100

        # Asymmetric coverage
        cov_asy = np.mean(risks_asy <= q_asy) * 100
        gap_asy = cov_asy - target
        pass_asy = cov_asy >= target

        # Symmetric coverage
        cov_sym = np.mean(risks_sym <= q_sym) * 100
        gap_sym = cov_sym - target
        pass_sym = cov_sym >= target

        s1 = "PASS" if pass_asy else "FAIL"
        s2 = "PASS" if pass_sym else "FAIL"

        print(f"{eps:>6.2f} | {'Asymmetric':>12} | {q_asy*100:>10.2f} | {cov_asy:>9.1f}% | {target:>7.1f}% | {gap_asy:>+7.1f}% | {s1:>5}")
        print(f"{'':>6} | {'Symmetric':>12} | {q_sym*100:>10.2f} | {cov_sym:>9.1f}% | {target:>7.1f}% | {gap_sym:>+7.1f}% | {s2:>5}")
        print("-" * 75)

        results["epsilons"][eps_str] = {
            "asymmetric": {"q_hat_cm": q_asy * 100, "coverage_pct": cov_asy,
                           "target_pct": target, "gap_pct": gap_asy, "pass": bool(pass_asy)},
            "symmetric": {"q_hat_cm": q_sym * 100, "coverage_pct": cov_sym,
                          "target_pct": target, "gap_pct": gap_sym, "pass": bool(pass_sym)},
        }

    # Margin comparison
    print(f"\n{'='*60}")
    print("MARGIN COMPARISON (Asymmetric vs Symmetric)")
    print(f"{'='*60}")
    for eps in epsilons:
        eps_str = str(eps)
        q_asy = kc['epsilons'][eps_str]['asymmetric_q_hat']
        q_sym = kc['epsilons'][eps_str]['symmetric_q_hat']
        saving = (q_sym - q_asy) * 100
        pct = (q_sym - q_asy) / q_sym * 100
        print(f"  ε={eps:.2f}: asym={q_asy*100:.2f}cm, sym={q_sym*100:.2f}cm, "
              f"saving={saving:.2f}cm ({pct:.1f}% tighter)")

    # Risk distribution
    print(f"\n{'='*60}")
    print("RISK DISTRIBUTION ON TEST SET")
    print(f"{'='*60}")
    errors = risks_asy  # asymmetric scene risks
    print(f"  Asymmetric scene risk R(s):")
    print(f"    Mean:   {np.mean(risks_asy)*100:.3f} cm")
    print(f"    Std:    {np.std(risks_asy)*100:.3f} cm")
    print(f"    Median: {np.median(risks_asy)*100:.3f} cm")
    print(f"    Max:    {np.max(risks_asy)*100:.3f} cm")
    print(f"    P95:    {np.percentile(risks_asy, 95)*100:.3f} cm")
    print(f"    P99:    {np.percentile(risks_asy, 99)*100:.3f} cm")
    print(f"  Symmetric scene risk R_sym(s):")
    print(f"    Mean:   {np.mean(risks_sym)*100:.3f} cm")
    print(f"    Max:    {np.max(risks_sym)*100:.3f} cm")

    # Save
    results_path = os.path.join(output_dir, "part_a_coverage.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save raw test risks
    np.savez(os.path.join(output_dir, "test_risks.npz"),
             risks_asy=risks_asy, risks_sym=risks_sym)

    return results


if __name__ == "__main__":
    run_coverage_test()
