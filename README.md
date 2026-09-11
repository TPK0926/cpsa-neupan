# CPSA-NeuPAN

This repository contains the source code for **Embodied Safe Navigation via Per-Point Conformal Safety Allocation**.

CPSA builds on NeuPAN's differentiable MPC-style point navigation pipeline and adds per-point conformal safety allocation. Instead of applying one uniform safety margin to every LiDAR obstacle point, CPSA predicts geometry-aware margins from local point features, temporal changes, passage structure, and global scene context. The resulting margins are injected into the point-wise navigation constraints, preserving clearance where needed while reducing unnecessary conservatism in free space.

## Main Components

- `neupan/`: core NeuPAN planner with CPSA integration.
- `neupan/blocks/cpsa_risk_net.py`: CPSA risk network, feature extraction, asymmetric loss, calibration, and online adapter.
- `neupan/risk_calibration/`: conformal calibration and risk-budget allocation utilities.
- `experiments/`: simulation experiment runners, ablation scripts, calibration scripts, and CPSA training pipeline.
- `example/`: IR-SIM navigation scenarios and DUNE checkpoints for differential, Ackermann, polygon, and TurtleBot3-style robots.
- `gazebo_sim/`: Gazebo/TurtleBot3 simulation helpers used for robot-level validation.
- `real_robot/`: Limo real-robot deployment scripts and log extraction utilities.

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/TPK0926/cpsa-neupan.git
cd cpsa-neupan
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[irsim]"
```

For CUDA training, install a PyTorch build matching your CUDA driver before installing this package.

## Quick Start

Run an IR-SIM example with the base NeuPAN planner:

```bash
python example/run_exp.py -e corridor -d diff
```

Run CPSA in a single benchmark configuration:

```bash
python experiments/run_config.py corridor 2 cpsa_v4_full 10
```

Run the visual test:

```bash
python experiments/visual_test.py --env corridor --noise 2 --method cpsa_v4
```

## CPSA Training

The released repository includes a compact CPSA checkpoint at:

```text
experiments/cp_head_output/cpsa_v4_universal.pth
```

To collect new data and retrain:

```bash
python experiments/train_cpsa_v4.py collect --noise_levels 0.0 0.02 0.03 0.05 --n_episodes 50
python experiments/train_cpsa_v4.py train --epochs 300
python experiments/train_cpsa_v4.py validate --noise_levels 0.0 0.02 0.05 --n_episodes 30
```

Large raw training tensors are intentionally not included in the public release. Regenerate them with the `collect` command when needed.

## Reproducing Paper Experiments

```bash
bash experiments/run_all.sh
python experiments/run_full_matrix.py
python experiments/run_ablation.py
```

See `docs/REPRODUCTION.md` for the cleaned release layout and expected workflow.

## Real-Robot Deployment

The `real_robot/` folder contains ROS-side integration code used for Limo deployment. Typical method names are:

- `vanilla`: original NeuPAN behavior.
- `cp_global`: scalar conformal safety margin.
- `cpsa_v4`: per-point CPSA safety allocation.

Update robot-specific ROS topics, frame names, and checkpoint paths before running on a new platform.

## License

This code is released under the GPL-3.0 license, following the upstream NeuPAN license.

## Acknowledgement

This project is based on NeuPAN. Please also cite the original NeuPAN work when using this repository.
