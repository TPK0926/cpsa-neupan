# Reproduction Guide

This guide describes the intended workflow for the cleaned public release.

## Environment

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[irsim]"
```

The optimization layers run primarily on CPU through CVXPY/CVXPYLayers. CUDA is useful for training the CPSA risk network, but it is not required for running the released IR-SIM examples.

## Method Keys

- `vanilla`: NeuPAN without conformal safety allocation.
- `cp_global`: one scalar conformal margin.
- `cp_dw`: distance-weighted conformal margin.
- `cpsa_v4_full`: full CPSA per-point safety allocation.
- `cpsa_v4_noTemporal`: CPSA without temporal point features.
- `cpsa_v4_noPassage`: CPSA without passage-width features.
- `cpsa_v4_noGlobalCtx`: CPSA without global scene pooling.
- `cpsa_v4_staticTau`: CPSA with static maximum margin.
- `cpsa_v4_staticQstar`: CPSA with static conformal correction.
- `cpsa_v4_online`: CPSA with deployment-time online adapter.

## Single Run

```bash
python experiments/run_config.py corridor 2 cpsa_v4_full 10
```

Arguments are:

```text
<environment> <noise_cm> <method> [n_episodes]
```

Available environments include `corridor`, `convex_obs`, `pf_obs`, `dyna_obs`, `non_obs`, and `mixed_corridor`.

## Full Matrix

```bash
python experiments/run_full_matrix.py
```

## Ablation

```bash
python experiments/run_ablation.py
```

## Training CPSA

```bash
python experiments/train_cpsa_v4.py collect --noise_levels 0.0 0.02 0.03 0.05 --n_episodes 50
python experiments/train_cpsa_v4.py train --epochs 300
python experiments/train_cpsa_v4.py validate --noise_levels 0.0 0.02 0.05 --n_episodes 30
```

The released checkpoint is stored at:

```text
experiments/cp_head_output/cpsa_v4_universal.pth
```

Raw training tensors are excluded from this release and should be regenerated with the `collect` command.

## Gazebo and Real Robot

`gazebo_sim/` contains the Gazebo validation scripts and worlds. `real_robot/` contains the Limo deployment scripts. These folders require a ROS/Gazebo environment and platform-specific topic/frame configuration.
