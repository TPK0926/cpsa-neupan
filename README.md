# CPSA-NeuPAN

This repository contains the public source code for **Embodied Safe Navigation via Per-Point Conformal Safety Allocation**.

CPSA extends NeuPAN's differentiable point-navigation pipeline with per-point conformal safety allocation. Instead of applying one uniform safety margin to every LiDAR obstacle point, CPSA predicts geometry-aware safety margins from local obstacle structure, temporal variation, passage geometry, and global scene context. These margins are injected into the point-wise navigation constraints, improving the balance between safety and conservatism.

This repository is intentionally source-code focused. Experiment batches, generated results, robot deployment files, trained checkpoints, and paper build artifacts are not included.

## Source Layout

- `neupan/neupan.py`: high-level planner interface and CPSA integration path.
- `neupan/blocks/`: differentiable navigation blocks, including point navigation, distance estimation, optimization, and CPSA risk modeling.
- `neupan/blocks/cpsa_risk_net.py`: CPSA feature extraction, risk network, asymmetric loss, calibration helper, and online adapter.
- `neupan/risk_calibration/`: conformal calibration and risk-budget allocation utilities.
- `neupan/robot/`: robot kinematic models used by the planner.
- `neupan/baselines/`: baseline safety-allocation and uncertainty modules retained for source-level comparison.
- `gazebo_sim/`: Gazebo integration reference code, launch files, models, and worlds.

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/TPK0926/cpsa-neupan.git
cd cpsa-neupan
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

For CUDA acceleration in learning components, install a PyTorch build matching your CUDA driver before installing this package.

## Minimal API

```python
from neupan import neupan
from neupan.blocks.cpsa_risk_net import CPSARiskNet
from neupan.risk_calibration.risk_budget import RiskBudgetAllocator

risk_net = CPSARiskNet()
allocator = RiskBudgetAllocator(strategy="cpsa")
```

Planner construction requires a robot configuration, path generator configuration, optimization block configuration, and trained distance/safety models matching the target platform. Those deployment assets are intentionally not bundled in this source-only release.

## Gazebo Integration

`gazebo_sim/` keeps the ROS2/Gazebo adapter source, simulation worlds, and model files used during development. These files are included as integration reference code. Running them requires a ROS2/Gazebo workspace and external planner/checkpoint assets configured for the target robot.

## License

This code is released under the GPL-3.0 license, following the upstream NeuPAN license.

## Acknowledgement

This project builds on NeuPAN. Please also cite the original NeuPAN work when using this repository.
