# Release Manifest

This cleaned source tree was prepared from the original mixed workspace for GitHub release.

## Included

- `neupan/`: planner, optimization blocks, robot models, baseline methods, and CPSA integration.
- `neupan/blocks/cpsa_risk_net.py`: CPSA risk network and online adapter.
- `neupan/risk_calibration/`: conformal calibration and risk allocation.
- `example/`: IR-SIM navigation examples and compact final DUNE checkpoints.
- `experiments/`: experiment, calibration, ablation, and CPSA training scripts.
- `experiments/cp_head_output/cpsa_v4_universal.pth`: compact CPSA checkpoint used by public scripts.
- `gazebo_sim/`: Gazebo worlds, model files, and validation nodes.
- `real_robot/`: Limo real-robot deployment utilities.
- `LICENSE`, `README.md`, `pyproject.toml`, `requirements.txt`, `.gitignore`.

## Excluded

- ROS build artifacts: `build/`, `install/`, `log/`.
- Python caches: `__pycache__/`, `.pytest_cache/`, `*.pyc`.
- Large raw training tensors such as `experiments/cp_training_data/v4_combined_data.pt`.
- Generated experiment output folders and temporary result logs.
- Paper build outputs and LaTeX intermediate files.
- Patent drafts, private notes, Word files, compressed archives, and unrelated hardware documents.
- `.claude/` tools and local automation assets.
- Unrelated Dexterous hand and TurtleBot3 source trees from the original mixed workspace root.

## Naming Cleanup

The method naming was aligned with the paper terminology. Public code now uses:

- `cpsa_v4_*` method keys.
- `CPSARiskNet`, `CPSAFeatureExtractor`, `CPSAAsymmetricLoss`, `CPSACalibrator`, `OnlineCPSAAdapter`.
- `neupan/blocks/cpsa_risk_net.py`.
- `experiments/train_cpsa_v4.py`.
- `experiments/cp_head_output/cpsa_v4_universal.pth`.

The original mixed workspace was not modified by this cleanup.
