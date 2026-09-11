# Training CPSA

This folder contains the public training entry point for the CPSA risk network.

The repository does not include raw navigation logs, generated datasets, or trained checkpoints. Prepare a point-level torch `.pt` dataset externally, then run:

```bash
python training/train_cpsa.py \
  --data /path/to/cpsa_point_data.pt \
  --output checkpoints/cpsa_v4.pth
```

The dataset must contain `d_pred`, `d_gt`, and `residual`. Optional fields such as `point_radius`, `angle_to_robot`, `local_density`, `noise_std`, temporal deltas, passage features, teacher margins, and episode outcomes are used when available.
