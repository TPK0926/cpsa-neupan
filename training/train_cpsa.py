#!/usr/bin/env python3
"""Train the CPSA risk network from prepared point-level navigation data.

The expected input is a torch ``.pt`` file containing point-wise tensors from
navigation rollouts. The public repository does not include raw data, robot
logs, or trained checkpoints.

Required keys:
    d_pred, d_gt, residual

Optional keys used when available:
    point_radius, angle_to_robot, local_density, speed, noise_std, env_type,
    d_pred_delta, radius_delta, lateral_margin, passage_ratio, teacher_tau,
    is_approaching, episode_outcome
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from neupan.blocks.cpsa_risk_net import CPSAAsymmetricLoss, CPSACalibrator, CPSARiskNet


class NavigationPointDataset(Dataset):
    """Point-level CPSA dataset with the 17-D feature definition used by CPSA."""

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        data = torch.load(self.data_path, map_location="cpu", weights_only=False)

        self.d_pred = self._required(data, "d_pred").float()
        self.d_gt = self._required(data, "d_gt").float()
        self.residual = self._required(data, "residual").float()
        n = len(self.d_pred)

        self.radius = self._optional(data, "point_radius", torch.zeros(n)).float()
        self.angle = self._optional(data, "angle_to_robot", torch.zeros(n)).float()
        self.density = self._optional(data, "local_density", torch.ones(n) * 5.0).float()
        self.speed = self._optional(data, "speed", torch.zeros(n)).float()
        self.noise_std = self._optional(data, "noise_std", torch.zeros(n)).float()
        self.env_type = self._optional(data, "env_type", torch.zeros(n, dtype=torch.long)).long()
        self.d_pred_delta = self._optional(data, "d_pred_delta", torch.zeros(n)).float()
        self.radius_delta = self._optional(data, "radius_delta", torch.zeros(n)).float()
        self.lateral_margin = self._optional(data, "lateral_margin", torch.ones(n) * 5.0).float()
        self.passage_ratio = self._optional(data, "passage_ratio", torch.ones(n) * 3.0).float()
        self.teacher_tau = data.get("teacher_tau")
        if self.teacher_tau is not None:
            self.teacher_tau = self.teacher_tau.float()
        self.is_approaching = self._optional(data, "is_approaching", torch.zeros(n, dtype=torch.bool)).bool()
        self.outcomes = data.get("episode_outcome", ["unknown"] * n)

    @staticmethod
    def _required(data: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
        if key not in data:
            raise KeyError(f"Training data is missing required key: {key}")
        return data[key]

    @staticmethod
    def _optional(data: Dict[str, torch.Tensor], key: str, default: torch.Tensor) -> torch.Tensor:
        value = data.get(key, default)
        return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)

    def __len__(self) -> int:
        return len(self.d_pred)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = {
            "features": self.build_features([index]).squeeze(0),
            "d_pred": self.d_pred[index],
            "d_gt": self.d_gt[index],
            "noise_std": self.noise_std[index],
            "lateral_margin": self.lateral_margin[index],
            "is_approaching": self.is_approaching[index].float(),
        }
        if self.teacher_tau is not None:
            item["teacher_tau"] = self.teacher_tau[index]
        return item

    def build_features(self, indices: Iterable[int] | torch.Tensor) -> torch.Tensor:
        idx = torch.as_tensor(list(indices), dtype=torch.long) if not isinstance(indices, torch.Tensor) else indices.long()
        eps = 1e-6

        d_pred = self.d_pred[idx]
        radius = self.radius[idx]
        angle = self.angle[idx]
        proximity_ratio = torch.abs(d_pred) / (radius + eps)
        density = self.density[idx]

        features = torch.stack(
            [
                d_pred,
                radius,
                angle,
                proximity_ratio,
                d_pred.square(),
                torch.cos(angle),
                torch.sin(angle),
                density,
                torch.clamp(density / 5.0, 0.0, 1.0),
                self.speed[idx],
                torch.zeros_like(d_pred),
                self.noise_std[idx],
                self.d_pred_delta[idx],
                self.radius_delta[idx],
                self.lateral_margin[idx],
                self.passage_ratio[idx],
                self.env_type[idx].float() / 4.0,
            ],
            dim=1,
        )
        return torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)

    def build_all_features(self) -> torch.Tensor:
        return self.build_features(torch.arange(len(self)))

    def sample_weights(self) -> torch.Tensor:
        noise = self.noise_std.numpy()
        unique, counts = np.unique(noise, return_counts=True)
        base = {level: len(noise) / (len(unique) * count) for level, count in zip(unique, counts)}
        weights = torch.tensor([base[level] for level in noise], dtype=torch.float32)
        for i, outcome in enumerate(self.outcomes):
            if outcome in ("collision", "timeout"):
                weights[i] *= 5.0
        return weights


class FeatureNoiseInjector:
    """Gaussian feature perturbation used as lightweight regularization."""

    def __init__(self, noise_ratio: float = 0.1):
        self.noise_ratio = noise_ratio

    def __call__(self, features: torch.Tensor, feature_std: torch.Tensor) -> torch.Tensor:
        return features + torch.randn_like(features) * feature_std.unsqueeze(0) * self.noise_ratio


def split_indices(n: int, seed: int, ratios: Tuple[float, float, float]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("split ratios must sum to 1.0")
    rng = np.random.default_rng(seed)
    perm = torch.as_tensor(rng.permutation(n), dtype=torch.long)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    return perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :]


def take(tensor: torch.Tensor | None, idx: torch.Tensor, device: torch.device) -> torch.Tensor | None:
    return None if tensor is None else tensor[idx].to(device)


def train(args: argparse.Namespace) -> Path:
    dataset = NavigationPointDataset(args.data)
    train_idx, val_idx, cal_idx = split_indices(len(dataset), args.seed, args.split)

    features = dataset.build_all_features()
    feat_mean = features[train_idx].mean(dim=0)
    feat_std = features[train_idx].std(dim=0).clamp(min=1e-6)
    features = (features - feat_mean) / feat_std

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net = CPSARiskNet(feature_dim=17, hidden_dims=tuple(args.hidden_dims)).to(device)
    loss_fn = CPSAAsymmetricLoss(
        coverage_weight=args.coverage_weight,
        collision_prob_weight=args.collision_prob_weight,
        narrow_passage_weight=args.narrow_passage_weight,
        ranking_weight=args.ranking_weight,
        smoothness_weight=args.smoothness_weight,
        reg_weight=args.reg_weight,
    )
    injector = FeatureNoiseInjector(args.feature_noise_ratio)

    train_features = features[train_idx].to(device)
    train_d_pred = dataset.d_pred[train_idx].to(device)
    train_d_gt = dataset.d_gt[train_idx].to(device)
    train_noise_std = dataset.noise_std[train_idx].to(device)
    train_lateral_margin = dataset.lateral_margin[train_idx].to(device)
    train_is_approaching = dataset.is_approaching[train_idx].float().to(device)
    train_teacher_tau = take(dataset.teacher_tau, train_idx, device)

    val_features = features[val_idx].to(device)
    val_d_pred = dataset.d_pred[val_idx].to(device)
    val_d_gt = dataset.d_gt[val_idx].to(device)
    val_noise_std = dataset.noise_std[val_idx].to(device)
    val_lateral_margin = dataset.lateral_margin[val_idx].to(device)
    val_is_approaching = dataset.is_approaching[val_idx].float().to(device)

    cal_features = features[cal_idx].to(device)
    cal_d_pred = dataset.d_pred[cal_idx]
    cal_d_gt = dataset.d_gt[cal_idx]
    cal_noise_std = dataset.noise_std[cal_idx]

    weights = dataset.sample_weights()[train_idx].to(device)
    feature_std_device = feat_std.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    log = {"train_loss": [], "val_loss": [], "tau_mean": [], "tau_std": []}
    best_state = None
    best_val = float("inf")
    n_train = len(train_idx)
    n_batches = max(1, (n_train + args.batch_size - 1) // args.batch_size)

    if train_teacher_tau is not None and args.teacher_epochs > 0:
        mse = nn.MSELoss()
        teacher_optimizer = optim.Adam(net.parameters(), lr=args.teacher_lr)
        for epoch in range(args.teacher_epochs):
            sample_idx = torch.multinomial(weights, n_train, replacement=True)
            losses = []
            for batch in sample_idx.split(args.batch_size):
                x = injector(train_features[batch], feature_std_device)
                target = train_teacher_tau[batch]
                ns = train_noise_std[batch]
                teacher_optimizer.zero_grad(set_to_none=True)
                tau = net(x, noise_std=ns)
                loss = mse(tau, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                teacher_optimizer.step()
                losses.append(loss.item())
            print(f"teacher_epoch={epoch + 1} mse={np.mean(losses):.6f}")

    for epoch in range(args.epochs):
        net.train()
        sample_idx = torch.multinomial(weights, n_train, replacement=True)
        batch_losses = []
        tau_means = []
        tau_stds = []

        for batch in sample_idx.split(args.batch_size):
            x = injector(train_features[batch], feature_std_device)
            optimizer.zero_grad(set_to_none=True)
            tau = net(x, noise_std=train_noise_std[batch])
            loss_dict = loss_fn(
                tau,
                train_d_pred[batch],
                train_d_gt[batch],
                model=net,
                noise_std=train_noise_std[batch],
                lateral_margins=train_lateral_margin[batch],
                is_approaching=train_is_approaching[batch],
            )
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()
            batch_losses.append(loss_dict["total"].item())
            tau_means.append(loss_dict["tau_mean"].item())
            tau_stds.append(loss_dict["tau_std"].item())

        scheduler.step()
        net.eval()
        with torch.no_grad():
            val_tau = net(val_features, noise_std=val_noise_std)
            val_loss = loss_fn(
                val_tau,
                val_d_pred,
                val_d_gt,
                model=net,
                noise_std=val_noise_std,
                lateral_margins=val_lateral_margin,
                is_approaching=val_is_approaching,
            )

        train_loss = float(np.mean(batch_losses))
        val_total = float(val_loss["total"].item())
        log["train_loss"].append(train_loss)
        log["val_loss"].append(val_total)
        log["tau_mean"].append(float(np.mean(tau_means)))
        log["tau_std"].append(float(np.mean(tau_stds)))

        if val_total < best_val:
            best_val = val_total
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        if epoch == 0 or (epoch + 1) % args.log_every == 0:
            print(
                f"epoch={epoch + 1}/{args.epochs} train={train_loss:.5f} "
                f"val={val_total:.5f} tau_mean={log['tau_mean'][-1]:.5f}"
            )

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    net.load_state_dict(best_state)
    net.to(device).eval()

    with torch.no_grad():
        cal_tau = net(cal_features).detach().cpu().numpy()
    calibrator = CPSACalibrator(epsilon=args.epsilon, window_size=args.calibration_window)
    q_star = calibrator.compute_q_star(cal_tau, cal_d_pred.numpy(), cal_d_gt.numpy(), d_min=args.d_min)
    q_star_conditional = calibrator.compute_q_star_conditional(
        cal_tau, cal_d_pred.numpy(), cal_d_gt.numpy(), cal_noise_std.numpy()
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": best_state,
        "config": {
            "version": 4,
            "feature_dim": 17,
            "hidden_dims": list(args.hidden_dims),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_cal": len(cal_idx),
        },
        "q_star": float(q_star),
        "q_star_conditional": {float(k): float(v) for k, v in q_star_conditional.items()},
        "feature_mean": feat_mean,
        "feature_std": feat_std,
        "calibrator_epsilon": args.epsilon,
        "training_log": log,
    }
    torch.save(checkpoint, output)

    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "best_val_loss": best_val,
                "q_star": float(q_star),
                "q_star_conditional": {str(k): float(v) for k, v in q_star_conditional.items()},
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_cal": len(cal_idx),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved_checkpoint={output}")
    print(f"saved_metrics={metrics_path}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CPSA risk network from point-level data")
    parser.add_argument("--data", required=True, help="Path to prepared torch .pt training data")
    parser.add_argument("--output", default="checkpoints/cpsa_v4.pth", help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--teacher-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--teacher-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--feature-noise-ratio", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--d-min", type=float, default=0.1)
    parser.add_argument("--calibration-window", type=int, default=100)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 64, 32])
    parser.add_argument("--split", type=float, nargs=3, default=[0.7, 0.1, 0.2], metavar=("TRAIN", "VAL", "CAL"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
