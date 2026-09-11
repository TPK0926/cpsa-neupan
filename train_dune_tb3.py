#!/usr/bin/env python3
"""Standalone DUNE training for TB3 robot geometry (0.14x0.14m).

Trains a DUNE distance prediction model on GPU without loading the full
NeuPAN planner (avoids CVXPY/CUDA conflict). Saves checkpoint to
example/model/tb3_default/.

Usage:
  python train_dune_tb3.py
"""

import sys, os, pickle, time
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torch.optim import Adam
import cvxpy as cp

# --- Path setup ---
NEUPAN_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, NEUPAN_ROOT)

from neupan.blocks import ObsPointNet
from neupan.configuration import np_to_tensor

# --- Robot geometry for TB3 (0.14m x 0.14m) ---
# Use NeuPAN's exact G/h generation (gen_inequal_from_vertex)
from neupan.util import gen_inequal_from_vertex

def make_G_h(length, width):
    """Generate G, h matrices matching NeuPAN's robot class."""
    wheelbase = 0.0
    start_x = -(length - wheelbase) / 2
    start_y = -width / 2
    vertices = np.array([
        [start_x, start_x + length, start_x + length, start_x],
        [start_y, start_y, start_y + width, start_y + width],
    ], dtype=np.float64)
    return gen_inequal_from_vertex(vertices)

G, h = make_G_h(0.14, 0.14)
print(f"G:\n{G}")
print(f"h: {h.T}")

# --- Dataset: random points -> convex optimization labels ---
class PointDataset(Dataset):
    def __init__(self, inputs, labels, distances):
        self.inputs = inputs
        self.labels = labels
        self.distances = distances

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx], self.distances[idx]


def generate_data(data_size=100000, data_range=[-25, -25, 25, 25]):
    """Generate training data by solving distance optimization for random points."""
    mu_var = cp.Variable((G.shape[0], 1), nonneg=True)
    p_param = cp.Parameter((2, 1))
    cost = mu_var.T @ (G @ p_param - h)
    constraints = [cp.norm(G.T @ mu_var) <= 1]
    prob = cp.Problem(cp.Maximize(cost), constraints)

    points = np.random.uniform(low=data_range[:2], high=data_range[2:],
                               size=(data_size, 2))
    inputs, labels, distances = [], [], []

    for i, p in enumerate(points):
        p_param.value = p.reshape(2, 1)
        prob.solve(solver=cp.ECOS)
        inputs.append(torch.tensor(p, dtype=torch.float32))
        labels.append(torch.tensor(mu_var.value, dtype=torch.float32).squeeze())
        distances.append(torch.tensor(prob.value, dtype=torch.float32))
        if (i + 1) % 10000 == 0:
            print(f"  Generated {i+1}/{data_size} samples...", flush=True)

    return PointDataset(inputs, labels, distances)


def train(model, G_t, h_t, checkpoint_dir, epochs=5000, batch_size=4096,
          lr=1e-4, lr_decay=0.5, decay_freq=1500, save_freq=500):
    """Train DUNE model on GPU with all data pre-loaded."""
    device = torch.device('cuda')
    model = model.to(device)
    G_t = G_t.to(device)
    h_t = h_t.to(device)

    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Generating dataset...", flush=True)
    t0 = time.time()
    dataset = generate_data(100000)
    print(f"Dataset generated in {time.time()-t0:.0f}s", flush=True)

    # Pre-load ALL data to GPU
    print("Pre-loading data to GPU...", flush=True)
    all_inp = torch.stack([d[0] for d in dataset]).to(device)
    all_mu = torch.stack([d[1] for d in dataset]).to(device)
    all_dist = torch.stack([d[2] for d in dataset]).to(device)
    print(f"Loaded {len(all_inp)} samples to GPU ({all_inp.element_size() * all_inp.numel() / 1e6:.1f} MB)", flush=True)

    # Split on GPU
    n_train = 80000
    perm = torch.randperm(len(all_inp), device=device)
    train_idx = perm[:n_train]
    valid_idx = perm[n_train:]

    # Record training config
    with open(os.path.join(checkpoint_dir, 'train_dict.pkl'), 'wb') as f:
        pickle.dump({
            'data_size': 100000, 'data_range': [-25, -25, 25, 25],
            'batch_size': batch_size, 'epoch': epochs, 'lr': lr,
            'robot_G': G, 'robot_h': h,
        }, f)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    n_batches = n_train // batch_size

    # Validation data (fixed)
    v_inp = all_inp[valid_idx]
    v_mu = all_mu[valid_idx].unsqueeze(2)
    v_dist = all_dist[valid_idx]

    print(f"Training {epochs} epochs on {torch.cuda.get_device_name(0)} "
          f"(batch={batch_size}, batches/epoch={n_batches})...", flush=True)
    t0 = time.time()

    for epoch in range(epochs + 1):
        model.train()
        # Shuffle training data
        ep_perm = torch.randperm(n_train, device=device)
        ep_inp = all_inp[train_idx][ep_perm]
        ep_mu = all_mu[train_idx][ep_perm].unsqueeze(2)
        ep_dist = all_dist[train_idx][ep_perm]

        for b in range(n_batches):
            start = b * batch_size
            end = start + batch_size
            inp = ep_inp[start:end]
            label_mu = ep_mu[start:end]
            label_dist = ep_dist[start:end]

            optimizer.zero_grad()
            out_mu = model(inp).unsqueeze(2)
            pred_dist = torch.bmm(out_mu.transpose(1, 2),
                                  G_t @ inp.unsqueeze(2) - h_t).squeeze()

            loss = (loss_fn(out_mu, label_mu) +
                    loss_fn(pred_dist, label_dist) * 0.1)
            loss.backward()
            optimizer.step()

        # Validation
        if epoch % 250 == 0:
            model.eval()
            with torch.no_grad():
                out_mu = model(v_inp).unsqueeze(2)
                pred_dist = torch.bmm(out_mu.transpose(1, 2),
                                      G_t @ v_inp.unsqueeze(2) - h_t).squeeze()
                v_loss = loss_fn(pred_dist, v_dist).item()
            elapsed = time.time() - t0
            eta = (elapsed / (epoch + 1)) * (epochs - epoch) if epoch > 0 else 0
            print(f"  Epoch {epoch}/{epochs} | val_loss={v_loss:.2e} | "
                  f"elapsed={elapsed:.0f}s | ETA={eta:.0f}s", flush=True)
            model.train()

        # Save checkpoint
        if epoch > 0 and epoch % save_freq == 0:
            path = os.path.join(checkpoint_dir, f'model_{epoch}.pth')
            torch.save(model.cpu().state_dict(), path)
            model = model.to(device)
            print(f"  Saved {path}", flush=True)

        # LR decay
        if (epoch + 1) % decay_freq == 0:
            for g in optimizer.param_groups:
                g['lr'] *= lr_decay
            print(f"  LR decayed to {optimizer.param_groups[0]['lr']:.1e}", flush=True)

    final_path = os.path.join(checkpoint_dir, f'model_{epochs}.pth')
    torch.save(model.cpu().state_dict(), final_path)
    print(f"\nTraining complete! Model saved to {final_path}")
    return final_path


def main():
    checkpoint_dir = os.path.join(NEUPAN_ROOT, 'example', 'model', 'tb3_default')
    print("=" * 60)
    print("DUNE Training for TB3 (0.14m x 0.14m)")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Original model at example/model/diff_robot_default/ UNTOUCHED")
    print("=" * 60)

    model = ObsPointNet(input_dim=2, output_dim=G.shape[0])
    G_t = torch.tensor(G, dtype=torch.float32)
    h_t = torch.tensor(h, dtype=torch.float32)

    train(model, G_t, h_t, checkpoint_dir, epochs=5000)


if __name__ == '__main__':
    main()
