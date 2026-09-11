#!/usr/bin/env python3
"""ROS2 node: Gazebo LaserScan → CPSA-v4 17D features → per-point tau margins.

Subscribes:
  /robot/scan_noisy (sensor_msgs/LaserScan) - noisy laser scan
  /robot/odom (nav_msgs/Odometry) - robot odometry

Publishes:
  /robot/tau_margins (std_msgs/Float32MultiArray) - per-point tau margins

Parameters:
  cpsa_checkpoint (string): path to CPSA-v4 model checkpoint
  noise_std (double): current sensor noise level
  env_type (int): environment type index (0=corridor, 3=dyna_obs, 4=non_obs)
  goal_x, goal_y (double): goal position in world frame
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from neupan.blocks.cpsa_risk_net import CPSARiskNet


class CPSAAdapterNode(Node):
    def __init__(self):
        super().__init__('cpsa_adapter')

        # Parameters
        self.declare_parameter('cpsa_checkpoint', '')
        self.declare_parameter('noise_std', 0.0)
        self.declare_parameter('env_type', 0)
        self.declare_parameter('goal_x', 40.0)
        self.declare_parameter('goal_y', 40.0)

        ckpt_path = self.get_parameter('cpsa_checkpoint').value
        self.noise_std = self.get_parameter('noise_std').value
        self.env_type = self.get_parameter('env_type').value
        self.goal = np.array([
            self.get_parameter('goal_x').value,
            self.get_parameter('goal_y').value
        ])

        # Load CPSA-v4 model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model(ckpt_path)

        # State
        self.robot_pose = None  # [x, y, theta]
        self.prev_d_pred = None
        self.prev_radius = None
        self.robot_width = 2.0

        # ROS interfaces
        self.scan_sub = self.create_subscription(
            LaserScan, '/robot/scan_noisy', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/robot/odom', self.odom_callback, 10)
        self.tau_pub = self.create_publisher(
            Float32MultiArray, '/robot/tau_margins', 10)

        self.get_logger().info(
            f'CPSAAdapter started: checkpoint={ckpt_path}, '
            f'noise_std={self.noise_std}, env_type={self.env_type}')

    def _load_model(self, ckpt_path):
        if not ckpt_path or not os.path.exists(ckpt_path):
            self.get_logger().warn(f'No valid checkpoint: {ckpt_path}')
            self.net = None
            return
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        config = ckpt.get('config', {})
        self.net = CPSARiskNet(
            feature_dim=config.get('feature_dim', 17),
            hidden_dims=tuple(config.get('hidden_dims', [128, 64, 32])),
        )
        self.net.load_state_dict(ckpt['model_state'])
        self.net.to(self.device)
        self.net.eval()
        self.q_star = ckpt.get('q_star', 0.0)
        self.q_star_cond = ckpt.get('q_star_conditional', {})
        self.feat_mean = ckpt.get('feature_mean', None)
        self.feat_std = ckpt.get('feature_std', None)
        self.feat_dim = config.get('feature_dim', 17)
        self.get_logger().info(
            f'Model loaded: {self.net.param_count} params, q_star={self.q_star:.4f}')

    def odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                           1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.robot_pose = np.array([x, y, theta])

    def scan_callback(self, msg: LaserScan):
        if self.net is None or self.robot_pose is None:
            return

        x, y, theta = self.robot_pose
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Convert LaserScan to local point cloud
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)
        valid = (ranges > msg.range_min) & (ranges < msg.range_max)
        if not valid.any():
            return

        r = ranges[valid]
        a = angles[valid]
        pts_local = np.stack([r * np.cos(a), r * np.sin(a)], axis=1)  # (N, 2)

        # For DUNE d_pred: use simplified distance estimate
        # (In full integration, this would come from the DUNE network)
        d_pred = torch.tensor(r, dtype=torch.float32)

        N = pts_local.shape[0]
        pts_t = torch.tensor(pts_local, dtype=torch.float32)

        eps = 1e-6
        radius = torch.sqrt(pts_t[:, 0]**2 + pts_t[:, 1]**2 + eps)
        angle = torch.arctan2(pts_t[:, 1], pts_t[:, 0])
        proximity_ratio = torch.abs(d_pred) / (radius + eps)

        # Local density (kNN)
        if N > 5:
            diff = pts_t.unsqueeze(1) - pts_t.unsqueeze(0)
            dists = torch.sqrt((diff**2).sum(dim=2) + eps)
            k = min(5, N - 1)
            knn_dists, _ = torch.topk(dists, k + 1, dim=1, largest=False)
            local_density = knn_dists[:, 1:].mean(dim=1)
            in_radius_count = (dists < 1.0).float().sum(dim=1) - 1
            in_radius = torch.clamp(in_radius_count, 0, 50) / 50.0
        elif N > 1:
            diff = pts_t.unsqueeze(1) - pts_t.unsqueeze(0)
            dists = torch.sqrt((diff**2).sum(dim=2) + eps)
            local_density = dists[dists > 0].mean().expand(N)
            in_radius = torch.zeros(N)
        else:
            local_density = torch.full((N,), 10.0)
            in_radius = torch.zeros(N)

        speed = 0.0  # Would come from odometry twist
        speed_t = torch.full((N,), speed)
        d_pred_rank = torch.argsort(torch.argsort(torch.abs(d_pred))).float() / max(N - 1, 1)
        noise_std_t = torch.full((N,), self.noise_std)

        # Temporal features
        d_pred_delta = torch.zeros(N)
        if self.prev_d_pred is not None and self.prev_d_pred.shape[0] == N:
            d_pred_delta = d_pred - self.prev_d_pred
        radius_delta = torch.zeros(N)
        if self.prev_radius is not None and self.prev_radius.shape[0] == N:
            radius_delta = radius - self.prev_radius

        # Passage features
        yy = pts_t[:, 1]
        left_mask = yy > 0
        right_mask = yy < 0
        left_min = torch.abs(yy[left_mask]).min().item() if left_mask.any() else 10.0
        right_min = torch.abs(yy[right_mask]).min().item() if right_mask.any() else 10.0
        lateral_margin_val = min(left_min, right_min)
        passage_ratio_val = (left_min + right_min) / max(self.robot_width, 0.1)
        lateral_margin = torch.full((N,), lateral_margin_val)
        passage_ratio = torch.full((N,), passage_ratio_val)
        env_type = torch.full((N,), float(self.env_type) / 4.0)

        # Build 17D features
        features = torch.stack([
            d_pred, radius, angle, proximity_ratio, d_pred**2,
            torch.cos(angle), torch.sin(angle),
            local_density, in_radius,
            speed_t, d_pred_rank, noise_std_t,
            d_pred_delta, radius_delta,
            lateral_margin, passage_ratio, env_type,
        ], dim=1)

        # Standardize
        if self.feat_mean is not None and self.feat_std is not None:
            fm = self.feat_mean.to(features.device)
            fs = self.feat_std.to(features.device)
            if fm.shape[0] < features.shape[1]:
                pad = features.shape[1] - fm.shape[0]
                fm = torch.cat([fm, torch.zeros(pad)])
                fs = torch.cat([fs, torch.ones(pad)])
            features = (features - fm) / (fs + 1e-6)
        features = torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)

        # Inference
        with torch.no_grad():
            tau_raw = self.net(features, noise_std=noise_std_t)

        # Conformal correction with dynamic q_star
        if self.noise_std < 1e-8:
            tau_final = torch.relu(tau_raw)
        else:
            q = self.q_star
            if self.q_star_cond:
                keys = sorted(self.q_star_cond.keys())
                closest = min(keys, key=lambda k: abs(k - self.noise_std))
                q = self.q_star_cond[closest]
            tau_raw_mean = tau_raw.mean().item()
            q_scale = max(0.4, 1.0 - tau_raw_mean / 0.04)
            q = q * q_scale
            tau_final = torch.relu(tau_raw + q)

        # Dynamic tau_max
        tau_max = 0.08
        if lateral_margin_val < 0.5:
            tau_max = 0.04
        tau_final = torch.clamp(tau_final, 0.0, tau_max)

        # Smooth
        if N > 3:
            tau_np = tau_final.numpy()
            sorted_idx = np.argsort(d_pred.numpy())
            tau_sorted = tau_np[sorted_idx]
            kernel = np.array([0.25, 0.5, 0.25])
            tau_smooth = np.convolve(tau_sorted, kernel, mode='same')
            unsort_idx = np.argsort(sorted_idx)
            tau_np = tau_smooth[unsort_idx]
            tau_final = torch.tensor(tau_np, dtype=torch.float32)

        # Store temporal state
        self.prev_d_pred = d_pred.detach().clone()
        self.prev_radius = radius.detach().clone()

        # Publish
        msg_out = Float32MultiArray()
        msg_out.data = tau_final.tolist()
        self.tau_pub.publish(msg_out)


def main(args=None):
    rclpy.init(args=args)
    node = CPSAAdapterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
