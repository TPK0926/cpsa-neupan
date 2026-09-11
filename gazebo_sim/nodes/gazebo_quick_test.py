#!/usr/bin/env python3
"""Quick single-episode Gazebo test: verify NeuPAN + CPSA-v4 runs end-to-end.

Reads /robot/scan, downsampled to 100 beams, runs through NeuPAN planner,
publishes /robot/cmd_vel. Checks for arrival/collision.

Usage:
  # Terminal 1: Start Gazebo
  export GAZEBO_MODEL_PATH=.../gazebo_sim/models:/usr/share/gazebo-11/models
  gzserver --verbose .../gazebo_sim/worlds/corridor.world

  # Terminal 2: Run this script
  source /opt/ros/humble/setup.bash
  python gazebo_quick_test.py --noise_std 0.02 --method cpsa_v4
"""

import sys, os, argparse, time
import numpy as np

NEUPAN_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, NEUPAN_ROOT)

import yaml, tempfile, torch
from neupan import neupan

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


class QuickNavNode(Node):
    def __init__(self):
        super().__init__('quick_nav')
        self.scan_msg = None
        self.pose = None  # [x, y, theta]
        self.scan_sub = self.create_subscription(LaserScan, '/robot/scan', self._scan, 10)
        self.odom_sub = self.create_subscription(Odometry, '/robot/odom', self._odom, 10)
        self.cmd_pub = self.create_publisher(Twist, '/robot/cmd_vel', 10)

    def _scan(self, msg):
        self.scan_msg = msg

    def _odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        th = np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        self.pose = np.array([p.x, p.y, th])

    def get_planner_input(self, noise_std, rng):
        """Convert scan + pose → NeuPAN format. Returns (state_dict, pts_tensor) or None."""
        if self.scan_msg is None or self.pose is None:
            return None

        msg = self.scan_msg
        ranges = np.array(msg.ranges, dtype=np.float64)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))

        # Downsample to 100 beams to match IR-SIM / NeuPAN training
        if len(ranges) > 100:
            idx = np.linspace(0, len(ranges) - 1, 100, dtype=int)
            ranges = ranges[idx]
            angles = angles[idx]

        valid = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
        if valid.sum() < 3:
            return None
        r = ranges[valid]
        a = angles[valid]

        # Inject noise
        if noise_std > 0:
            r = r + rng.normal(0, noise_std, size=r.shape)
            r = np.clip(r, 0.05, 10.0)

        # Local frame points
        px = r * np.cos(a)
        py = r * np.sin(a)

        # World frame points (for planner state)
        x, y, th = self.pose
        cos_t, sin_t = np.cos(th), np.sin(th)
        wx = px * cos_t - py * sin_t + x
        wy = px * sin_t + py * cos_t + y

        # Build scan dict compatible with NeuPAN's scan_to_point
        scan_dict = {
            'ranges': r,
            'angles': a,
            'range_min': msg.range_min,
            'range_max': msg.range_max,
            'points_world': np.stack([wx, wy], axis=1),
            'points_local': np.stack([px, py], axis=1),
        }
        return scan_dict

    def send_cmd(self, vx, omega):
        msg = Twist()
        msg.linear.x = float(np.clip(vx, -2.0, 2.0))
        msg.angular.z = float(np.clip(omega, -3.14, 3.14))
        self.cmd_pub.publish(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_std', type=float, default=0.02)
    parser.add_argument('--method', default='cpsa_v4', choices=['vanilla', 'cpsa_v4'])
    parser.add_argument('--max_steps', type=int, default=400)
    parser.add_argument('--goal_x', type=float, default=40.0)
    parser.add_argument('--goal_y', type=float, default=40.0)
    parser.add_argument('--goal_thresh', type=float, default=2.0)
    parser.add_argument('--collision_thresh', type=float, default=0.3)
    parser.add_argument('--env_type', type=int, default=0)
    parser.add_argument('--planner_yaml', default='example/corridor/diff/planner.yaml')
    args = parser.parse_args()

    os.chdir(NEUPAN_ROOT)
    rng = np.random.default_rng(42)

    # Load planner
    plan_cfg = yaml.safe_load(open(args.planner_yaml))
    plan_cfg['time_print'] = False
    tf = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(plan_cfg, tf); tf.close()

    planner = neupan.init_from_yaml(tf.name)
    if args.method == 'cpsa_v4':
        planner.enable_risk_calibration(
            budget_strategy='cpsa',
            cp_checkpoint='experiments/cp_head_output/cpsa_v4_universal.pth',
            collision_q_hat=0.01, noise_std=args.noise_std,
            env_type=args.env_type)
    print(f"Method: {args.method}, noise: {args.noise_std*100:.0f}cm")

    rclpy.init()
    node = QuickNavNode()
    goal = np.array([args.goal_x, args.goal_y])

    # Wait for data
    print("Waiting for Gazebo data...", flush=True)
    for _ in range(100):
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.scan_msg is not None and node.pose is not None:
            break
    if node.scan_msg is None:
        print("ERROR: No scan data. Is Gazebo running?")
        node.destroy_node(); rclpy.shutdown(); os.unlink(tf.name); return

    print(f"Start pose: {node.pose}, scan beams: {len(node.scan_msg.ranges)}")

    min_dist = 999.0
    for step in range(args.max_steps):
        rclpy.spin_once(node, timeout_sec=0.05)

        # Check goal
        dist_goal = np.linalg.norm(node.pose[:2] - goal)
        if dist_goal < args.goal_thresh:
            print(f"ARRIVED at step {step} (dist={dist_goal:.2f}m)")
            break

        # Get planner input
        scan_data = node.get_planner_input(args.noise_std, rng)
        if scan_data is None:
            node.send_cmd(0.5, 0.0)
            time.sleep(0.1)
            continue

        pts_local = scan_data['points_local']
        r = np.sqrt(pts_local[:, 0]**2 + pts_local[:, 1]**2)

        if r.min() < args.collision_thresh:
            print(f"COLLISION at step {step} (min_dist={r.min():.3f}m)")
            node.send_cmd(0.0, 0.0)
            break
        min_dist = min(min_dist, r.min())

        # Run NeuPAN planner
        try:
            # NeuPAN expects numpy: state (3,1), points (2,N)
            state_np = node.pose.reshape(3, 1).astype(np.float64)
            pts_np = pts_local.T.astype(np.float64)  # (2, N)

            action, info = planner(state_np, pts_np, None)

            if info.get('stop'):
                print(f"STOP (collision) at step {step}")
                node.send_cmd(0.0, 0.0)
                break
            if info.get('arrive'):
                print(f"ARRIVE at step {step}")
                break

            # Extract velocity
            if isinstance(action, np.ndarray) and action.size >= 2:
                vx, omega = float(action.flat[0]), float(action.flat[1])
            elif isinstance(action, torch.Tensor) and action.numel() >= 2:
                vx, omega = float(action[0]), float(action[1])
            else:
                vx, omega = 1.0, 0.0

            # Scale down for Gazebo physics stability
            vx = np.clip(vx, -2.0, 2.0) * 0.7
            omega = np.clip(omega, -2.0, 2.0)

            node.send_cmd(vx, omega)

            if step % 50 == 0:
                print(f"  step {step}: pose=({node.pose[0]:.1f},{node.pose[1]:.1f},{np.degrees(node.pose[2]):.0f}°) "
                      f"goal_dist={dist_goal:.1f}m min_d={min_dist:.3f}m cmd=({vx:.2f},{omega:.2f})")

        except Exception as e:
            if step % 20 == 0:
                print(f"  step {step}: planner error: {e}")
            node.send_cmd(0.5, 0.0)

        time.sleep(0.1)
    else:
        print(f"TIMEOUT at step {args.max_steps}")

    node.send_cmd(0.0, 0.0)
    print(f"Final: min_dist={min_dist:.3f}m")

    node.destroy_node()
    rclpy.shutdown()
    os.unlink(tf.name)


if __name__ == '__main__':
    main()
