#!/usr/bin/env python3
"""Gazebo + NeuPAN CPSA-v4 closed-loop navigation.

Launches Gazebo, runs NeuPAN planner with CPSA-v4 safety margins,
repeats for multiple episodes, and logs results.

Usage:
  source /opt/ros/humble/setup.bash
  export GAZEBO_MODEL_PATH=.../gazebo_sim/models:/usr/share/gazebo-11/models
  python gazebo_nav_loop.py --world corridor --noise_std 0.02 --n_ep 10

ROS2 topics used:
  Subscribe: /robot/scan (LaserScan), /robot/odom (Odometry)
  Publish:   /robot/cmd_vel (Twist)
"""

import sys, os, argparse, time, json, subprocess, signal
import numpy as np

# NeuPAN imports
NEUPAN_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, NEUPAN_ROOT)

import yaml, tempfile
import torch
from neupan import neupan
from neupan.configuration import np_to_tensor, tensor_to_np

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

# Gazebo sensors publish with BEST_EFFORT; must match on subscriber side
SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


# ── Environment configs matching IR-SIM ──────────────────────

WORLD_CONFIGS = {
    'corridor': {
        'world_file': 'worlds/corridor.world',
        'planner_yaml': 'example/corridor/diff/planner.yaml',
        'env_yaml': 'example/corridor/diff/env.yaml',
        'env_type': 0,
        'goal': (40.0, 40.0),
        'start': (-5.0, 20.0, 0.0),
        'goal_threshold': 2.0,
        'max_steps': 400,
        'step_time': 0.1,
    },
    'dyna_obs': {
        'world_file': 'worlds/dyna_obs.world',
        'planner_yaml': 'example/dyna_obs/diff/planner.yaml',
        'env_yaml': 'example/dyna_obs/diff/env.yaml',
        'env_type': 3,
        'goal': (40.0, 40.0),
        'start': (10.0, 42.0, np.pi/2),
        'goal_threshold': 2.5,
        'max_steps': 400,
        'step_time': 0.1,
    },
}


class GazeboNavNode(Node):
    """ROS2 node that bridges Gazebo ↔ NeuPAN planner."""

    def __init__(self):
        super().__init__('gazebo_nav')
        self.scan_data = None
        self.odom_data = None
        self.robot_pose = None  # [x, y, theta]
        self.robot_vel = np.zeros(2)  # [vx, vy]

        self.scan_sub = self.create_subscription(
            LaserScan, '/robot/scan', self._scan_cb, SENSOR_QOS)
        self.odom_sub = self.create_subscription(
            Odometry, '/robot/odom', self._odom_cb, SENSOR_QOS)
        self.cmd_pub = self.create_publisher(Twist, '/robot/cmd_vel', 10)

    def _scan_cb(self, msg: LaserScan):
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges, dtype=np.float64)
        valid = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
        if valid.any():
            r = ranges[valid]
            a = angles[valid]
            self.scan_data = (r, a, msg.range_min, msg.range_max)

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        theta = np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        self.robot_pose = np.array([p.x, p.y, theta])
        v = msg.twist.twist
        self.robot_vel = np.array([v.linear.x, v.angular.z])

    def get_scan_points(self):
        """Return (N, 2) obstacle points in world frame."""
        if self.scan_data is None or self.robot_pose is None:
            return None
        r, a, rmin, rmax = self.scan_data
        x, y, th = self.robot_pose
        # Points in world frame
        local_x = r * np.cos(a)
        local_y = r * np.sin(a)
        cos_t, sin_t = np.cos(th), np.sin(th)
        world_x = local_x * cos_t - local_y * sin_t + x
        world_y = local_x * sin_t + local_y * cos_t + y
        return np.stack([world_x, world_y], axis=1)

    def get_scan_points_local(self):
        """Return (N, 2) obstacle points in robot-local frame."""
        if self.scan_data is None:
            return None
        r, a, _, _ = self.scan_data
        return np.stack([r * np.cos(a), r * np.sin(a)], axis=1)

    def get_scan_points_noisy(self, noise_std, rng):
        """Return (local_pts, world_pts) with Gaussian noise on ranges."""
        if self.scan_data is None or self.robot_pose is None:
            return None, None
        r, a, rmin, rmax = self.scan_data
        n = len(r)
        # Add Gaussian noise to ranges
        r_noisy = r + rng.normal(0, noise_std, size=n)
        r_noisy = np.clip(r_noisy, 0.05, 10.0)
        # Local frame
        local_x = r_noisy * np.cos(a)
        local_y = r_noisy * np.sin(a)
        local_pts = np.stack([local_x, local_y], axis=1)
        # World frame
        x, y, th = self.robot_pose
        cos_t, sin_t = np.cos(th), np.sin(th)
        world_x = local_x * cos_t - local_y * sin_t + x
        world_y = local_x * sin_t + local_y * cos_t + y
        world_pts = np.stack([world_x, world_y], axis=1)
        return local_pts, world_pts

    def send_velocity(self, vx, omega):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.angular.z = float(omega)
        self.cmd_pub.publish(msg)

    def wait_for_data(self, timeout=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.scan_data is not None and self.robot_pose is not None:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False


def launch_gazebo(world_path, headless=True):
    """Launch Gazebo in a subprocess. headless=False opens GUI."""
    env = os.environ.copy()
    env['GAZEBO_MODEL_PATH'] = (
        os.path.join(NEUPAN_ROOT, 'gazebo_sim', 'models') + ':' +
        '/usr/share/gazebo-11/models' + ':' +
        env.get('GAZEBO_MODEL_PATH', ''))
    if headless:
        cmd = ['gzserver', '--verbose', world_path]
    else:
        cmd = ['gazebo', world_path]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    return proc


def kill_gazebo():
    subprocess.run(['killall', '-9', 'gzserver', 'gzclient'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def inject_noise(scan_ranges, noise_std, rng):
    """Add Gaussian noise to scan ranges."""
    if noise_std <= 0:
        return scan_ranges
    noisy = scan_ranges + rng.normal(0, noise_std, size=scan_ranges.shape)
    return np.clip(noisy, 0.05, 10.0)


def run_episode(nav_node, planner, cfg, noise_std, max_steps, rng):
    """Run a single navigation episode in Gazebo."""
    goal = np.array(cfg['goal'])
    goal_thresh = cfg['goal_threshold']
    collision_thresh = 0.3  # meters

    # Reset scan
    nav_node.scan_data = None
    time.sleep(0.5)

    arrived = False
    collision = False
    min_dist = 999.0
    start_pos = None
    last_vx, last_omega = 0.0, 0.0

    for step in range(max_steps):
        # Wait for scan
        if nav_node.scan_data is None:
            rclpy.spin_once(nav_node, timeout_sec=0.1)
            continue

        rclpy.spin_once(nav_node, timeout_sec=0.01)

        if nav_node.robot_pose is None:
            continue

        pos = nav_node.robot_pose
        if start_pos is None:
            start_pos = pos.copy()

        dist_to_goal = np.linalg.norm(pos[:2] - goal)
        if dist_to_goal < goal_thresh:
            arrived = True
            break

        # Get noisy scan points (local for collision, world for planner)
        pts_local, pts_world = nav_node.get_scan_points_noisy(noise_std, rng)
        if pts_local is None or len(pts_local) < 3:
            nav_node.send_velocity(1.0, 0.0)
            time.sleep(0.1)
            continue

        # Collision check
        min_d = pts_local[:, 0]**2 + pts_local[:, 1]**2
        if np.sqrt(min_d.min()) < collision_thresh:
            collision = True
            break
        min_dist = min(min_dist, float(np.sqrt(min_d.min())))

        # Build NeuPAN-compatible state (ndarray (3,1) matching IR-SIM format)
        try:
            state = pos.reshape(3, 1).astype(np.float64)

            # Use world-frame points as numpy array (planner expects ndarray)
            points_np = pts_world.T.astype(np.float64)  # (2, N)

            # Run planner
            action, info = planner(state, points_np, None)

            if info.get('stop'):
                collision = True
                break
            if info.get('arrive'):
                arrived = True
                break

            # Extract velocity command from (2,1) or (2,) ndarray
            if isinstance(action, np.ndarray) and action.size >= 2:
                vx = float(action.flat[0])
                omega = float(action.flat[1])
            elif isinstance(action, torch.Tensor) and action.numel() >= 2:
                vx = float(action.flatten()[0])
                omega = float(action.flatten()[1])
            else:
                vx, omega = 1.0, 0.0

            # Clamp to safe range
            vx = np.clip(vx, -3.0, 3.0)
            omega = np.clip(omega, -2.0, 2.0)
            last_vx, last_omega = vx, omega
            nav_node.send_velocity(vx, omega)

        except Exception as e:
            # Fallback: drive forward slowly
            if step < 5:
                import traceback
                print(f"      [ERROR step {step}] {e}", flush=True)
                traceback.print_exc()
            last_vx, last_omega = 0.5, 0.0
            nav_node.send_velocity(0.5, 0.0)

        time.sleep(0.1)

        # Progress logging every 100 steps
        if step > 0 and step % 100 == 0:
            d2g = np.linalg.norm(pos[:2] - goal)
            odom_v = nav_node.robot_vel
            print(f"      step={step} pos=({pos[0]:.1f},{pos[1]:.1f}) d2g={d2g:.1f}m | cmd_v=({last_vx:.2f},{last_omega:.2f}) odom_v=({odom_v[0]:.2f},{odom_v[1]:.2f})", flush=True)

    nav_node.send_velocity(0.0, 0.0)

    # Debug: report trajectory summary
    if start_pos is not None:
        traveled = np.linalg.norm(pos[:2] - start_pos[:2])
    else:
        traveled = 0.0
    return arrived, collision, min_dist, traveled, pos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', default='corridor', choices=['corridor', 'dyna_obs'])
    parser.add_argument('--noise_std', type=float, default=0.02)
    parser.add_argument('--n_ep', type=int, default=10)
    parser.add_argument('--method', default='cpsa_v4', choices=['vanilla', 'cpsa_v4'])
    parser.add_argument('--headless', action='store_true', default=False)
    args = parser.parse_args()

    cfg = WORLD_CONFIGS[args.world]
    os.chdir(NEUPAN_ROOT)

    # Load NeuPAN planner
    plan_path = os.path.join(NEUPAN_ROOT, cfg['planner_yaml'])
    plan_cfg = yaml.safe_load(open(plan_path))
    plan_cfg['time_print'] = False
    tf_p = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(plan_cfg, tf_p); tf_p.close()

    planner = neupan.init_from_yaml(tf_p.name)
    if args.method == 'cpsa_v4':
        planner.enable_risk_calibration(
            budget_strategy='cpsa',
            cp_checkpoint='experiments/cp_head_output/cpsa_v4_universal.pth',
            collision_q_hat=0.01, noise_std=args.noise_std,
            env_type=cfg['env_type'])

    # Initialize ROS2
    rclpy.init()
    rng = np.random.default_rng(42)

    results = []
    for ep in range(args.n_ep):
        # Launch Gazebo
        world_path = os.path.join(NEUPAN_ROOT, 'gazebo_sim', cfg['world_file'])
        print(f"  Episode {ep+1}/{args.n_ep}: launching Gazebo...", flush=True)
        gz_proc = launch_gazebo(world_path, headless=args.headless)

        # Wait for Gazebo to start and DDS to discover entities
        time.sleep(10)

        # Recreate node each episode for clean DDS discovery
        nav_node = GazeboNavNode()

        # Spin to get initial data
        for _ in range(50):
            rclpy.spin_once(nav_node, timeout_sec=0.1)

        if not nav_node.wait_for_data(timeout=15.0):
            print(f"    WARNING: No scan data received, skipping episode")
            nav_node.destroy_node()
            kill_gazebo()
            results.append({'arrived': False, 'collision': False, 'min_dist': 0.0})
            time.sleep(2)
            continue

        # Run episode
        t0 = time.time()
        arrived, collision, min_dist, traveled, final_pos = run_episode(
            nav_node, planner, cfg, args.noise_std, cfg['max_steps'], rng)
        dt = time.time() - t0

        status = "ARRIVED" if arrived else ("COLLISION" if collision else "TIMEOUT")
        print(f"    {status} min_dist={min_dist:.3f}m traveled={traveled:.1f}m final_pos=({final_pos[0]:.1f},{final_pos[1]:.1f}) ({dt:.1f}s)", flush=True)
        results.append({
            'arrived': arrived, 'collision': collision,
            'min_dist': float(min_dist), 'time': dt
        })

        nav_node.destroy_node()
        kill_gazebo()
        time.sleep(2)

    # Summary
    n_arrived = sum(r['arrived'] for r in results)
    n_collision = sum(r['collision'] for r in results)
    avg_min = np.mean([r['min_dist'] for r in results if r['min_dist'] < 999])

    print(f"\n{'='*60}")
    print(f"RESULTS: {args.world} noise={args.noise_std*100:.0f}cm method={args.method}")
    print(f"  Success: {n_arrived}/{args.n_ep} ({100*n_arrived/args.n_ep:.0f}%)")
    print(f"  Collision: {n_collision}/{args.n_ep} ({100*n_collision/args.n_ep:.0f}%)")
    print(f"  Avg min dist: {avg_min:.3f}m")
    print(f"{'='*60}")

    # Save
    out_file = os.path.join(NEUPAN_ROOT, 'gazebo_sim', 'results',
                            f'{args.world}_{args.noise_std*100:.0f}cm_{args.method}.json')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump({'config': vars(args), 'results': results,
                   'summary': {'success': n_arrived, 'collision': n_collision,
                               'total': args.n_ep, 'avg_min_dist': float(avg_min)}}, f, indent=2)
    print(f"  Saved to {out_file}")

    rclpy.shutdown()
    os.unlink(tf_p.name)


if __name__ == '__main__':
    main()
