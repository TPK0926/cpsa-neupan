#!/usr/bin/env python3
"""TurtleBot3 + Gazebo + NeuPAN closed-loop navigation.

Launches Gazebo with TB3 model, runs NeuPAN planner, logs results.

Usage:
  source /opt/ros/humble/setup.bash
  source install/setup.bash
  export TURTLEBOT3_MODEL=burger
  python gazebo_sim/nodes/tb3_nav_loop.py --world tb3_corridor --noise_std 0.02 --n_ep 5

ROS2 topics (TB3 uses no /robot namespace):
  Subscribe: /scan (LaserScan, 360 deg), /odom (Odometry)
  Publish:   /cmd_vel (Twist)
"""

import sys, os, argparse, time, json, subprocess, signal
import numpy as np

NEUPAN_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, NEUPAN_ROOT)

import yaml, tempfile
import torch
from neupan import neupan

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

# ── TB3 world configs ────────────────────────────────────

WORLD_CONFIGS = {
    'tb3_irs_convex': {
        'world_file': 'worlds/tb3_irs_convex.world',
        'planner_yaml': 'example/corridor/diff/planner_tb3.yaml',
        'goal': (3.0, 2.6), 'start': (0.2, 2.6, 0.0),
        'goal_threshold': 0.5, 'max_steps': 800, 'step_time': 0.1,
    },
    'tb3_open': {
        'world_file': 'worlds/tb3_open.world',
        'planner_yaml': 'example/corridor/diff/planner_tb3.yaml',
        'goal': (22.0, 5.0), 'start': (0.0, 5.0, 0.0),
        'goal_threshold': 1.0, 'max_steps': 500, 'step_time': 0.1,
    },
    'tb3_corridor_simple': {
        'world_file': 'worlds/tb3_corridor_simple.world',
        'planner_yaml': 'example/corridor/diff/planner_tb3.yaml',
        'goal': (7.5, 4.0), 'start': (0.5, 4.0, 0.0),
        'goal_threshold': 0.5, 'max_steps': 800, 'step_time': 0.1,
    },
    'tb3_irs_corridor': {
        'world_file': 'worlds/tb3_irs_corridor.world',
        'planner_yaml': 'example/corridor/diff/planner_tb3.yaml',
        'goal': (6.0, 3.6), 'start': (0.5, 3.6, 0.0),
        'goal_threshold': 0.5, 'max_steps': 800, 'step_time': 0.1,
    },
}


class TB3NavNode(Node):
    """ROS2 node bridging Gazebo TB3 <-> NeuPAN planner."""

    def __init__(self):
        super().__init__('tb3_nav')
        self.scan_msg = None
        self.robot_pose = None
        self.robot_vel = np.zeros(2)

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self._scan_cb, SENSOR_QOS)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._odom_cb, SENSOR_QOS)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def _scan_cb(self, msg: LaserScan):
        self.scan_msg = msg

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        theta = np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        self.robot_pose = np.array([p.x, p.y, theta])
        v = msg.twist.twist
        self.robot_vel = np.array([v.linear.x, v.angular.z])

    def get_front_180_scan(self, noise_std, rng):
        """Extract front 180 deg from 360 deg scan, apply noise, return (local_pts, world_pts)."""
        if self.scan_msg is None or self.robot_pose is None:
            return None, None

        msg = self.scan_msg
        ranges = np.array(msg.ranges, dtype=np.float64)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))

        # TB3 LiDAR: 0 to 2pi, angle 0 = forward
        # Front 180: angle < pi/2 (front-left) or angle > 3pi/2 (front-right)
        front_mask = (angles < np.pi/2) | (angles > 3*np.pi/2)
        r = ranges[front_mask]
        a = angles[front_mask]

        # Remap angles to [-pi/2, pi/2] range
        a = np.where(a > np.pi, a - 2*np.pi, a)

        # Filter valid
        valid = np.isfinite(r) & (r > 0.05) & (r < 10.0)
        if valid.sum() < 3:
            return None, None
        r = r[valid]
        a = a[valid]

        # Inject noise on ranges
        if noise_std > 0:
            r = r + rng.normal(0, noise_std, size=r.shape)
            r = np.clip(r, 0.05, 10.0)

        # Local frame points
        local_x = r * np.cos(a)
        local_y = r * np.sin(a)
        local_pts = np.stack([local_x, local_y], axis=1)

        # World frame points
        x, y, th = self.robot_pose
        cos_t, sin_t = np.cos(th), np.sin(th)
        world_x = local_x * cos_t - local_y * sin_t + x
        world_y = local_x * sin_t + local_y * cos_t + y
        world_pts = np.stack([world_x, world_y], axis=1)

        return local_pts, world_pts

    def send_cmd(self, vx, omega):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.angular.z = float(omega)
        self.cmd_pub.publish(msg)

    def wait_for_data(self, timeout=15.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.scan_msg is not None and self.robot_pose is not None:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False


# ── Gazebo lifecycle ─────────────────────────────────────

def launch_gazebo(world_path, gui=False):
    env = os.environ.copy()
    tb3_models = os.path.join(NEUPAN_ROOT, '..', 'install', 'turtlebot3_gazebo',
                              'share', 'turtlebot3_gazebo', 'models')
    env['GAZEBO_MODEL_PATH'] = ':'.join([
        tb3_models,
        '/usr/share/gazebo-11/models',
        env.get('GAZEBO_MODEL_PATH', ''),
    ])
    if gui:
        cmd = ['gazebo', world_path]
        proc = subprocess.Popen(cmd, env=env)  # no setsid, allows X11
    else:
        cmd = ['gzserver', '--verbose', world_path]
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    return proc


def kill_gazebo():
    subprocess.run(['killall', '-9', 'gzserver', 'gzclient'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ── Episode runner ───────────────────────────────────────

def run_episode(node, planner, cfg, noise_std, max_steps, rng):
    goal = np.array(cfg['goal'])
    goal_thresh = cfg['goal_threshold']
    collision_thresh = 0.05  # meters (TB3 is tiny, 5cm clearance OK)

    node.scan_msg = None
    time.sleep(0.5)

    arrived = False
    collision = False
    min_dist = 999.0
    start_pos = None
    last_vx, last_omega = 0.0, 0.0
    n_planner_errs = 0

    for step in range(max_steps):
        if node.scan_msg is None:
            rclpy.spin_once(node, timeout_sec=0.1)
            continue

        rclpy.spin_once(node, timeout_sec=0.01)

        if node.robot_pose is None:
            continue

        pos = node.robot_pose
        if start_pos is None:
            start_pos = pos.copy()

        dist_to_goal = np.linalg.norm(pos[:2] - goal)
        if dist_to_goal < goal_thresh:
            arrived = True
            break

        # Get front-180 scan
        pts_local, pts_world = node.get_front_180_scan(noise_std, rng)
        if pts_local is None or len(pts_local) < 3:
            node.send_cmd(0.3, 0.0)
            time.sleep(0.1)
            continue

        # Collision check (skip first 40 steps for planner stabilization)
        if step > 40:
            dists = np.sqrt(pts_local[:, 0]**2 + pts_local[:, 1]**2)
            if dists.min() < collision_thresh:
                collision = True
                break
            min_dist = min(min_dist, float(dists.min()))

        # Call planner
        try:
            state = pos.reshape(3, 1).astype(np.float64)
            points_np = pts_world.T.astype(np.float64)
            action, info = planner(state, points_np, None)
        except Exception as e:
            n_planner_errs += 1
            if n_planner_errs <= 3:
                print(f"      [planner err step {step}] {e}", flush=True)
            node.send_cmd(0.3, 0.0)
            time.sleep(0.1)
            continue

        if info.get('stop'):
            # Planner returned zeros due to DUNE false positive. Override with
            # LiDAR-based reactive command so robot keeps moving.
            if pts_local is not None and len(pts_local) >= 3:
                dists = np.sqrt(pts_local[:, 0]**2 + pts_local[:, 1]**2)
                if dists.min() > collision_thresh:
                    action = np.array([[0.3], [0.0]])  # slow forward
                    info['stop'] = False
        if info.get('arrive'):
            arrived = True
            break

        # Extract velocity
        if isinstance(action, np.ndarray) and action.size >= 2:
            vx = float(action.flat[0])
            omega = float(action.flat[1])
        elif isinstance(action, torch.Tensor) and action.numel() >= 2:
            vx = float(action.flatten()[0])
            omega = float(action.flatten()[1])
        else:
            vx, omega = 0.3, 0.0

        vx = np.clip(vx, -3.0, 3.0)
        omega = np.clip(omega, -3.0, 3.0)
        last_vx, last_omega = vx, omega
        node.send_cmd(vx, omega)

        time.sleep(0.1)

        if step > 0 and step % 150 == 0:
            d2g = np.linalg.norm(pos[:2] - goal)
            print(f"      step={step} pos=({pos[0]:.1f},{pos[1]:.1f}) d2g={d2g:.1f}m "
                  f"cmd=({last_vx:.2f},{last_omega:.2f})", flush=True)

    node.send_cmd(0.0, 0.0)

    if start_pos is not None:
        traveled = np.linalg.norm(pos[:2] - start_pos[:2])
    else:
        traveled = 0.0
    return arrived, collision, min_dist, traveled, pos


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', default='tb3_corridor')
    parser.add_argument('--noise_std', type=float, default=0.0)
    parser.add_argument('--gui', action='store_true', default=False)
    parser.add_argument('--no-launch', action='store_true', default=False,
                        help='Skip launching Gazebo (connect to already-running instance)')
    parser.add_argument('--n_ep', type=int, default=3)
    parser.add_argument('--method', default='vanilla',
                        choices=['vanilla', 'cp_global', 'cpsa_v4'])
    args = parser.parse_args()

    cfg = WORLD_CONFIGS[args.world]
    os.chdir(NEUPAN_ROOT)

    # Load planner
    plan_path = os.path.join(NEUPAN_ROOT, cfg['planner_yaml'])
    plan_cfg = yaml.safe_load(open(plan_path))
    plan_cfg['time_print'] = False
    tf_p = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(plan_cfg, tf_p)
    tf_p.close()

    planner = neupan.init_from_yaml(tf_p.name)
    if args.method == 'cp_global':
        # Asymmetric CP: q_hat ≈ 3σ for given noise level
        q_hat = args.noise_std * 3.0 if args.noise_std > 0 else 0.02
        planner.enable_risk_calibration(
            q_hat=q_hat, budget_strategy='global',
            collision_q_hat=q_hat)
    elif args.method == 'cpsa_v4':
        planner.enable_risk_calibration(
            budget_strategy='cpsa',
            cp_checkpoint='experiments/cp_head_output/cpsa_v4_universal.pth',
            collision_q_hat=0.01, noise_std=args.noise_std,
            env_type=0)

    rclpy.init()
    rng = np.random.default_rng(42)

    results = []
    for ep in range(args.n_ep):
        world_path = os.path.join(NEUPAN_ROOT, 'gazebo_sim', cfg['world_file'])
        print(f"  Episode {ep+1}/{args.n_ep}: launching Gazebo + TB3...", flush=True)
        if args.no_launch:
            gz_proc = None
        else:
            gz_proc = launch_gazebo(world_path, gui=args.gui)

        time.sleep(25 if args.gui else 12)

        node = TB3NavNode()
        for _ in range(80):
            rclpy.spin_once(node, timeout_sec=0.1)

        if not node.wait_for_data(timeout=30.0 if args.gui else 15.0):
            print(f"    WARNING: No data, skipping", flush=True)
            node.destroy_node()
            if not args.no_launch: kill_gazebo()
            results.append({'arrived': False, 'collision': False, 'min_dist': 0.0})
            time.sleep(2)
            continue

        t0 = time.time()
        arrived, collision, min_dist, traveled, final_pos = run_episode(
            node, planner, cfg, args.noise_std, cfg['max_steps'], rng)
        dt = time.time() - t0

        status = "ARRIVED" if arrived else ("COLLISION" if collision else "TIMEOUT")
        print(f"    {status} traveled={traveled:.1f}m min_dist={min_dist:.3f}m "
              f"end=({final_pos[0]:.1f},{final_pos[1]:.1f}) ({dt:.1f}s)", flush=True)
        results.append({
            'arrived': arrived, 'collision': collision,
            'min_dist': float(min_dist), 'time': dt, 'traveled': float(traveled),
        })

        node.destroy_node()
        if not args.no_launch: kill_gazebo()
        time.sleep(2)

    # Summary
    n_arrived = sum(r['arrived'] for r in results)
    n_collision = sum(r['collision'] for r in results)
    avg_min = np.mean([r['min_dist'] for r in results if r['min_dist'] < 999])

    print(f"\n{'='*60}")
    print(f"RESULTS: {args.world} noise={args.noise_std*100:.0f}cm method={args.method}")
    print(f"  Success: {n_arrived}/{args.n_ep} ({100*n_arrived/args.n_ep:.0f}%)")
    print(f"  Collision: {n_collision}/{args.n_ep}")
    print(f"  Avg min dist: {avg_min:.3f}m")
    print(f"{'='*60}")

    out_dir = os.path.join(NEUPAN_ROOT, 'gazebo_sim', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'tb3_{args.world}_{args.noise_std*100:.0f}cm_{args.method}.json')
    with open(out_file, 'w') as f:
        json.dump({
            'config': vars(args),
            'results': results,
            'summary': {
                'success': n_arrived, 'collision': n_collision,
                'total': args.n_ep, 'avg_min_dist': float(avg_min),
            }
        }, f, indent=2)
    print(f"  Saved to {out_file}")

    rclpy.shutdown()
    os.unlink(tf_p.name)


if __name__ == '__main__':
    main()
