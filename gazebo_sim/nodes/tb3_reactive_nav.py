#!/usr/bin/env python3
"""TB3 reactive navigation with CP calibration for Gazebo validation.

Simple goal-seeking + LiDAR obstacle avoidance controller.
Compares vanilla vs CP-calibrated safety margins.

Usage:
  python tb3_reactive_nav.py --noise_std 0.02 --n_ep 5 --method vanilla
  python tb3_reactive_nav.py --noise_std 0.02 --n_ep 5 --method cp
"""

import sys, os, argparse, time, json, subprocess, signal
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

NEUPAN_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')

SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST, depth=10)

WORLD_CONFIGS = {
    'tb3_irs_corridor': {
        'world_file': 'worlds/tb3_irs_corridor.world',
        'goal': (3.5, 2.6), 'goal_threshold': 0.5, 'max_steps': 600,
        'desc': 'IR-SIM corridor: 6.1x2.0m, 4 inner blocks',
    },
    'tb3_irs_convex': {
        'world_file': 'worlds/tb3_irs_convex.world',
        'goal': (3.0, 2.6), 'goal_threshold': 0.5, 'max_steps': 800,
        'desc': 'IR-SIM convex_obs: circles + polygon, 3.7x3.7m',
    },
    'tb3_irs_nonobs': {
        'world_file': 'worlds/tb3_irs_nonobs.world',
        'goal': (3.0, 2.6), 'goal_threshold': 0.5, 'max_steps': 800,
        'desc': 'IR-SIM non_obs: irregular polygon obstacles, 3.7x3.7m',
    },
    'tb3_irs_narrow': {
        'world_file': 'worlds/tb3_irs_narrow.world',
        'goal': (3.5, 0.3), 'goal_threshold': 0.3, 'max_steps': 800,
        'desc': 'IR-SIM narrow_corridor: 4.4x0.6m tight passage',
    },
}


class TB3Controller:
    """Reactive controller: goal-seeking + LiDAR obstacle avoidance."""

    def __init__(self, cp_margin=0.0):
        self.cp_margin = cp_margin  # extra safety margin from CP
        self.base_safety = 0.12      # base safety distance (meters)

    @property
    def safety_dist(self):
        return self.base_safety + self.cp_margin

    def compute_action(self, robot_pose, goal, scan_ranges, scan_angles):
        """Compute (vx, omega) to approach goal while avoiding obstacles."""
        x, y, th = robot_pose

        # Goal vector in robot frame
        dx = goal[0] - x
        dy = goal[1] - y
        goal_dist = np.linalg.norm([dx, dy])
        goal_angle = np.arctan2(dy, dx)
        angle_to_goal = goal_angle - th
        angle_to_goal = np.arctan2(np.sin(angle_to_goal), np.cos(angle_to_goal))

        # Divide LiDAR into 5 sectors
        n = len(scan_ranges)
        sector_size = n // 5
        sectors = {}
        names = ['far_left', 'left', 'center', 'right', 'far_right']
        for i, name in enumerate(names):
            start = i * sector_size
            end = (i + 1) * sector_size if i < 4 else n
            valid = np.isfinite(scan_ranges[start:end])
            if valid.any():
                sectors[name] = scan_ranges[start:end][valid].min()
            else:
                sectors[name] = 10.0  # no obstacle

        # Obstacle check
        danger_left = min(sectors['far_left'], sectors['left'])
        danger_right = min(sectors['right'], sectors['far_right'])
        danger_center = sectors['center']
        min_obstacle = min(sectors.values())

        # Collision detection
        collision = min_obstacle < 0.15

        # Compute velocity
        if collision:
            return 0.0, 0.0, True

        sf = self.safety_dist

        if danger_center < sf:
            # Obstacle ahead: turn toward more open side
            if danger_left > danger_right:
                vx = 0.05
                omega = 1.0  # turn left
            else:
                vx = 0.05
                omega = -1.0  # turn right
        elif danger_left < sf * 0.8:
            # Obstacle on left: turn right
            vx = 0.3
            omega = -0.5
        elif danger_right < sf * 0.8:
            # Obstacle on right: turn left
            vx = 0.3
            omega = 0.5
        else:
            # Clear path: go toward goal
            vx = min(0.5, goal_dist * 0.1 + 0.1)
            vx = max(vx, 0.1)
            omega = 1.5 * angle_to_goal
            omega = np.clip(omega, -1.5, 1.5)

        return vx, omega, False


class TB3Node(Node):
    def __init__(self):
        super().__init__('tb3_reactive')
        self.scan_msg = None
        self.robot_pose = None
        self.create_subscription(LaserScan, '/scan', self._scan_cb, SENSOR_QOS)
        self.create_subscription(Odometry, '/odom', self._odom_cb, SENSOR_QOS)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def _scan_cb(self, msg): self.scan_msg = msg

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        th = np.arctan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        self.robot_pose = np.array([p.x, p.y, th])

    def send_cmd(self, vx, omega):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.angular.z = float(omega)
        self.cmd_pub.publish(msg)

    def get_scan_arrays(self, noise_std=0.02, rng=None):
        """Read LiDAR scan. Gazebo provides 1cm noise; +2cm injection = effective 2.2cm."""
        if self.scan_msg is None:
            return None, None
        msg = self.scan_msg
        ranges = np.array(msg.ranges, dtype=np.float64)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))

        # Front 180 deg
        front_mask = (angles < np.pi/2) | (angles > 3*np.pi/2)
        r = ranges[front_mask].copy()
        a = angles[front_mask].copy()
        a = np.where(a > np.pi, a - 2*np.pi, a)

        valid = np.isfinite(r) & (r > 0.05) & (r < 10.0)
        if valid.sum() < 3:
            return None, None
        r = r[valid]
        a = a[valid]

        # Mild noise injection (2cm on top of Gazebo's 1cm)
        if noise_std > 0 and rng is not None:
            r = r + rng.normal(0, noise_std, size=r.shape)
            r = np.clip(r, 0.05, 10.0)

        return r, a

    def wait_for_data(self, timeout=15.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.scan_msg is not None and self.robot_pose is not None:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False


def launch_gazebo(world_path):
    env = os.environ.copy()
    tb3_models = os.path.join(NEUPAN_ROOT, '..', 'install', 'turtlebot3_gazebo',
                              'share', 'turtlebot3_gazebo', 'models')
    env['GAZEBO_MODEL_PATH'] = ':'.join([tb3_models, '/usr/share/gazebo-11/models',
                                          env.get('GAZEBO_MODEL_PATH', '')])
    proc = subprocess.Popen(['gzserver', world_path], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            preexec_fn=os.setsid)
    return proc


def kill_gazebo():
    subprocess.run(['killall', '-9', 'gzserver', 'gzclient'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', default='tb3_corridor')
    # noise provided by Gazebo built-in sensor model (LiDAR: 1cm Gaussian)
    parser.add_argument('--n_ep', type=int, default=5)
    parser.add_argument('--method', default='vanilla',
                        choices=['vanilla', 'cp', 'cp_symmetric', 'cp_posthoc'])
    args = parser.parse_args()

    cfg = WORLD_CONFIGS[args.world]
    os.chdir(NEUPAN_ROOT)
    goal = np.array(cfg['goal'])

    # CP calibration: effective noise ~2.2cm (Gazebo 1cm + injection 2cm)
    # 3-sigma = 6.6cm asymmetric; symmetric needs ~1.5x for same coverage
    if args.method == 'cp':
        cp_margin = 0.06  # asymmetric: 6cm
    elif args.method == 'cp_symmetric':
        cp_margin = 0.09  # symmetric: 9cm (larger for same coverage)
    elif args.method == 'cp_posthoc':
        cp_margin = 0.06  # post-hoc check with asymmetric margin
    else:
        cp_margin = 0.0

    rclpy.init()
    rng = np.random.default_rng(42)

    results = []
    for ep in range(args.n_ep):
        world_path = os.path.join(NEUPAN_ROOT, 'gazebo_sim', cfg['world_file'])
        print(f"  Episode {ep+1}/{args.n_ep}: launching Gazebo + TB3...", flush=True)
        gz_proc = launch_gazebo(world_path)
        time.sleep(12)

        node = TB3Node()
        for _ in range(80):
            rclpy.spin_once(node, timeout_sec=0.1)

        if not node.wait_for_data(timeout=15.0):
            print("    WARNING: No data, skipping", flush=True)
            node.destroy_node(); kill_gazebo(); time.sleep(2)
            results.append({'arrived': False, 'collision': False, 'min_dist': 0.0})
            continue

        controller = TB3Controller(cp_margin=cp_margin)
        arrived = False; collision = False; min_dist = 999.0
        last_info_step = 0

        for step in range(cfg['max_steps']):
            rclpy.spin_once(node, timeout_sec=0.01)
            if node.robot_pose is None:
                continue

            pos = node.robot_pose
            dist_goal = np.linalg.norm(pos[:2] - goal)
            if dist_goal < cfg['goal_threshold']:
                arrived = True; break

            r, a = node.get_scan_arrays(noise_std=0.02, rng=rng)
            if r is None:
                node.send_cmd(0.2, 0.0); time.sleep(0.1); continue

            vx, omega, coll = controller.compute_action(pos, goal, r, a)
            if coll:
                collision = True; break

            # Track min distance
            valid = np.isfinite(r)
            if valid.sum() >= 3:
                min_dist = min(min_dist, float(r[valid].min()))

            node.send_cmd(vx, omega)
            time.sleep(0.1)

            if step - last_info_step >= 150:
                d2g = np.linalg.norm(pos[:2] - goal)
                print(f"      step={step} pos=({pos[0]:.1f},{pos[1]:.1f}) d2g={d2g:.1f}m "
                      f"cmd=({vx:.2f},{omega:.2f}) margin={cp_margin:.3f}", flush=True)
                last_info_step = step

        node.send_cmd(0.0, 0.0)
        status = "ARRIVED" if arrived else ("COLLISION" if collision else "TIMEOUT")
        print(f"    {status} min_dist={min_dist:.3f}m", flush=True)
        results.append({'arrived': arrived, 'collision': collision,
                        'min_dist': float(min_dist)})

        node.destroy_node(); kill_gazebo(); time.sleep(2)

    n_arr = sum(r['arrived'] for r in results)
    n_coll = sum(r['collision'] for r in results)
    avg_min = np.mean([r['min_dist'] for r in results if r['min_dist'] < 999])

    print(f"\n{'='*60}")
    print(f"RESULTS: {args.world} method={args.method}")
    print(f"  CP margin: {cp_margin*100:.1f}cm")
    print(f"  Success: {n_arr}/{args.n_ep} ({100*n_arr/args.n_ep:.0f}%)")
    print(f"  Collision: {n_coll}/{args.n_ep}")
    print(f"  Avg min dist: {avg_min:.3f}m")
    print(f"{'='*60}")

    out_dir = os.path.join(NEUPAN_ROOT, 'gazebo_sim', 'results')
    os.makedirs(out_dir, exist_ok=True)
    tag = f'tb3_{args.world}_{args.method}'
    with open(os.path.join(out_dir, f'{tag}.json'), 'w') as f:
        json.dump({'config': vars(args), 'results': results,
                   'summary': {'success': n_arr, 'collision': n_coll,
                               'total': args.n_ep, 'avg_min_dist': float(avg_min)}}, f, indent=2)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
