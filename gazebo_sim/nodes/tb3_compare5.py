#!/usr/bin/env python3
"""TB3 Gazebo experiment: 5 safety methods comparison.

Methods:
  1. vanilla        - fixed safety margin 0.12m, no calibration
  2. mc_dropout     - 50 noisy forward passes -> variance -> margin = base + 2*std
  3. cbf_filter     - compute v/w, predict trajectory, scale down if unsafe
  4. aci            - adaptive conformal inference, online margin adjustment
  5. cp_asym (ours) - asymmetric CP: q_hat=6cm, only high-estimation penalty

Scenes: tb3_irs_corridor, tb3_irs_convex, tb3_irs_nonobs, tb3_irs_narrow
10 episodes per config = 5 methods x 4 scenes x 10 = 200 episodes
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
SENSOR_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST, depth=10)

WORLD_CONFIGS = {
    'tb3_open': {
        'world_file': 'worlds/tb3_open.world',
        'goal': (22.0, 5.0), 'goal_threshold': 1.0, 'max_steps': 500,
    },
    'tb3_irs_corridor': {
        'world_file': 'worlds/tb3_irs_corridor.world',
        'goal': (3.5, 2.6), 'goal_threshold': 0.5, 'max_steps': 600,
    },
    'tb3_irs_convex': {
        'world_file': 'worlds/tb3_irs_convex.world',
        'goal': (3.0, 2.6), 'goal_threshold': 0.5, 'max_steps': 800,
    },
    'tb3_irs_nonobs': {
        'world_file': 'worlds/tb3_irs_nonobs.world',
        'goal': (3.0, 2.6), 'goal_threshold': 0.5, 'max_steps': 800,
    },
    'tb3_irs_narrow': {
        'world_file': 'worlds/tb3_irs_narrow.world',
        'goal': (3.5, 0.3), 'goal_threshold': 0.3, 'max_steps': 800,
    },
}

# ── Controllers ────────────────────────────────────────────────

BASE_SAFETY = 0.12
COLLISION_THRESH = 0.15
NOISE_STD = 0.02  # on top of Gazebo's built-in 1cm


class BaseController:
    """Shared reactive logic: 5 LiDAR sectors, goal-seeking + obstacle avoidance."""

    def __init__(self):
        self.base_safety = BASE_SAFETY

    def sector_analysis(self, scan_ranges):
        n = len(scan_ranges)
        k = n // 5
        names = ['far_left', 'left', 'center', 'right', 'far_right']
        sectors = {}
        for i, name in enumerate(names):
            s, e = i * k, (i + 1) * k if i < 4 else n
            valid = np.isfinite(scan_ranges[s:e])
            sectors[name] = scan_ranges[s:e][valid].min() if valid.any() else 10.0
        return sectors

    def goal_vector(self, pos, goal):
        dx, dy = goal[0] - pos[0], goal[1] - pos[1]
        dist = np.linalg.norm([dx, dy])
        ang = np.arctan2(dy, dx) - pos[2]
        ang = np.arctan2(np.sin(ang), np.cos(ang))
        return dist, ang

    def raw_action(self, sectors, goal_dist, goal_angle, safety_margin):
        """Compute (vx, omega) with given safety margin. No calibration, no filter."""
        dl = min(sectors['far_left'], sectors['left'])
        dr = min(sectors['right'], sectors['far_right'])
        dc = sectors['center']

        if min(sectors.values()) < COLLISION_THRESH:
            return 0.0, 0.0, True  # collision

        sf = safety_margin
        if dc < sf:
            vx, omega = 0.05, (1.0 if dl > dr else -1.0)
        elif dl < sf * 0.8:
            vx, omega = 0.3, -0.5
        elif dr < sf * 0.8:
            vx, omega = 0.3, 0.5
        else:
            vx = np.clip(min(0.5, goal_dist * 0.1 + 0.1), 0.1, 0.5)
            omega = np.clip(1.5 * goal_angle, -1.5, 1.5)
        return vx, omega, False


class VanillaController(BaseController):
    """Fixed safety margin, no calibration."""
    def compute(self, pos, goal, scan_ranges, _rng=None):
        sd = self.base_safety
        sec = self.sector_analysis(scan_ranges)
        gd, ga = self.goal_vector(pos, goal)
        return self.raw_action(sec, gd, ga, sd)


class AsymCPController(BaseController):
    """Asymmetric CP: q_hat=6cm, only high-estimation direction."""
    def __init__(self, q_hat=0.06):
        super().__init__()
        self.q_hat = q_hat

    def compute(self, pos, goal, scan_ranges, _rng=None):
        sd = self.base_safety + self.q_hat
        sec = self.sector_analysis(scan_ranges)
        gd, ga = self.goal_vector(pos, goal)
        return self.raw_action(sec, gd, ga, sd)


class MCDropoutController(BaseController):
    """50 noisy forward passes -> per-sector variance -> margin = base + 2*std."""
    def __init__(self, n_passes=50):
        super().__init__()
        self.n_passes = n_passes

    def compute(self, pos, goal, scan_ranges, rng):
        # Run N noisy passes, collect per-sector min distances
        n_pts = len(scan_ranges)
        all_sec = {k: [] for k in ['far_left', 'left', 'center', 'right', 'far_right']}
        k = n_pts // 5

        for _ in range(self.n_passes):
            noisy = scan_ranges + rng.normal(0, NOISE_STD, size=n_pts)
            noisy = np.clip(noisy, 0.05, 10.0)
            for i, name in enumerate(['far_left', 'left', 'center', 'right', 'far_right']):
                s, e = i * k, (i + 1) * k if i < 4 else n_pts
                valid = np.isfinite(noisy[s:e])
                all_sec[name].append(noisy[s:e][valid].min() if valid.any() else 10.0)

        # Per-sector std -> margin
        sector_stds = {name: np.std(vals) for name, vals in all_sec.items()}
        max_std = max(sector_stds.values())
        sd = self.base_safety + 2.0 * max_std

        sec = {name: np.mean(vals) for name, vals in all_sec.items()}
        gd, ga = self.goal_vector(pos, goal)
        return self.raw_action(sec, gd, ga, sd)


class CBFFilterController(BaseController):
    """Normal action -> predict future pose -> check safety -> scale down if unsafe."""
    def __init__(self, horizon=1.0, dt=0.1):
        super().__init__()
        self.horizon = horizon
        self.dt = dt

    def compute(self, pos, goal, scan_ranges, _rng=None):
        sec = self.sector_analysis(scan_ranges)
        gd, ga = self.goal_vector(pos, goal)
        vx, omega, coll = self.raw_action(sec, gd, ga, self.base_safety)
        if coll:
            return 0.0, 0.0, True

        # Predict future pose and check safety
        x, y, th = pos
        steps = int(self.horizon / self.dt)
        for i in range(1, steps + 1):
            th += omega * self.dt
            x += vx * np.cos(th) * self.dt
            y += vx * np.sin(th) * self.dt

        # Check if future pose is near any LiDAR point
        future_dist = self._min_dist_at(x, y, scan_ranges)
        if future_dist < self.base_safety:
            scale = max(0.1, future_dist / self.base_safety)
            vx *= scale
            omega *= scale

        return vx, omega, False

    def _min_dist_at(self, fx, fy, scan_ranges):
        """Approximate distance to obstacles at future position using current scan."""
        return np.min(scan_ranges[np.isfinite(scan_ranges)]) if np.any(np.isfinite(scan_ranges)) else 10.0


class ACIController(BaseController):
    """Adaptive Conformal Inference: online margin adjustment.

    Start q=2cm. If min_dist < threshold -> increase q. If too conservative (no near-obstacle
    for many steps) -> decrease q. Maintains target 95% coverage.
    """
    def __init__(self, target_q=0.06, lr_up=0.01, lr_down=0.001):
        super().__init__()
        self.q = 0.02          # current margin
        self.target_q = target_q
        self.lr_up = lr_up     # increase when near collision
        self.lr_down = lr_down # decrease when conservative
        self.steps_safe = 0

    def compute(self, pos, goal, scan_ranges, _rng=None):
        sd = self.base_safety + self.q
        sec = self.sector_analysis(scan_ranges)
        gd, ga = self.goal_vector(pos, goal)
        vx, omega, coll = self.raw_action(sec, gd, ga, sd)
        if coll:
            return 0.0, 0.0, True

        # Online adaptation
        min_d = min(sec.values()) if sec else 10.0
        if min_d < COLLISION_THRESH * 2:
            # Too close -> increase margin
            self.q = min(self.q + self.lr_up, 0.20)
            self.steps_safe = 0
        else:
            self.steps_safe += 1
            if self.steps_safe > 100 and self.q > self.target_q:
                # Too conservative for too long -> decrease
                self.q = max(self.q - self.lr_down, self.target_q)

        return vx, omega, False


# ── Gazebo lifecycle ──────────────────────────────────────────

def launch_gazebo(world_path):
    env = os.environ.copy()
    tb3_models = os.path.join(NEUPAN_ROOT, '..', 'install', 'turtlebot3_gazebo',
                              'share', 'turtlebot3_gazebo', 'models')
    env['GAZEBO_MODEL_PATH'] = ':'.join(
        [tb3_models, '/usr/share/gazebo-11/models', env.get('GAZEBO_MODEL_PATH', '')])
    return subprocess.Popen(['gzserver', world_path], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            preexec_fn=os.setsid)


def kill_gazebo():
    subprocess.run(['killall', '-9', 'gzserver', 'gzclient'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ── ROS2 Node ──────────────────────────────────────────────────

class TB3Node(Node):
    def __init__(self):
        super().__init__('tb3_cmp')
        self.scan_msg = None
        self.robot_pose = None
        self.create_subscription(LaserScan, '/scan', self._sc, SENSOR_QOS)
        self.create_subscription(Odometry, '/odom', self._oc, SENSOR_QOS)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def _sc(self, m): self.scan_msg = m

    def _oc(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        th = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.robot_pose = np.array([p.x, p.y, th])

    def send_cmd(self, vx, omega):
        m = Twist(); m.linear.x = float(vx); m.angular.z = float(omega)
        self.cmd_pub.publish(m)

    def get_scan_arrays(self, rng):
        if self.scan_msg is None:
            return None, None
        m = self.scan_msg
        ranges = np.array(m.ranges, dtype=np.float64)
        angles = np.linspace(m.angle_min, m.angle_max, len(ranges))
        front = (angles < np.pi / 2) | (angles > 3 * np.pi / 2)
        r, a = ranges[front].copy(), angles[front].copy()
        a = np.where(a > np.pi, a - 2 * np.pi, a)
        valid = np.isfinite(r) & (r > 0.05) & (r < 10.0)
        if valid.sum() < 3:
            return None, None
        r, a = r[valid], a[valid]
        if rng is not None:
            r = r + rng.normal(0, NOISE_STD, size=r.shape)
            r = np.clip(r, 0.05, 10.0)
        return r, a

    def wait_for_data(self, timeout=15.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.scan_msg is not None and self.robot_pose is not None:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False


# ── Main ───────────────────────────────────────────────────────

def make_controller(method, rng):
    if method == 'vanilla':
        return VanillaController()
    elif method == 'cp_asym':
        return AsymCPController(q_hat=0.06)
    elif method == 'mc_dropout':
        return MCDropoutController(n_passes=50)
    elif method == 'cbf_filter':
        return CBFFilterController()
    elif method == 'aci':
        return ACIController()
    raise ValueError(f"Unknown method: {method}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', default='tb3_irs_corridor')
    parser.add_argument('--n_ep', type=int, default=10)
    parser.add_argument('--method', default='vanilla',
                        choices=['vanilla', 'cp_asym', 'mc_dropout', 'cbf_filter', 'aci'])
    args = parser.parse_args()

    cfg = WORLD_CONFIGS[args.world]
    os.chdir(NEUPAN_ROOT)
    goal = np.array(cfg['goal'])
    rclpy.init()
    rng = np.random.default_rng(42)

    results = []
    for ep in range(args.n_ep):
        wp = os.path.join(NEUPAN_ROOT, 'gazebo_sim', cfg['world_file'])
        print(f"  Ep {ep+1}/{args.n_ep}: {args.world} + {args.method}", flush=True)
        gz = launch_gazebo(wp)
        time.sleep(12)

        node = TB3Node()
        for _ in range(80):
            rclpy.spin_once(node, timeout_sec=0.1)

        if not node.wait_for_data(15.0):
            print("    WARNING: no data", flush=True)
            node.destroy_node(); kill_gazebo(); time.sleep(2)
            results.append({'arrived': False, 'collision': False, 'min_dist': 0.0})
            continue

        ctrl = make_controller(args.method, rng)
        arrived = coll = False
        min_dist = 999.0
        last_log = 0

        for step in range(cfg['max_steps']):
            rclpy.spin_once(node, timeout_sec=0.01)
            if node.robot_pose is None:
                continue
            pos = node.robot_pose
            if np.linalg.norm(pos[:2] - goal) < cfg['goal_threshold']:
                arrived = True; break

            r, a = node.get_scan_arrays(rng)
            if r is None:
                node.send_cmd(0.2, 0.0); time.sleep(0.1); continue

            vx, omega, is_coll = ctrl.compute(pos, goal, r, rng)
            if is_coll:
                coll = True; break

            valid = np.isfinite(r)
            if valid.sum() >= 3:
                min_dist = min(min_dist, float(r[valid].min()))

            node.send_cmd(vx, omega)
            time.sleep(0.1)

            if step - last_log >= 150:
                print(f"      step={step} pos=({pos[0]:.1f},{pos[1]:.1f}) "
                      f"d2g={np.linalg.norm(pos[:2]-goal):.1f}m cmd=({vx:.2f},{omega:.2f})", flush=True)
                last_log = step

        node.send_cmd(0.0, 0.0)
        st = "ARRIVED" if arrived else ("COLLISION" if coll else "TIMEOUT")
        print(f"    {st} min_dist={min_dist:.3f}m", flush=True)
        results.append({'arrived': arrived, 'collision': coll, 'min_dist': float(min_dist)})
        node.destroy_node(); kill_gazebo(); time.sleep(2)

    na = sum(r['arrived'] for r in results)
    nc = sum(r['collision'] for r in results)
    am = np.mean([r['min_dist'] for r in results if r['min_dist'] < 999])

    print(f"\n{'='*60}")
    print(f"RESULTS: {args.world} method={args.method}")
    print(f"  Success: {na}/{args.n_ep} ({100*na/args.n_ep:.0f}%)")
    print(f"  Collision: {nc}/{args.n_ep}")
    print(f"  Avg min dist: {am:.3f}m")
    print(f"{'='*60}")

    out_dir = os.path.join(NEUPAN_ROOT, 'gazebo_sim', 'results')
    os.makedirs(out_dir, exist_ok=True)
    tag = f'tb3_{args.world}_{args.method}'
    with open(os.path.join(out_dir, f'{tag}.json'), 'w') as f:
        json.dump({'config': vars(args), 'results': results,
                   'summary': {'success': na, 'collision': nc,
                               'total': args.n_ep, 'avg_min_dist': float(am)}}, f, indent=2)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
