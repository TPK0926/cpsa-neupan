#!/usr/bin/env python3
"""
LIMO NeuPAN ROS1 Navigation Node

Bridges ROS1 topics (/scan, /odom, /cmd_vel) to the NeuPAN differentiable
MPC planner with conformal risk calibration (Static CP + CPSA v4).

Hardware: AgileX LIMO (differential drive, 0.322m x 0.220m)
LiDAR: EAI X2L (360 deg FOV, 0.1-8m range)
ROS: Noetic (rospy)
Python: 3.8+ (system Python on Ubuntu 20.04)

Usage:
  source /opt/ros/noetic/setup.bash
  python limo_neupan_node.py --config config/limo_config.yaml --method cpsa_v4

Subscribes:
  /scan   (sensor_msgs/LaserScan)
  /odom   (nav_msgs/Odometry)

Publishes:
  /cmd_vel          (geometry_msgs/Twist)
  /neupan/status    (std_msgs/String)        - planner status
  /neupan/min_dist  (std_msgs/Float32)       - current min obstacle distance
"""

import sys
import os
import time
import json
import argparse
import signal
import traceback
import threading

import numpy as np
import yaml

# ── Add NeuPAN to path ──
NEUPAN_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, NEUPAN_ROOT)

# ── ROS1 imports ──
import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32, Header

# ── NeuPAN imports ──
from neupan import neupan
from neupan.configuration import np_to_tensor, tensor_to_np

# ── Constants ──
NODE_NAME = 'limo_neupan_nav'
CONTROL_RATE = 10  # Hz


class LimoNeuPANNode:
    """ROS1 node that bridges LIMO sensors to NeuPAN planner."""

    def __init__(self, config_path, method='cpsa_v4', noise_std=0.0,
                 epsilon=0.05, cp_checkpoint=None):
        rospy.init_node(NODE_NAME, anonymous=False)

        # Load config
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        self.method = method
        self.noise_std = noise_std
        self.epsilon = epsilon
        self.cp_checkpoint = cp_checkpoint

        # ── State ──
        self.scan_msg = None
        self.odom_msg = None
        self.robot_pose = None       # np.array([x, y, theta])
        self.robot_vel = np.zeros(2)
        self.lock = threading.Lock()

        # Episode state
        self.goal_pos = None         # np.array([x, y])
        self.episode_active = False
        self.episode_step = 0
        self.min_distance = float('inf')
        self.start_pose = None
        self.episode_results = []

        # LiDAR config
        lidar_cfg = self.cfg.get('lidar', {})
        self.scan_offset = lidar_cfg.get('scan_offset', [0, 0, 0])
        self.angle_range = lidar_cfg.get('use_angle_range', [-np.pi/2, np.pi/2])
        self.down_sample = lidar_cfg.get('down_sample', 1)

        # Safety config
        safety_cfg = self.cfg.get('safety', {})
        self.emergency_stop_range = safety_cfg.get('emergency_stop_range', 0.12)
        self.max_episode_steps = safety_cfg.get('max_episode_steps', 300)
        self.goal_threshold = safety_cfg.get('goal_arrival_threshold', 0.2)

        # Collision threshold from planner config
        self.collision_threshold = self.cfg.get('collision_threshold', 0.06)

        # ── Initialize NeuPAN planner ──
        rospy.loginfo("Loading NeuPAN planner...")
        self._init_planner()

        # ── ROS publishers ──
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.status_pub = rospy.Publisher('/neupan/status', String, queue_size=10)
        self.mindist_pub = rospy.Publisher('/neupan/min_dist', Float32, queue_size=10)

        # ── ROS subscribers ──
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self._scan_cb,
                                          queue_size=1)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self._odom_cb,
                                          queue_size=1)
        # Also support LIMO-specific topic names
        self.scan_sub2 = rospy.Subscriber('/limo/scan', LaserScan, self._scan_cb,
                                           queue_size=1)

        # ── Status ──
        self.rate = rospy.Rate(CONTROL_RATE)
        rospy.loginfo(f"LIMO NeuPAN node initialized. Method: {method}, "
                      f"Rate: {CONTROL_RATE}Hz")

    def _init_planner(self):
        """Initialize NeuPAN planner with the LIMO configuration."""
        plan_cfg = dict(self.cfg)
        plan_cfg['time_print'] = False
        plan_cfg['device'] = 'cpu'

        import tempfile
        tf = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(plan_cfg, tf)
        tf.close()
        self._tmp_config = tf.name

        self.planner = neupan.init_from_yaml(tf.name)

        # Enable risk calibration based on method
        if self.method in ('cp_global', 'cp_dw', 'cpsa_v4', 'cpsa_v4_zero_shot'):
            budget_strategy = {
                'cp_global': 'global',
                'cp_dw': 'distance_weighted',
                'cpsa_v4': 'cpsa',
                'cpsa_v4_zero_shot': 'cpsa',
            }.get(self.method, 'cpsa')

            cp_ckpt = self.cp_checkpoint or \
                'experiments/cp_head_output/cpsa_v4_universal.pth'

            env_map = {
                'corridor': 0, 'obstacle_field': 1,
                'narrow_passage': 4, 'dynamic_obstacle': 3,
            }
            env_type = env_map.get(self._current_env(), 0)

            rospy.loginfo(f"Enabling risk calibration: budget={budget_strategy}, "
                          f"ckpt={cp_ckpt}, env_type={env_type}")

            self.planner.enable_risk_calibration(
                budget_strategy=budget_strategy,
                cp_checkpoint=cp_ckpt,
                collision_q_hat=self.epsilon,
                noise_std=self.noise_std,
                env_type=env_type,
            )

    def _current_env(self):
        return getattr(self, '_env_name', 'corridor')

    def _scan_cb(self, msg):
        with self.lock:
            self.scan_msg = msg

    def _odom_cb(self, msg):
        with self.lock:
            self.odom_msg = msg
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            theta = np.arctan2(2 * (q.w * q.z + q.x * q.y),
                               1 - 2 * (q.y * q.y + q.z * q.z))
            self.robot_pose = np.array([p.x, p.y, theta])
            v = msg.twist.twist
            self.robot_vel = np.array([v.linear.x, v.angular.z])

    def _get_scan_dict(self):
        """Convert ROS LaserScan to NeuPAN-compatible scan dict."""
        with self.lock:
            if self.scan_msg is None:
                return None
            msg = self.scan_msg

        return {
            'ranges': list(msg.ranges),
            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,
            'range_min': msg.range_min,
            'range_max': msg.range_max,
        }

    def _check_emergency_stop(self):
        """Check if any obstacle is within emergency stop range."""
        with self.lock:
            if self.scan_msg is None:
                return True  # No data = stop
            ranges = np.array(self.scan_msg.ranges)

        valid = np.isfinite(ranges) & (ranges > 0.01)
        if not valid.any():
            return False

        min_range = float(ranges[valid].min())
        return min_range < self.emergency_stop_range

    def _compute_min_distance(self):
        """Compute minimum distance to any obstacle from scan."""
        with self.lock:
            if self.scan_msg is None:
                return float('inf')
            ranges = np.array(self.scan_msg.ranges)

        valid = np.isfinite(ranges) & (ranges > 0.01)
        if not valid.any():
            return float('inf')

        return float(ranges[valid].min())

    def publish_cmd_vel(self, vx, omega):
        """Publish velocity command."""
        msg = Twist()
        msg.linear.x = float(vx)
        msg.angular.z = float(omega)
        self.cmd_pub.publish(msg)

    def publish_status(self, status):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)

    def stop(self):
        """Emergency stop - publish zero velocity."""
        self.publish_cmd_vel(0.0, 0.0)
        self.publish_status("STOPPED")

    def wait_for_sensors(self, timeout=10.0):
        """Wait for initial sensor data."""
        t0 = time.time()
        while not rospy.is_shutdown():
            with self.lock:
                if self.scan_msg is not None and self.robot_pose is not None:
                    rospy.loginfo("Sensor data received.")
                    return True
            if time.time() - t0 > timeout:
                rospy.logerr(f"No sensor data after {timeout}s. "
                             f"Check LiDAR (/scan) and odometry (/odom) topics.")
                return False
            self.rate.sleep()
        return False

    def set_goal(self, goal_xy):
        """Set navigation goal position."""
        self.goal_pos = np.array(goal_xy[:2], dtype=np.float64)

    def start_episode(self, goal_xy, env_name='corridor'):
        """Start a new navigation episode."""
        self.set_goal(goal_xy)
        self._env_name = env_name
        self.episode_active = True
        self.episode_step = 0
        self.min_distance = float('inf')

        with self.lock:
            if self.robot_pose is not None:
                self.start_pose = self.robot_pose.copy()
            else:
                self.start_pose = None

        self.publish_status(f"RUNNING:{env_name}")

    def step(self):
        """Execute one control cycle. Returns (action, info)."""
        if not self.episode_active:
            return np.zeros((2, 1)), {'stop': True, 'arrive': False}

        # Emergency stop check
        if self._check_emergency_stop():
            rospy.logwarn_throttle(2.0, "EMERGENCY STOP: obstacle too close!")
            self.stop()
            self.episode_active = False
            return np.zeros((2, 1)), {'stop': True, 'arrive': False, 'collision': True}

        # Check collision
        min_d = self._compute_min_distance()
        self.min_distance = min(self.min_distance, min_d)
        if min_d < self.collision_threshold:
            rospy.logwarn(f"COLLISION: min_dist={min_d:.3f}m < "
                          f"threshold={self.collision_threshold:.3f}m")
            self.stop()
            self.episode_active = False
            return np.zeros((2, 1)), {'stop': True, 'arrive': False, 'collision': True}

        # Publish min distance for monitoring
        self.mindist_pub.publish(Float32(data=float(min_d)))

        # Check arrival
        with self.lock:
            pose = self.robot_pose

        if pose is not None and self.goal_pos is not None:
            dist_to_goal = np.linalg.norm(pose[:2] - self.goal_pos)
            if dist_to_goal < self.goal_threshold:
                rospy.loginfo(f"ARRIVED: dist_to_goal={dist_to_goal:.3f}m")
                self.stop()
                self.episode_active = False
                return np.zeros((2, 1)), {'stop': True, 'arrive': True}

        # Check timeout
        self.episode_step += 1
        if self.episode_step >= self.max_episode_steps:
            rospy.logwarn(f"TIMEOUT after {self.max_episode_steps} steps")
            self.stop()
            self.episode_active = False
            return np.zeros((2, 1)), {'stop': True, 'arrive': False, 'timeout': True}

        # ── Run NeuPAN planner ──
        try:
            scan_dict = self._get_scan_dict()
            if scan_dict is None or pose is None:
                self.rate.sleep()
                return np.zeros((2, 1)), {'stop': False, 'arrive': False}

            # Convert scan to obstacle points (world frame)
            points = self.planner.scan_to_point(
                pose, scan_dict,
                scan_offset=self.scan_offset,
                angle_range=self.angle_range,
                down_sample=self.down_sample,
            )

            if points is None or points.shape[1] < 3:
                # Not enough points - move forward slowly or stop
                self.publish_cmd_vel(0.3, 0.0)
                return np.array([[0.3], [0.0]]), {'stop': False, 'arrive': False}

            # Run planner forward pass
            state = pose.reshape(3, 1).astype(np.float64)
            action, info = self.planner(state, points, None)

            if info.get('arrive'):
                self.stop()
                self.episode_active = False
                return action, info

            if info.get('stop'):
                self.stop()
                self.episode_active = False
                return action, info

            # Extract velocity command
            if isinstance(action, np.ndarray) and action.size >= 2:
                vx = float(action.flat[0])
                omega = float(action.flat[1])
            else:
                vx, omega = 0.3, 0.0

            # Clamp to LIMO safe range
            max_lin = self.cfg['robot']['max_speed'][0]
            max_ang = self.cfg['robot']['max_speed'][1]
            vx = np.clip(vx, -max_lin, max_lin)
            omega = np.clip(omega, -max_ang, max_ang)

            self.publish_cmd_vel(vx, omega)

            return np.array([[vx], [omega]]), info

        except Exception as e:
            rospy.logerr(f"Planner error: {e}")
            traceback.print_exc()
            self.stop()
            self.episode_active = False
            return np.zeros((2, 1)), {'stop': True, 'arrive': False, 'error': str(e)}

    def run_episode(self, goal_xy, env_name='corridor',
                    record_trajectory=False):
        """Run a full navigation episode to completion.

        Returns:
            dict with keys: arrived, collision, timeout, error, min_dist,
                           trajectory (if record_trajectory=True),
                           steps, final_pose
        """
        self.start_episode(goal_xy, env_name)

        trajectory = [] if record_trajectory else None
        start_time = time.time()

        while self.episode_active and not rospy.is_shutdown():
            action, info = self.step()

            if record_trajectory:
                with self.lock:
                    if self.robot_pose is not None:
                        trajectory.append(self.robot_pose.copy().tolist())

            if info.get('stop'):
                break

            self.rate.sleep()

        elapsed = time.time() - start_time

        with self.lock:
            final_pose = self.robot_pose.copy() if self.robot_pose is not None else None

        result = {
            'arrived': info.get('arrive', False),
            'collision': info.get('collision', False),
            'timeout': info.get('timeout', False),
            'error': info.get('error', None),
            'min_dist': float(self.min_distance),
            'steps': self.episode_step,
            'elapsed': elapsed,
            'final_pose': final_pose.tolist() if final_pose is not None else None,
            'trajectory': trajectory,
        }
        return result

    def cleanup(self):
        """Clean shutdown."""
        self.stop()
        if hasattr(self, '_tmp_config') and os.path.exists(self._tmp_config):
            os.unlink(self._tmp_config)
        rospy.loginfo("LIMO NeuPAN node shut down.")


def main():
    parser = argparse.ArgumentParser(
        description='LIMO NeuPAN ROS1 Navigation Node')
    parser.add_argument('--config', default=None,
                        help='Path to LIMO config YAML')
    parser.add_argument('--method', default='cpsa_v4',
                        choices=['vanilla', 'cp_global', 'cp_dw',
                                 'cpsa_v4', 'cpsa_v4_zero_shot'])
    parser.add_argument('--noise_std', type=float, default=0.0,
                        help='Artificial noise std for testing (m)')
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='Conformal prediction error rate')
    parser.add_argument('--cp_checkpoint', default=None,
                        help='Path to CPSA/CP checkpoint')
    parser.add_argument('--goal_x', type=float, default=4.0,
                        help='Goal X position (m)')
    parser.add_argument('--goal_y', type=float, default=0.0,
                        help='Goal Y position (m)')
    parser.add_argument('--env', default='corridor',
                        choices=['corridor', 'obstacle_field',
                                 'narrow_passage', 'dynamic_obstacle'])
    parser.add_argument('--record', action='store_true',
                        help='Record trajectory')
    parser.add_argument('--output', default=None,
                        help='Output JSON file for results')
    args = parser.parse_args()

    # Default config path
    if args.config is None:
        args.config = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'config', 'limo_config.yaml')

    rospy.loginfo(f"Starting LIMO NeuPAN node. Config: {args.config}")
    rospy.loginfo(f"Method: {args.method}, Goal: ({args.goal_x}, {args.goal_y})")

    node = LimoNeuPANNode(
        config_path=args.config,
        method=args.method,
        noise_std=args.noise_std,
        epsilon=args.epsilon,
        cp_checkpoint=args.cp_checkpoint,
    )

    # Handle Ctrl+C gracefully
    def shutdown_handler(signum, frame):
        rospy.loginfo("Shutdown signal received.")
        node.cleanup()
        rospy.signal_shutdown("User interrupt")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Wait for sensors
    if not node.wait_for_sensors(timeout=15.0):
        rospy.logerr("Failed to get sensor data. Exiting.")
        node.cleanup()
        return

    rospy.loginfo("Starting navigation episode...")
    result = node.run_episode(
        goal_xy=[args.goal_x, args.goal_y],
        env_name=args.env,
        record_trajectory=args.record,
    )

    # Print result
    status = ("ARRIVED" if result['arrived'] else
              "COLLISION" if result['collision'] else
              "TIMEOUT" if result['timeout'] else "ERROR")
    rospy.loginfo(f"Episode result: {status}")
    rospy.loginfo(f"  Min distance: {result['min_dist']:.3f}m")
    rospy.loginfo(f"  Steps: {result['steps']}, Elapsed: {result['elapsed']:.1f}s")

    if result.get('final_pose'):
        fp = result['final_pose']
        rospy.loginfo(f"  Final pose: ({fp[0]:.2f}, {fp[1]:.2f}, {fp[2]:.2f})")

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({
                'config': {
                    'method': args.method,
                    'noise_std': args.noise_std,
                    'epsilon': args.epsilon,
                    'env': args.env,
                    'goal': [args.goal_x, args.goal_y],
                },
                'result': {k: v for k, v in result.items()
                           if k != 'trajectory'},
            }, f, indent=2)
        rospy.loginfo(f"Results saved to {args.output}")

    node.cleanup()


if __name__ == '__main__':
    main()
