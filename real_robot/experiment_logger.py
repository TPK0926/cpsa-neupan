#!/usr/bin/env python
"""
真机实验数据记录器。

用法 (在小车上):
  python experiment_logger.py --method vanilla --scene corridor --episode 1

然后给机器人发 goal 让它跑。跑完按 Ctrl+C，自动保存 JSON。

每轮数据:
  method, scene, episode, outcome, steps, min_dist, path_len, elapsed_s
"""

import rospy
import json
import os
import time
import argparse
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from collections import deque


class ExperimentLogger:
    def __init__(self, method, scene, max_steps=500, collision_thresh=0.15):
        self.method = method
        self.scene = scene
        self.max_steps = max_steps
        self.collision_thresh = collision_thresh

        # 状态
        self.outcome = 'timeout'
        self.min_dist = float('inf')
        self.start_pose = None
        self.current_pose = None
        self.step_count = 0
        self.episode_active = False
        self.path_len = 0.0
        self.prev_x = None
        self.prev_y = None

        # ROS
        rospy.init_node('experiment_logger', anonymous=True)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.goal_sub = rospy.Subscriber('/neupan_goal', PoseStamped, self.goal_callback)
        self.min_dist_sub = rospy.Subscriber('/min_distance', Float32, self.min_dist_callback)

        self.rate = rospy.Rate(10)

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        self.current_pose = (p.x, p.y)
        if self.episode_active and self.prev_x is not None:
            dx = p.x - self.prev_x
            dy = p.y - self.prev_y
            self.path_len += np.sqrt(dx*dx + dy*dy)
        self.prev_x = p.x
        self.prev_y = p.y

        if self.start_pose is None:
            self.start_pose = (p.x, p.y)

    def goal_callback(self, msg):
        if not self.episode_active:
            rospy.loginfo("[LOGGER] New episode started!")
            self.episode_active = True
            self.start_time = time.time()
            self.min_dist = float('inf')
            self.path_len = 0.0
            self.step_count = 0
            self.outcome = 'timeout'
            if self.current_pose:
                self.start_pose = self.current_pose

    def min_dist_callback(self, msg):
        md = msg.data
        if md < self.min_dist:
            self.min_dist = md

    def run(self, n_episodes):
        results = []
        ep = 0

        while ep < n_episodes and not rospy.is_shutdown():
            # 等待 episode 开始
            rospy.loginfo(f"[LOGGER] 等待第 {ep+1}/{n_episodes} 轮 goal ...")
            while not self.episode_active and not rospy.is_shutdown():
                self.rate.sleep()

            rospy.loginfo(f"[LOGGER] 第 {ep+1} 轮开始!")
            self.start_time = time.time()

            # 跑 episode
            while self.episode_active and not rospy.is_shutdown():
                self.step_count += 1

                # 碰撞检测
                if self.min_dist < self.collision_thresh:
                    self.outcome = 'collision'
                    self.episode_active = False
                    break

                # 超时
                if self.step_count > self.max_steps:
                    self.outcome = 'timeout'
                    self.episode_active = False
                    break

                self.rate.sleep()

            elapsed = time.time() - self.start_time

            # 记录本轮
            ep_result = {
                'ep': ep,
                'method': self.method,
                'scene': self.scene,
                'outcome': self.outcome,
                'steps': self.step_count,
                'min_dist': round(self.min_dist, 4),
                'path_len': round(self.path_len, 2),
                'elapsed_s': round(elapsed, 1),
            }
            results.append(ep_result)
            rospy.loginfo(f"[LOGGER] 第 {ep+1} 轮: {self.outcome} | "
                          f"steps={self.step_count} | minD={self.min_dist*100:.1f}cm | "
                          f"path={self.path_len:.1f}m | {elapsed:.0f}s")

            # 保存备份
            out_dir = os.path.join(os.environ.get('EXPERIMENT_DATA_DIR', os.path.expanduser('~/experiment_data')), self.scene)
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, f'{self.method}_results.json')
            with open(out_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            ep += 1

        rospy.loginfo(f"[LOGGER] 完成! 保存到 {out_file}")

        # 打印汇总
        arrived = sum(1 for r in results if r['outcome'] == 'arrived')
        coll = sum(1 for r in results if r['outcome'] == 'collision')
        rospy.loginfo(f"[LOGGER] 汇总: SR={arrived}/{n_episodes} ({arrived/n_episodes*100:.0f}%) "
                      f"collisions={coll}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, help='vanilla / cp_global / cpsa_v4')
    parser.add_argument('--scene', required=True, help='场景名')
    parser.add_argument('--episodes', type=int, default=10, help='轮数')
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--collision_thresh', type=float, default=0.15)
    args = parser.parse_args()

    logger = ExperimentLogger(
        method=args.method,
        scene=args.scene,
        max_steps=args.max_steps,
        collision_thresh=args.collision_thresh,
    )
    logger.run(args.episodes)
