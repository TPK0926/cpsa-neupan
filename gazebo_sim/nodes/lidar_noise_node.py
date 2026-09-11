#!/usr/bin/env python3
"""ROS2 node: Inject Gaussian noise into LaserScan messages.

Subscribes: /robot/scan (sensor_msgs/LaserScan)
Publishes:  /robot/scan_noisy (sensor_msgs/LaserScan)

Parameters:
  noise_std (double): standard deviation of Gaussian noise in meters (default: 0.0)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np


class LidarNoiseNode(Node):
    def __init__(self):
        super().__init__('lidar_noise_injector')
        self.declare_parameter('noise_std', 0.0)
        self.noise_std = self.get_parameter('noise_std').value
        self.rng = np.random.default_rng(42)
        self.sub = self.create_subscription(
            LaserScan, '/robot/scan', self.callback, 10)
        self.pub = self.create_publisher(
            LaserScan, '/robot/scan_noisy', 10)
        self.get_logger().info(
            f'LidarNoiseNode started, noise_std={self.noise_std:.3f}m')

    def callback(self, msg: LaserScan):
        noisy = LaserScan()
        noisy.header = msg.header
        noisy.angle_min = msg.angle_min
        noisy.angle_max = msg.angle_max
        noisy.angle_increment = msg.angle_increment
        noisy.time_increment = msg.time_increment
        noisy.scan_time = msg.scan_time
        noisy.range_min = msg.range_min
        noisy.range_max = msg.range_max

        ranges = np.array(msg.ranges, dtype=np.float64)
        if self.noise_std > 0:
            noise = self.rng.normal(0, self.noise_std, size=ranges.shape)
            ranges = ranges + noise
            # Clamp to sensor limits
            ranges = np.clip(ranges, msg.range_min, msg.range_max)
            # Set inf values back
            inf_mask = np.isinf(msg.ranges)
            ranges[inf_mask] = msg.range_max

        noisy.ranges = ranges.tolist()
        noisy.intensities = msg.intensities
        self.pub.publish(noisy)


def main(args=None):
    rclpy.init(args=args)
    node = LidarNoiseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
