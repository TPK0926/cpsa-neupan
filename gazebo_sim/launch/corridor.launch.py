"""Launch file for Corridor environment in Gazebo with CPSA-v4.

Usage:
  ros2 launch gazebo_sim corridor.launch.py noise_std:=0.02
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    noise_std = LaunchConfiguration('noise_std', default='0.0')
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    return LaunchDescription([
        # Start Gazebo with corridor world
        ExecuteProcess(
            cmd=['gazebo', '--verbose',
                 os.path.join(pkg_dir, 'worlds', 'corridor.world'),
                 '-s', 'libgazebo_ros_factory.so'],
            output='screen'
        ),
        # Noise injection node
        Node(
            package='gazebo_sim',
            executable='lidar_noise_node.py',
            name='lidar_noise',
            parameters=[{'noise_std': noise_std}],
            output='screen'
        ),
        # CPSA-v4 adapter node
        Node(
            package='gazebo_sim',
            executable='cpsa_adapter_node.py',
            name='cpsa_adapter',
            parameters=[{
                'cpsa_checkpoint': os.path.join(
                    pkg_dir, 'experiments', 'cp_head_output', 'cpsa_v4_universal.pth'),
                'noise_std': noise_std,
                'env_type': 0,
                'goal_x': 40.0,
                'goal_y': 40.0,
            }],
            output='screen'
        ),
    ])
