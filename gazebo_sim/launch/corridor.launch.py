"""Launch Gazebo with the CPSA-NeuPAN reference adapter.

This launch file is source-level integration code. Provide a compatible CPSA
checkpoint through the `cpsa_checkpoint` launch argument when running the
adapter with a trained model.
"""

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    noise_std = LaunchConfiguration('noise_std', default='0.0')
    cpsa_checkpoint = LaunchConfiguration('cpsa_checkpoint', default='')
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    return LaunchDescription([
        DeclareLaunchArgument('noise_std', default_value='0.0'),
        DeclareLaunchArgument('cpsa_checkpoint', default_value=''),
        ExecuteProcess(
            cmd=[
                'gazebo', '--verbose',
                os.path.join(pkg_dir, 'worlds', 'corridor.world'),
                '-s', 'libgazebo_ros_factory.so',
            ],
            output='screen',
        ),
        Node(
            package='gazebo_sim',
            executable='lidar_noise_node.py',
            name='lidar_noise',
            parameters=[{'noise_std': noise_std}],
            output='screen',
        ),
        Node(
            package='gazebo_sim',
            executable='cpsa_adapter_node.py',
            name='cpsa_adapter',
            parameters=[{
                'cpsa_checkpoint': cpsa_checkpoint,
                'noise_std': noise_std,
                'env_type': 0,
                'goal_x': 40.0,
                'goal_y': 40.0,
            }],
            output='screen',
        ),
    ])
