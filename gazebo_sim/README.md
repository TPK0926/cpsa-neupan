# Gazebo Integration Reference

This folder contains ROS2/Gazebo integration source code for CPSA-NeuPAN development.

Included components:

- `nodes/`: ROS2 Python nodes for LaserScan/Odometry bridging, noise injection, closed-loop navigation tests, and CPSA margin publication.
- `launch/`: launch entry points for Gazebo-based validation.
- `models/`: lightweight Gazebo robot model assets used by the reference worlds.
- `worlds/`: small Gazebo world files used for integration checks.

This folder is not a full reproduction package. Public experiment configs, trained checkpoints, result logs, and robot deployment files are intentionally not included in this source-only repository. To run the nodes, provide your own planner YAML and compatible trained distance/safety checkpoints, then update the paths in the launch or command-line arguments.

Typical ROS2 prerequisites:

```bash
source /opt/ros/humble/setup.bash
pip install -e .
```

The scripts are kept mainly to show how LiDAR, odometry, Gazebo simulation, and CPSA-NeuPAN are connected at the source-code level.
