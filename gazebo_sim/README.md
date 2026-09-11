# Gazebo Simulation for NeuPAN + TB3

## 环境

```bash
# 每次新终端先执行
source /opt/ros/humble/setup.bash
source <ros2_ws>/install/setup.bash
cd <repo-root>
export GAZEBO_MODEL_PATH=<ros2_ws>/install/turtlebot3_gazebo/share/turtlebot3_gazebo/models:/usr/share/gazebo-11/models
```

Python 必须用 py310 环境（ROS2 Humble 需要 Python 3.10）：

```bash
python
```

## 快速启动

```bash
# headless 模式（服务器，无 GUI）
python gazebo_sim/nodes/tb3_nav_loop.py \
    --world tb3_irs_convex --n_ep 3 --method vanilla

# GUI 模式（桌面，可视化）
python gazebo_sim/nodes/tb3_nav_loop.py \
    --world tb3_irs_convex --n_ep 1 --method vanilla --gui
```

## 可选择的世界

| 参数 | 场景 | 大小 |
|------|------|------|
| `tb3_irs_convex` | 圆形 + 多边形障碍物 | 5×5m |
| `tb3_irs_corridor` | 走廊 + 内部障碍物 | 8.4×2.8m |
| `tb3_irs_nonobs` | 不规则多边形 | 5×5m |
| `tb3_irs_narrow` | 窄通道 (0.8m) | 6×0.8m |
| `tb3_open` | 空旷直道 | 25×3m |

世界文件在 `gazebo_sim/worlds/`。

## 可选择的方法

| 参数 | 含义 |
|------|------|
| `vanilla` | NeuPAN 基线，无校准 |
| `cp_global` | 固定 q_hat 全局 CP |
| `cpsa_v4` | CPSA 风险网络 + 逐点自适应 margin |

## 关键参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--world` | 场景名称 | `tb3_irs_corridor` |
| `--n_ep` | 每配置 episode 数 | 10 |
| `--method` | 方法 | `vanilla` |
| `--noise_std` | 激光噪声标准差 (m) | 0.0 |
| `--gui` | 开启 Gazebo GUI | false |

## 目录结构

```
gazebo_sim/
├── README.md
├── nodes/
│   ├── tb3_nav_loop.py          ← 主实验脚本（NeuPAN 完整管线）
│   ├── tb3_compare5.py          ← 5 方法对比脚本
│   ├── tb3_reactive_nav.py      ← 反应式控制器（废弃）
│   └── gazebo_nav_loop.py       ← 最早的自定义机器人脚本（废弃）
├── worlds/
│   ├── tb3_open.world           ← 空旷走廊
│   ├── tb3_irs_convex.world     ← 圆形+多边形
│   ├── tb3_irs_corridor.world   ← 走廊+内部障碍
│   ├── tb3_irs_nonobs.world     ← 不规则多边形
│   └── tb3_irs_narrow.world     ← 窄通道
├── models/                      ← 自定义机器人模型（废弃，改用 TB3）
├── results/                     ← 实验输出 JSON
└── generate_irsim_worlds.py     ← IR-SIM 场景生成器
```

## 修改指南

### 改场景

1. 编辑 `gazebo_sim/worlds/tb3_xxx.world`（SDF 格式）
2. 或者修改 `gazebo_sim/generate_irsim_worlds.py` 重新生成

障碍物是 `<model>` 标签，包含 `<collision>` 和 `<visual>`。例如加一个柱子：

```xml
<model name="pillar"><static>true</static><pose>2.0 3.0 0.5 0 0 0</pose>
  <link name="link">
    <collision name="c"><geometry><cylinder><radius>0.15</radius><length>1.0</length></cylinder></geometry></collision>
    <visual name="v"><geometry><cylinder><radius>0.15</radius><length>1.0</length></cylinder></geometry></visual>
  </link>
</model>
```

### 改 planner 参数

编辑 `example/corridor/diff/planner_tb3.yaml`：

| 参数 | 作用 |
|------|------|
| `robot.length/width` | TB3 尺寸（0.14×0.14） |
| `ipath.waypoints` | 参考路径 |
| `collision_threshold` | DUNE 碰撞停止阈值（0.01=几乎禁用） |
| `adjust.d_min` | NRMP 最小距离约束 |
| `pan.dune_max_num` | DUNE 处理点数 |
| `pan.dune_checkpoint` | DUNE 模型路径 |

## 重新训练 DUNE

```bash
python train_dune_tb3.py
```

模型保存到 `example/model/tb3_default/`。

## 查看结果

```bash
ls gazebo_sim/results/          # JSON 文件
cat gazebo_sim/results/tb3_xxx.json | python -m json.tool
```

## 停止实验

```bash
killall -9 gzserver gzclient   # 强制杀死 Gazebo
```

## 常见问题

**机器人不动**: DUNE false-stop → planner 返回零速度。检查 `planner_tb3.yaml` 的 `collision_threshold <= 0.01`。

**Gazebo 启动失败 "Address already in use"**: 有残留 gzserver 进程，执行 `killall -9 gzserver gzclient`。

**ImportError rclpy**: 用了错误的 Python，必须用 py310 环境。
