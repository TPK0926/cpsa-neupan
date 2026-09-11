# 真机实验操作手册

## 一、目录结构

```
~/neupan_ws/src/neupan_ros/example/gazebo_limo/experiment_data/
├── corridor/          # 场景1: 直走廊
│   ├── vanilla.log
│   ├── cp_global.log
│   └── cpsa_v4.log
├── mixed/             # 场景2: 宽窄交替 (核心)
│   ├── vanilla.log
│   ├── cp_global.log
│   └── cpsa_v4.log
├── obstacle/          # 场景3: 散落障碍物
│   ├── vanilla.log
│   ├── cp_global.log
│   └── cpsa_v4.log
└── extract_results.py
```

实验前确认目录存在：

```bash
cd ~/neupan_ws/src/neupan_ros/example/gazebo_limo/experiment_data
mkdir -p corridor mixed obstacle
```

## 二、启动底盘

```bash
roslaunch limo_bringup <你的底盘launch>.launch
```

## 三、运行实验

环境变量（省打字）：

```bash
WS=~/neupan_ws/src/neupan_ros/example/gazebo_limo/experiment_data
```

### 场景1: 直走廊

| 方法 | 命令 |
|------|------|
| vanilla | `roslaunch neupan_ros neupan_cpsa.launch 2>&1 \| tee $WS/corridor/vanilla.log` |
| cp_global | `roslaunch neupan_ros neupan_cpsa.launch core_type:=cpsa_core.py method:=cp_global 2>&1 \| tee $WS/corridor/cp_global.log` |
| cpsa_v4 | `roslaunch neupan_ros neupan_cpsa.launch core_type:=cpsa_core.py method:=cpsa_v4 2>&1 \| tee $WS/corridor/cpsa_v4.log` |

### 场景2: 宽窄交替

| 方法 | 命令 |
|------|------|
| vanilla | `roslaunch neupan_ros neupan_cpsa.launch 2>&1 \| tee $WS/mixed/vanilla.log` |
| cp_global | `roslaunch neupan_ros neupan_cpsa.launch core_type:=cpsa_core.py method:=cp_global 2>&1 \| tee $WS/mixed/cp_global.log` |
| cpsa_v4 | `roslaunch neupan_ros neupan_cpsa.launch core_type:=cpsa_core.py method:=cpsa_v4 2>&1 \| tee $WS/mixed/cpsa_v4.log` |

### 场景3: 散落障碍物

| 方法 | 命令 |
|------|------|
| vanilla | `roslaunch neupan_ros neupan_cpsa.launch 2>&1 \| tee $WS/obstacle/vanilla.log` |
| cp_global | `roslaunch neupan_ros neupan_cpsa.launch core_type:=cpsa_core.py method:=cp_global 2>&1 \| tee $WS/obstacle/cp_global.log` |
| cpsa_v4 | `roslaunch neupan_ros neupan_cpsa.launch core_type:=cpsa_core.py method:=cpsa_v4 2>&1 \| tee $WS/obstacle/cpsa_v4.log` |

## 四、实验操作流程

每轮操作：

1. 启动对应方法的 launch 命令
2. 在 RVIZ 里点击 `2D Nav Goal` 设置目标点
3. 机器人自动导航，到达/碰撞后终端自动打印结果：
   ```
   [EP_RESULT] outcome=arrived | minD=0.3421m | path=5.2m | time=23s
   ```
4. 再次点击 `2D Nav Goal` 设置下一个目标点
5. 跑够轮数后 `Ctrl+C` 停止

**每场景每方法建议 10-15 轮，核心场景 mixed 建议跑 15 轮。**

## 五、提取实验数据

### 单个场景汇总

```bash
cd ~/neupan_ws/src/neupan_ros/example/gazebo_limo/experiment_data
python extract_results.py corridor/
```

输出示例：

```
vanilla.log:    SR=80% (8/10)   CR=0%   avgMinD=31.2cm
cp_global.log:  SR=90% (9/10)   CR=0%   avgMinD=32.5cm
cpsa_v4.log:     SR=90% (9/10)   CR=0%   avgMinD=38.1cm
```

### 三个场景全部汇总

```bash
for scene in corridor mixed obstacle; do
    echo "=== $scene ==="
    python extract_results.py $scene/
    echo
done
```

### 导出 JSON

```bash
python extract_results.py corridor/cpsa_v4.log
# 生成 corridor/cpsa_v4.json
```

## 六、方法说明

| method | 核心文件 | 安全边际 |
|--------|----------|----------|
| vanilla (默认) | neupan_node.py | 无 |
| cp_global | cpsa_core.py | 全局标量 q_asym=1.21cm |
| cpsa_v4 | cpsa_core.py | per-point 学习式 τ(p) |

## 七、评估指标

| 指标 | 含义 | 终端字段 |
|------|------|----------|
| SR (成功率) | arrived / 总轮数 | outcome=arrived |
| CR (碰撞率) | collision / 总轮数 | outcome=collision |
| avgMinD | 全程最近距离均值 | minD |
| Path | 实际行驶距离 | path |
| Time | 到达用时 | time |
