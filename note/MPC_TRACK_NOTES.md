# MPC Track Notes

这份说明整理 `scripts/mpc_track` 中轨迹跟踪代码的接入方式、脚本职责和资源路径配置。重点是说明当前代码如何接入 Mini2 资源，以及各脚本在轨迹跟踪流程中的作用。

## 1. 目录定位

当前轨迹跟踪代码目录为：

```text
scripts/mpc_track/
```

该目录面向的是“已有末端轨迹点 -> 逆运动学/动力学模型 -> MPC 跟踪 -> MuJoCo 可视化”的链路。它与 MPD 原项目中的扩散模型训练、推理不是同一层代码，而是用于对生成或给定轨迹做后续跟踪控制验证。

## 2. Mini2 资源路径

当前默认使用仓库内已有的 Mini2 资源：

```text
deps/torch_robotics/torch_robotics/data/urdf/robots/Mini2/
```

该目录包含：

```text
urdf/Mini2.urdf
xml/scene.xml
xml/Mini2.xml
meshes/
```

新增的 `asset_paths.py` 统一处理资源路径。默认情况下不再依赖外部路径：

```text
/home/yuancz/PycharmProjects/dpcc/Mini2/xml/scene.xml
```

如果后续要切换到另一份 Mini2 资源，可以设置：

```bash
export MINI2_ASSET_ROOT=/path/to/Mini2
```

该路径下必须包含 `urdf/`、`xml/` 和 `meshes/`。

## 3. 脚本职责

`cascaded_mpc.py` 是当前最完整的主脚本。它按 seed 读取 Mini2 inference 产生的最优无碰撞轨迹，并使用同一套 Mini2 inference 环境障碍。脚本用 Pinocchio 和 CasADi 构建动力学 MPC，跟踪末端位置和 inference 关节状态参考，并可选启用额外动态球实验障碍、高层重规划和 MuJoCo 播放。

当前主要通过 `cascaded_mpc.py` 顶部变量切换运行行为：

```python
inference_seed = 3
use_mujoco_viewer = True
use_dynamic_obstacle = True
record_video = True
```

其中：

- `inference_seed` 指定读取 `scripts/inference/logs/mini2/<seed>/results_data_dict.pickle`。
- `use_mujoco_viewer` 控制是否打开 MuJoCo viewer。
- `use_dynamic_obstacle` 控制是否启用额外动态球实验障碍和高层重规划。seed 环境障碍约束始终启用，不受该开关关闭。
- `record_video` 控制是否录制仿真视频，当前默认为 `True`。

`jaka_robot.py` 是 Mini2 的 Pinocchio 封装，提供：

- 正运动学 FK
- 阻尼最小二乘迭代 IK
- 逆动力学
- 质量矩阵
- 科氏/离心矩阵
- 重力项

`mujoco_sim.py` 是 MuJoCo 可视化封装，负责加载 `scene.xml`、播放关节轨迹、显示 seed 环境障碍、参考轨迹和实际末端轨迹。仿真开始前会按原有追加几何体方式一次性加入完整红色参考轨迹点，随后运动过程中逐步追加绿色实际末端轨迹点，红色和绿色轨迹点大小均为 `0.005`。启用 `use_dynamic_obstacle=True` 时，它还会显示并更新蓝色透明额外动态球；关闭该开关时，普通播放路径会隐藏 XML 中预置的动态球，不在仿真显示中使用它。

运行输出统一保存在 `scripts/mpc_track/logs/` 下。目录名只由 seed 和 MPC 类型决定，例如 `seed4_cascaded` 或 `seed4_single`。启用 `use_plot=True` 时，末端位置跟踪、关节力矩和关节角图片会保存为 PNG；启用 `record_video=True` 且打开 MuJoCo viewer 时，仿真视频会保存到同一目录。每次运行还会保存 `run_summary.yaml`，记录 seed、MPC 使用类型、动态障碍物参数、`MPC total`、平均每步耗时、求解失败次数、是否成功，以及约束违反量、障碍物最小距离、跟踪误差和求解耗时分布等数值指标。

`ik_error.py` 用于检查 IK 结果。它对参考轨迹逐点求 IK，再用 Pinocchio FK 和 MuJoCo FK 对比末端位置误差。

`get_workspace.py` 通过关节限位随机采样，统计并绘制 Mini2 的末端可达工作空间。

`forward_kinematic_DH.py` 是独立的改进 DH 正运动学对照脚本，依赖 Robotics Toolbox。

当前 `scripts/mpc_track/` 下不再保留单独的 `test_function.py` 测试脚本。需要检查 `jaka_robot.py` 的 FK/IK 时，优先使用 `ik_error.py`、`get_workspace.py` 或在对应模块中补充临时调试入口。

`asset_paths.py` 是资源路径工具脚本，统一返回 Mini2 的 URDF 和 MuJoCo XML 路径。

`inference_trajectory_loader.py` 按 seed 读取 `scripts/inference/logs/mini2/<seed>/results_data_dict.pickle`，选择 inference 中保存的最优无碰撞轨迹，并转换出 MPC 使用的 `q_ref`、`v_ref` 和末端位置 `p_ref`。同时它返回 Mini2 inference 默认环境障碍，当前为基础环境 10 个球加 `use_extra_objects=True` 对应的 4 个 extra spheres；这些障碍在当前 seed 结果之间保持一致。

`p_ref` 由 Pinocchio 根据 `q_ref` 逐点正运动学得到。读取 `data.oMf[frame_id].translation` 时必须复制坐标值，避免保存 Pinocchio 内部缓存视图导致整条参考轨迹全部变成最后一个点。

`dynamic_obstacle.py` 按 seed 读取 inference 最优轨迹，并根据主脚本传入的动态球速度、半径、命中时刻、碰撞关节和碰撞强度，反推出动态球初始三维位置。`cascaded_mpc.py` 启用 `use_dynamic_obstacle=True` 时会直接调用该脚本中的计算函数，不再需要手动复制 `dynamic_sphere_c0`。

`metrics_and_plots.py` 集中负责 MPC 后处理，包含关键指标计算、控制台指标摘要、`run_summary.yaml` 组装和三张 PNG 图保存。当前 `cascaded_mpc.py` 不再直接展开指标参数和 YAML 字段，只把当前运行上下文 `run_context` 交给该模块。关节角图中蓝线是 MPC 实际执行角度，红色虚线是 inference 原始关节角参考。

`trajectory_data/` 保存轨迹数据文件，当前主要由跟踪和验证脚本读取。

## 4. 修改说明

### 改动 1. Mini2 资源路径统一走当前仓库

原脚本中存在硬编码资源路径。现在统一改为通过 `asset_paths.py` 获取 `mini2_urdf_path()` 和 `mini2_scene_path()`。

这样 `jaka_robot.py`、`cascaded_mpc.py`、`mujoco_sim.py`、`get_workspace.py` 不再依赖外部 `dpcc` 项目路径，默认直接使用当前仓库内的 Mini2 资源。

### 改动 2. MPC 参考轨迹改为读取 inference 最优无碰撞结果

新增 `inference_trajectory_loader.py`，按 `cascaded_mpc.py` 中的 `inference_seed` 读取：

```text
scripts/inference/logs/mini2/<seed>/results_data_dict.pickle
```

当前会从 inference 结果中选择最优无碰撞轨迹，转换出 `q_ref`、`v_ref` 和 `p_ref`。`p_ref` 由 Pinocchio 正运动学逐点计算，读取 `data.oMf[frame_id].translation` 时使用 `.copy()`，避免保存 Pinocchio 内部缓存视图导致整条参考轨迹变成最后一个点。

`cascaded_mpc.py` 已不再使用原来的 `trajectory_data/interpolation.txt` 作为主参考来源，MPC 初始状态 `q0/v0` 也改为 inference 最优轨迹首点。

### 改动 3. seed 环境障碍统一用于 MPC 约束和 MuJoCo 可视化

`inference_trajectory_loader.py` 同步返回 Mini2 inference 环境球障碍，当前为基础环境 10 个球加 `use_extra_objects=True` 对应的 4 个 extra spheres。

这些障碍同时用于：

- 单层 MPC 的 seed 环境障碍约束
- 高层重规划 MPC 的 seed 环境障碍约束
- 底层 MPC 的 seed 环境障碍约束
- MuJoCo 中的静态障碍可视化

### 改动 4. 动态球恢复为可选实验障碍

`use_dynamic_obstacle=False` 时，`cascaded_mpc.py` 跳过高层重规划 MPC，直接使用单层状态跟踪 MPC 跟踪 inference 的 `q_ref/v_ref`，仍使用 seed 环境障碍约束，但不在 MPC 或 MuJoCo 中使用额外动态球。

`use_dynamic_obstacle=True` 时，才加入额外动态球并使用高层 MPC 做重规划；若 `use_cascaded_mpc=True`，再由底层 MPC 跟踪高层重规划轨迹。此时 MPC 约束包含 seed 环境障碍和额外动态球，MuJoCo 同时显示 seed 环境障碍、动态球、红色参考轨迹点和绿色实际轨迹点。

### 改动 5. MuJoCo 轨迹显示恢复为原脚本方式

`mujoco_sim.py` 中参考轨迹和实际轨迹的显示恢复为原脚本的核心方式：直接写入 `self.handle.user_scn`。

当前行为是：

- 仿真开始前一次性加入完整红色参考轨迹点
- 仿真运行时逐步追加绿色实际末端轨迹点
- 红色和绿色轨迹点半径均为 `0.005`
- 关闭动态障碍时隐藏 XML 中预置的动态球
- 开启动态障碍时将动态球显示为蓝色透明球，避免与红色参考轨迹混淆

### 改动 6. MPC 输出保存到固定 seed/type 目录，并使用 YAML 日志

运行输出保存在：

```text
scripts/mpc_track/logs/seed<seed>_<single|cascaded>/
```

例如：

```text
scripts/mpc_track/logs/seed4_cascaded/
```

如果 seed 和 MPC 使用类型不变，结果会保存到同一个目录中，后一次运行会覆盖同名图片、视频和 YAML 日志。

启用 `use_plot=True` 时保存：

```text
terminal_position_tracking.png
joint_torque.png
joint_angle_tracking.png
```

其中 `joint_angle_tracking.png` 当前同时包含：

- 蓝色实线：MPC 实际执行关节角
- 红色虚线：inference 原始关节角参考

启用 `record_video=True` 且打开 MuJoCo viewer 时保存：

```text
mpc_obstacle_avoidance.mp4
```

每次运行保存：

```text
run_summary.yaml
```

其中记录 seed、MPC 使用类型、是否启用额外动态球、动态球起始位置、动态球速度、动态球半径、命中归一化时刻、碰撞关节、碰撞强度、命中步、命中时间、`MPC total`、平均每步耗时、求解失败次数、是否成功和数值验证指标。图片和视频仍保存在同一 `logs` 目录中，但 `run_summary.yaml` 不再写入 `plot_enabled`、`plot_paths`、`video_enabled`、`video_path` 字段。

### 改动 7. 新增按 seed 生成动态球初始位置的脚本

文件：

```text
scripts/mpc_track/dynamic_obstacle.py
```

`cascaded_mpc.py` 中统一设置：

```python
dynamic_sphere_v = np.array([0.06, -0.07, 0.03])
dynamic_sphere_radius = 0.06
dynamic_obstacle_hit_fraction = 0.4
dynamic_obstacle_hit_frame = "5_joint"
dynamic_obstacle_collision_strength = 0.8
```

`dynamic_obstacle.py` 的命令行默认值仍可独立运行测试；在 MPC 主流程中，seed、轨迹总时长、动态球速度、动态球半径、安全裕度、命中时刻、碰撞关节和碰撞强度均由 `cascaded_mpc.py` 传入并保持一致。

在默认 `inference_seed=3`、`dynamic_obstacle_hit_fraction=0.4`、`dynamic_obstacle_hit_frame="5_joint"`、`dynamic_obstacle_collision_strength=0.8` 条件下，当前 MPC 会自动计算 `5_joint` 对应的动态球初始球心，不再手动写入固定结果。

```text
dynamic_sphere_c0 = compute_dynamic_sphere_position(...)[dynamic_obstacle_hit_frame]
```

输入 seed 后，独立脚本默认仍会分别对 `4_joint`、`5_joint`、`6_joint` 计算命中位置；主脚本运行时只使用 `dynamic_obstacle_hit_frame` 指定的碰撞关节。`dynamic_obstacle_collision_strength` 控制动态球在命中时刻与对应 joint 的接近程度，取值范围为 `[0, 1]`：

- `dynamic_obstacle_collision_strength=1.0`：球心穿过 joint，碰撞影响最强。
- `dynamic_obstacle_collision_strength=0.0`：球心只擦到 `dynamic_sphere_radius + safety_margin` 对应的安全包络边界。
- 中间值会沿轨迹切向的法向方向偏移，强度越大，偏移越小，动态障碍越容易影响 MPC。

计算关系为：

```text
effective_radius = dynamic_sphere_radius + safety_margin
offset_distance = (1 - dynamic_obstacle_collision_strength) * effective_radius
sphere_position_at_hit = target_position + offset_direction * offset_distance
dynamic_sphere_c0 = sphere_position_at_hit - dynamic_sphere_v * hit_time
```

常用命令：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/yuancz/miniconda3/envs/mpd/bin/python scripts/mpc_track/dynamic_obstacle.py --seed 4
```

指定碰撞强度：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/yuancz/miniconda3/envs/mpd/bin/python scripts/mpc_track/dynamic_obstacle.py --seed 4 --collision-strength 0.8
```

只输出三维位置：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/yuancz/miniconda3/envs/mpd/bin/python scripts/mpc_track/dynamic_obstacle.py --seed 4 --quiet
```

独立脚本使用 `seed=4`、`collision_strength=0.8` 时，示例输出为：

```text
4_joint: [0.06810256, 0.30241907, 0.37051373]
5_joint: [0.11686587, 0.33684822, 0.42765985]
6_joint: [0.12124783, 0.46486216, 0.51849427]
```

### 改动 8. 新增 MPC 数值验证指标

文件：

```text
scripts/mpc_track/cascaded_mpc.py
scripts/mpc_track/metrics_and_plots.py
```

当前指标和绘图逻辑已从 `cascaded_mpc.py` 拆到 `metrics_and_plots.py`。  
`cascaded_mpc.py` 只负责固化当前运行上下文并调用：

```python
run_context = dict(locals())
metrics_data = build_mpc_run_metrics_from_context(run_context)
print_mpc_metrics_summary(metrics_data)
plot_mpc_tracking_results_from_context(run_context, metrics_data, show=True)
run_summary = build_run_summary(run_context, metrics_data)
```

这样主脚本可以继续聚焦在 MPC 建模、求解和 MuJoCo 播放，不再堆叠长参数列表、指标统计细节和 YAML 字段拼装。

当前 `run_summary.yaml` 主要保留：

```text
verification_passed
key_metrics
```

`key_metrics` 只保留十个左右关键指标，中文含义如下：

- `ee_position_rmse_m`：末端位置跟踪均方根误差，反映整体跟踪精度。
- `ee_position_max_error_m`：末端位置最大误差，反映最差时刻偏差。
- `joint_position_rmse_rad`：关节角跟踪均方根误差，反映整体关节跟踪精度。
- `seed_obstacle_min_surface_clearance_m`：机器人与 seed 静态球障碍表面的最小净间距，越大越安全。
- `dynamic_obstacle_min_surface_clearance_m`：机器人与额外动态球表面的最小净间距；未启用动态障碍时为 `None`。
- `max_q_limit_violation_rad`：关节位置约束最大越界量，正常应为 `0`。
- `max_v_limit_violation_rad_s`：关节速度约束最大越界量，正常应为 `0`。
- `max_tau_limit_violation_nm`：关节力矩约束最大越界量，正常应为 `0`。
- `solver_mean_time_s`：全部求解器调用的平均耗时。
- `solver_p95_time_s`：全部求解器调用耗时的 95 分位值，用来看大多数步骤下的求解时延上界。

`success` 的含义仍然保持原来的弱口径：

```text
solver_failure_count == 0 and states_recorded >= steps_expected
```

新增的 `verification_passed` 是更严格的后验检查，当前要求：

```text
success == true
所有状态/控制约束检查通过
所有障碍物低安全裕度检查通过
```

这意味着 `success: true` 只能说明求解过程没有失败并且状态数量完整；是否真的满足约束和安全距离，应优先看 `verification_passed` 和上面的关键指标。旧版按四大类展开的详细指标说明已经移除，避免和当前精简后的 `key_metrics` 重复。

## 5. 未完成验证

- 基础数值指标已经写入 `run_summary.yaml`，但还没有形成批量对比表。是否保留级联重规划路径仍不能下结论，需要在相同 seed、相同 inference 参考、相同 seed 环境障碍和相同动态球设置下，对比单层 MPC 与级联 MPC 的跟踪误差、安全裕度、失败次数和耗时。
- 当前指标是单次运行的后验统计，还没有加入统一阈值筛选、跨 seed 汇总和显著性分析。后续如果要支撑“级联更优”或“动态避障有效”，应增加批量 runner，把多个 `run_summary.yaml` 汇总成表格。

当前 `cascaded_mpc.py` 中各层 MPC 实际跟踪参量如下：

- 上层 MPC：主要跟踪末端位置参考 `p_ref`，对应代价项为 `pk - P_ref_param[:, k]`；同时辅助跟踪 inference 关节状态 `x_ref=[q_ref, v_ref]`，对应代价项为 `xk - X_ref_high_param[:, k]`。
- 下层 MPC：跟踪上层 MPC 输出的状态轨迹 `X_guess_prev`，经样条插值后作为 `X_ref_param`；实际代价项为 `xk - X_ref_param[:, k]`，也就是跟踪关节状态 `x=[q, v]`。
- 无动态障碍的单层 tracker：直接跟踪 inference 关节状态 `x_ref=[q_ref, v_ref]`，对应代价项为 `xk - X_ref_track_param[:, k]`。

因此，当前只有上层 MPC 直接把末端位置 `p_ref` 放进目标函数；下层 MPC 和无动态障碍单层 tracker 都是状态跟踪。`p_ref` 只有末端三维位置，没有末端姿态，所以应称为“末端位置跟踪”，不能称为完整“末端位姿跟踪”。

对照论文中 MPC 重规划和 MPC 跟踪的常见代价函数，这个设计总体是合理的，但还没有验证充分。经典 MPC 一般在有限预测域内最小化预测输出或状态与参考轨迹的误差，同时惩罚控制量、控制增量或控制平滑项，并显式处理状态/输入约束。机械臂任务空间 NMPC 中，也常把末端位置或末端位姿误差作为主要任务代价，同时加入关节状态、控制量、限位、奇异性和碰撞约束；冗余机械臂还需要关节正则或参考姿态来避免仅跟踪 3D 末端位置时出现多解漂移。

因此，当前高层 MPC 用末端位置误差做动态障碍重规划主目标，并保留 inference 关节状态跟踪和 `q_nom_param` 关节正则，这与任务空间重规划的一般做法一致；底层 MPC 跟踪高层输出的 `q/v` 状态轨迹，也符合“高层规划/重规划，底层状态跟踪”的级联控制思路。

不充分的地方主要在对比和指标。单层 inference tracker 只在关节状态空间跟踪 `q_ref/v_ref`，而高层动态障碍 MPC 同时跟踪末端位置和关节状态；二者目标函数并不完全相同。如果要比较单层和级联方案，不能只比较 `success` 或 MP4，而应同时报告关节状态误差、末端位置误差、障碍最小距离、控制/加速度/速度约束余量和求解耗时。否则无法判断差异来自控制结构本身，还是来自代价函数跟踪对象不同。
