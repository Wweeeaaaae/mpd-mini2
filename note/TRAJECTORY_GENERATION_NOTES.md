# Mini2 Trajectory Generation Notes

本文不再按改动发生的时间顺序记录，而是按“已经落地的改动项”整理 Mini2 轨迹生成接入。重点是把当前仓库里真实保留的修改、已经撤回的临时改动、验证结果和后续注意事项分开写清楚。

## 改动范围

当前与 Mini2 轨迹生成直接相关的文件包括：

```text
deps/torch_robotics/torch_robotics/torch_kinematics_tree/models/robots.py
deps/torch_robotics/torch_robotics/data/configs/mini2/mini2_sphere_config.yaml
deps/torch_robotics/torch_robotics/robots/robot_mini2.py
deps/torch_robotics/torch_robotics/robots/__init__.py
deps/torch_robotics/torch_robotics/environments/env_spheres_3d_mini2.py
deps/torch_robotics/torch_robotics/environments/__init__.py
deps/torch_robotics/torch_robotics/robots/robot_panda.py
deps/torch_robotics/torch_robotics/robots/robot_point_mass.py
scripts/generate_data/launch_generate_trajectories_mini2.py
scripts/generate_data/launch_generate_trajectories.py
```

已经恢复原样、不再作为最终 Mini2 方案保留的临时改动包括：

```text
deps/torch_robotics/torch_robotics/environments/env_spheres_3d.py
deps/torch_robotics/torch_robotics/torch_kinematics_tree/geometrics/utils.py
deps/torch_robotics/torch_robotics/torch_kinematics_tree/models/rigid_body.py
```

## 改动 1. Mini2 资产已经确认可接入现有规划链路

Mini2 资产目录：

```text
deps/torch_robotics/torch_robotics/data/urdf/robots/Mini2/
```

已确认的基础条件：

- 目录结构是标准 ROS 风格，包含 `urdf/`、`meshes/`、`launch/`、`config/`、`package.xml`。
- URDF 机器人名为 `Mini2`。
- 关节为 `1_joint` 到 `6_joint`，均为 `revolute`，符合 6-DoF 机械臂。
- link 为 `base_link`、`1_Link` 到 `6_Link`。
- 每个关节都有 `limit`，可以直接被 `DifferentiableTree.get_joint_limit_array()` 读取。
- mesh 路径使用 `package://Mini2/meshes/...`，与当前目录结构匹配。

读取到的关节上下限为：

```text
q_min = [-6.2832, -2.0944, -2.1816, -6.2832, -2.0944, -6.2832]
q_max = [ 6.2832,  2.0944,  2.1816,  6.2832,  2.0944,  6.2832]
```

当前仍会出现的 warning：

```text
Unknown tag "mujoco" in /robot[@name='Mini2']
```

这来自 URDF 中的 `<mujoco>` 标签。当前解析器会忽略它，不影响：

- 关节限位读取
- 正运动学
- 轨迹生成
- 碰撞检查 smoke test

## 改动 2. 新增 `DifferentiableMini2`，把 Mini2 接入 URDF 解析和正运动学入口

文件：

```text
deps/torch_robotics/torch_robotics/torch_kinematics_tree/models/robots.py
```

新增类：

```python
class DifferentiableMini2(DifferentiableTree):
```

这一项改动解决的是“Mini2 URDF 如何进入现有 `DifferentiableTree` 链路”的问题。当前实现负责：

- 指向 `Mini2.urdf`
- 提供所有 link 的正运动学
- 提供关节限位读取能力
- 在需要时切换到 collision-model URDF

当前构造函数支持：

```python
use_collision_spheres=False
```

当 `use_collision_spheres=True` 时，会额外调用：

```python
modidy_mini2_urdf_collision_model(robot_file)
```

该逻辑会：

- 读取 `mini2_sphere_config.yaml`
- 把球体写成挂在原 link 下的 fixed child link
- 生成 `Mini2_collision_model.urdf`
- 返回 `link_collision_names` 和 `link_collision_margins`

这一步是后续 `RobotMini2` 默认碰撞模型的基础。

## 改动 3. 新增 `RobotMini2`，补齐规划任务所需的机器人抽象接口

文件：

```text
deps/torch_robotics/torch_robotics/robots/robot_mini2.py
```

新增类：

```python
class RobotMini2(RobotBase):
```

这项改动的目标不是“能读 URDF”这么简单，而是让 Mini2 能像 Panda 一样被 `PlanningTask` 直接使用。当前 `RobotMini2` 已补齐：

- `q_limits`
- `q_dim`
- `fk_map_collision_impl(q)`
- `link_names_for_object_collision_checking`
- `link_margins_for_object_collision_checking`
- `link_idxs_for_object_collision_checking`
- 自碰撞检查配置
- 渲染接口

当前关键配置为：

```python
self.link_name_ee = '6_Link'
self.diff_robot = DifferentiableMini2(
    device=tensor_args['device'],
    use_collision_spheres=use_collision_spheres,
)
```

关节限位直接从 URDF 读取：

```python
self.jl_lower, self.jl_upper, _, _ = self.diff_robot.get_joint_limit_array()
q_limits = torch.tensor(np.array([self.jl_lower, self.jl_upper]), **tensor_args)
```

碰撞位置仍然通过正运动学映射输出：

```python
def fk_map_collision_impl(self, q, **kwargs):
    ...
    link_pose_dict = self.diff_robot.compute_forward_kinematics_all_links(q, return_dict=True)
    link_tensor = convert_link_dict_to_tensor(link_pose_dict, self.diff_robot.get_link_names())
    link_pos = link_pos_from_link_tensor(link_tensor)
    ...
    return link_pos
```

这意味着 Mini2 现在已经进入了与 Panda 同类的机器人抽象接口，而不是停留在单独读 URDF 的阶段。

## 改动 4. Mini2 默认碰撞模型改成 Panda 风格 collision sphere，而不是早期粗略 fallback

相关文件：

```text
deps/torch_robotics/torch_robotics/data/configs/mini2/mini2_sphere_config.yaml
deps/torch_robotics/torch_robotics/torch_kinematics_tree/models/robots.py
deps/torch_robotics/torch_robotics/robots/robot_mini2.py
```

当前默认行为：

```python
RobotMini2(use_collision_spheres=True)
```

这项改动的核心是把 Mini2 从“原始 link 点 + 线性插值 + 每个 link 一个粗半径”的简化模型，切换到与 Panda 默认方案一致的 collision sphere 模型。

### 4.1 新增 `mini2_sphere_config.yaml`

文件：

```text
deps/torch_robotics/torch_robotics/data/configs/mini2/mini2_sphere_config.yaml
```

格式与 Panda 保持一致：

```yaml
2_Link:
- [-0.020, 0.0, 0.010, 0.065]
- [0.040, 0.0, 0.010, 0.065]
- [0.100, 0.0, 0.010, 0.065]
```

每一项表示：

```text
[link-local x, link-local y, link-local z, sphere radius]
```

这些球心和半径是根据 Mini2 STL mesh 的 bounds 和 extent 做的工程初始化。用于估计的 mesh 尺寸记录如下：

```text
base_link.STL extents [0.123849, 0.123945, 0.031000]
1_Link.STL    extents [0.098730, 0.114005, 0.208000]
2_Link.STL    extents [0.300501, 0.098037, 0.125001]
3_Link.STL    extents [0.082987, 0.171802, 0.097502]
4_Link.STL    extents [0.075636, 0.092501, 0.120708]
5_Link.STL    extents [0.074998, 0.160800, 0.092502]
6_Link.STL    extents [0.076489, 0.075000, 0.039000]
```

当前结论：

- 这不是 mesh 精确碰撞
- 但它比早期 fallback 更接近 Panda 默认方案
- 如果碰撞判断偏严或偏松，优先调这个 YAML

### 4.2 `DifferentiableMini2` 支持自动生成 collision-model URDF

新增函数：

```python
def modidy_mini2_urdf_collision_model(robot_file):
```

当前作用是：

- 读取 `mini2_sphere_config.yaml`
- 为每个球生成 fixed joint 和空 child link
- 写出 `Mini2_collision_model.urdf`
- 把 sphere link 名称和半径返回给 `RobotMini2`

### 4.3 `RobotMini2` 默认走 sphere-link 碰撞检查

当 `use_collision_spheres=True` 时：

```python
link_names_for_object_collision_checking = self.diff_robot.link_collision_names
link_margins_for_object_collision_checking = torch.tensor(
    self.diff_robot.link_collision_margins,
    **tensor_args,
).view(-1, 1)
```

这会让 `CollisionObjectDistanceField` 最终走到：

```python
interpolate_link_pos = not self.robot.use_collision_spheres
```

也就是说，默认不再对原始 link 点做线性插值，而是直接检查 sphere center 的位置。

### 4.4 早期 fallback 模型仍然保留

显式传：

```python
RobotMini2(use_collision_spheres=False)
```

时，会回退到原始 link 列表：

```python
[
    'base_link',
    '1_Link',
    '2_Link',
    '3_Link',
    '4_Link',
    '5_Link',
    '6_Link',
]
```

对应的粗略半径为：

```python
[
    0.07,
    0.075,
    0.09,
    0.065,
    0.06,
    0.06,
    0.045,
]
```

这套 fallback 只用于排查问题，不是当前默认方案。

### 4.5 自碰撞目前仍使用 link pair 方法

当前保留的自碰撞检查对为：

```python
OrderedDict({
    '4_Link': ['base_link'],
    '5_Link': ['base_link', '1_Link'],
    '6_Link': ['base_link', '1_Link', '2_Link'],
})
```

保留原因：

- 相邻 link 天然连接，不适合直接当作自碰撞
- 当前主要关心远端 link 回折到前段结构的情况
- 这是一组偏保守的初值，后续可继续调

当前没有接入 Panda 的 STORM 风格自碰撞方案，因为其依赖：

```text
robot_self/franka_self_sdf.pt
```

这套权重是 Franka 专用，不能直接给 Mini2 复用。

## 改动 5. 完成 Mini2 机器人注册，并修复由此暴露出的循环导入

相关文件：

```text
deps/torch_robotics/torch_robotics/robots/__init__.py
deps/torch_robotics/torch_robotics/robots/robot_mini2.py
deps/torch_robotics/torch_robotics/robots/robot_panda.py
deps/torch_robotics/torch_robotics/robots/robot_point_mass.py
```

### 5.1 注册 `RobotMini2`

在：

```text
deps/torch_robotics/torch_robotics/robots/__init__.py
```

中新增：

```python
from .robot_mini2 import *
```

必要性在于：

```python
robot_class = getattr(robots, robot_id)
```

如果不导出，`robot_id='RobotMini2'` 无法工作。

当前验证结果：

```python
import torch_robotics.robots as robots
'RobotMini2' in dir(robots)  # True
```

### 5.2 修复循环导入

在接入 `RobotMini2` 后，真实导入路径暴露出循环导入问题：

```text
ImportError: cannot import name 'RobotPointMass' from partially initialized module 'torch_robotics.robots'
```

原因是 `robot_mini2.py` 顶层导入了 `torch_robotics.environments.primitives`，而 `environments` 包又反向依赖 `robots` 包。

当前修复方式是把仅用于渲染的导入移动到函数内部：

```python
if draw_links_spheres:
    from torch_robotics.environments.primitives import MultiSphereField
```

以及：

```python
from torch_robotics.environments.primitives import plot_sphere
```

也改成了真正绘制时再导入。

结果：

- `torch_robotics.robots` 可以正常导入
- `RobotMini2` 可以正常注册和实例化
- Panda 和 PointMass 原有渲染行为不变

## 改动 6. Mini2 不再复用 `EnvSpheres3D`，而是使用专属环境 `EnvSpheres3DMini2`

相关文件：

```text
deps/torch_robotics/torch_robotics/environments/env_spheres_3d_mini2.py
deps/torch_robotics/torch_robotics/environments/__init__.py
scripts/generate_data/launch_generate_trajectories_mini2.py
```

这项改动的原因很直接：Mini2 的工作范围约为 `0.6 m`，明显小于 Panda 在 `EnvSpheres3D` 中对应的工作尺度。继续复用 Panda 环境会造成：

- 一部分障碍物超出 Mini2 主要可达区
- 一部分障碍物相对 Mini2 工作空间过大
- 起终点采样和规划难度不匹配

因此当前正式方案是：

```text
EnvSpheres3D       -> Panda 原始用途
EnvSpheres3DMini2  -> Mini2 专属轨迹生成环境
```

### 6.1 新增 `EnvSpheres3DMini2`

文件：

```text
deps/torch_robotics/torch_robotics/environments/env_spheres_3d_mini2.py
```

新增类：

```python
class EnvSpheres3DMini2(EnvBase):
```

并在：

```text
deps/torch_robotics/torch_robotics/environments/__init__.py
```

中导出：

```python
from .env_spheres_3d_mini2 import *
```

### 6.2 Mini2 专属障碍物配置

当前障碍物球心：

```python
torch.tensor([
    [-0.25,  0.25, 0.52],
    [-0.30, -0.24, 0.34],
    [-0.34,  0.08, 0.16],
    [ 0.34,  0.24, 0.16],
    [ 0.38,  0.18, 0.43],
    [ 0.42, -0.28, 0.24],
    [ 0.18, -0.32, 0.43],
    [ 0.28,  0.05, 0.58],
    [ 0.05, -0.30, 0.16],
    [ 0.05,  0.34, 0.30],
])
```

当前障碍物半径：

```python
torch.tensor([
    0.10,
    0.10,
    0.10,
    0.10,
    0.10,
    0.10,
    0.09,
    0.09,
    0.09,
    0.09,
])
```

代码中保留的可调范围：

```text
centers x/y: roughly [-0.45, 0.45]
centers z:   roughly [0.05, 0.60]
obstacle radii: [0.07, 0.12]
```

### 6.3 Mini2 专属工作空间边界

当前工作空间：

```python
limits=torch.tensor([[-0.75, -0.75, -0.15], [0.75, 0.75, 0.85]], **tensor_args)
```

代码中保留的调参范围：

```text
x/y limits: +/-[0.65, 0.90]
z limits:   [-0.20, 0.90]
```

### 6.4 Mini2 专属 GPMP2 参数

当前参数：

```python
opt_iters=300
num_samples=64
sigma_gp=5e-2
sigma_coll=8e-5
step_size=7.5e-1
stop_criteria=0.05
```

代码中保留的调参范围：

```text
opt_iters:     [250, 500]
num_samples:   [32, 128]
sigma_gp:      [3e-2, 1e-1]
sigma_coll:    [5e-5, 2e-4]
step_size:     [3e-1, 1.0]
stop_criteria: [0.03, 0.10]
```

### 6.5 Mini2 专属 RRTConnect 参数

当前参数：

```python
n_iters=10000
step_size=torch.pi / 36
n_radius=torch.pi / 5
n_pre_samples=30000
max_time=180
```

代码中保留的调参范围：

```text
n_iters:       [10000, 30000]
step_size:     [pi/45, pi/24]
n_radius:      [pi/6, pi/4]
n_pre_samples: [20000, 50000]
max_time:      [120, 300]
```

## 改动 7. 新增 Mini2 专用 launcher，并把 Panda 通用 launcher 整理成可读模板

相关文件：

```text
scripts/generate_data/launch_generate_trajectories_mini2.py
scripts/generate_data/launch_generate_trajectories.py
```

### 7.1 `launch_generate_trajectories_mini2.py` 已稳定为正式入口

当前核心配置：

```python
env_id = 'EnvSpheres3DMini2'
robot_id = 'RobotMini2'
num_contexts = 500
num_trajectories_per_context = 20
threshold_start_goal_pos = float(6 * np.deg2rad(15))
obstacle_cutoff_margin = 0.05
```

当前还固定了：

```python
exp_name = 'generate_trajectories_mini2'
USE_CUDA = False
N_EXPS_IN_PARALLEL = 15
N_CORES = 8
CONDA_ENV = 'mpd'
```

这项改动当前体现为：

- 正式入口已经固定走 `EnvSpheres3DMini2 + RobotMini2`
- `threshold_start_goal_pos` 已转成普通 Python `float`
- 默认配置已经恢复到 `500 x 20` 批量模式
- 默认按 CPU 并发运行
- 输出目录固定为 `generate_trajectories_mini2_<timestamp>`

`threshold_start_goal_pos` 当前含义仍然是关节空间阈值，而不是末端笛卡尔距离：

- Panda 经验值约为 `1.83`
- Mini2 当前设为 `float(6 * np.deg2rad(15)) ~= 1.57`

### 7.2 `launch_generate_trajectories.py` 已整理为通用配置模板

这一轮没有改变 Panda 的生成逻辑，但把原来的 launcher 改成了可直接阅读、可直接修改的配置模板。当前默认配置是：

```python
env_id = 'EnvSpheres3D'
robot_id = 'RobotPanda'
num_contexts = 500
num_trajectories_per_context = 20
threshold_start_goal_pos = 1.83
obstacle_cutoff_margin = 0.05
```

并补充了：

- 参数含义说明
- CPU / GPU、并发数、内存、分区说明
- `launcher.add_experiment(...)` 各字段的注释

这让 Panda 与 Mini2 的 launcher 现在可以直接做对照。

## 改动 8. 早期临时改动已经清理，当前方案只保留正式入口

这部分内容需要单独写清楚，因为仓库里确实出现过早期验证用的临时方案。

### 8.1 `EnvSpheres3D` 中给 Mini2 加参数分支的临时改动已撤回

早期为了快速验证，曾临时让原始：

```text
deps/torch_robotics/torch_robotics/environments/env_spheres_3d.py
```

支持 `RobotMini2` 的 `get_rrt_connect_params()` 和 `get_gpmp2_params()` 分支。

当前这些分支已经移除，`EnvSpheres3D` 恢复为 Panda 用途，不再作为 Mini2 正式方案的一部分。

### 8.2 两个无关注释改动已恢复

下面两个文件曾有临时注释类修改，但与 Mini2 接入逻辑无关，当前已经恢复：

```text
deps/torch_robotics/torch_robotics/torch_kinematics_tree/geometrics/utils.py
deps/torch_robotics/torch_robotics/torch_kinematics_tree/models/rigid_body.py
```

## 改动 9. 现有验证结果已经覆盖“注册、环境、采样、生成、重新加载”几个关键链路

### 9.1 静态检查

对以下文件做过 AST 检查：

```text
deps/torch_robotics/torch_robotics/robots/robot_mini2.py
deps/torch_robotics/torch_robotics/robots/robot_panda.py
deps/torch_robotics/torch_robotics/robots/robot_point_mass.py
deps/torch_robotics/torch_robotics/environments/env_spheres_3d_mini2.py
deps/torch_robotics/torch_robotics/environments/__init__.py
scripts/generate_data/launch_generate_trajectories_mini2.py
```

结果：

```text
AST OK
```

### 9.2 `RobotMini2` 注册和关节限位检查

使用环境：

```text
/home/yuancz/miniconda3/envs/mpd/bin/python
```

输出：

```text
Unknown tag "mujoco" in /robot[@name='Mini2']
registered True
RobotMini2 6
[-6.2832, -2.0944, -2.1816, -6.2832, -2.0944, -6.2832]
[6.2832, 2.0944, 2.1816, 6.2832, 2.0944, 6.2832]
```

当前可以确认：

- `RobotMini2` 已完成注册
- `q_dim = 6`
- 关节限位读取正确

### 9.3 `EnvSpheres3DMini2` 构造与采样检查

输出：

```text
Unknown tag "mujoco" in /robot[@name='Mini2']
exported True
Precomputing the SDF grid and gradients took: 1.138 sec
EnvSpheres3DMini2 [[-0.75, -0.75, -0.15000000596046448], [0.75, 0.75, 0.8500000238418579]]
{'n_iters': 10000, 'step_size': 0.08726646259971647, 'n_radius': 0.6283185307179586, 'n_pre_samples': 30000, 'max_time': 180}
300 0.05
torch.Size([2, 6]) 7.0271
```

当前可以确认：

- 新环境已导出
- 新环境边界生效
- Mini2 专属 RRT / GPMP2 参数生效
- 可采样到无碰撞 6 维配置

### 9.4 历史 smoke test 已证明整条规划链路能跑通

早期在 `EnvSpheres3D + RobotMini2` 上做过历史 smoke test。虽然这条环境路径已经不再是正式方案，但它仍然说明当时“Mini2 机器人包装 + 规划器 + 结果保存”链路是打通的。

历史运行输出：

```text
Iteration:    10/10000 | Time: 0.092 s| Nodes: 24 | Success: True
Sample-based Planner -- Optimization time: 0.095 sec
Optimization-based Planner -- Optimization time: 18.119 sec
Hybrid-based Planner -- Optimization time: 18.215 sec
percentage free trajs: 100.00
percentage collision intensity 0.00
success 1
```

历史结果文件包括：

- `metadata.yaml`
- `args.yaml`
- `trajs-free.pt`
- `trajs-collision.pt`
- `results_data_dict.pickle`
- `trajectories.png`
- `logfile`

重新加载后得到：

```text
fraction_free 1.0
collision_intensity 0.0
success 1
```

当前应如何理解这条历史结果：

- 它证明 Mini2 接入规划和保存链路曾经跑通
- 但它使用的是旧环境路径
- 当前正式生成应以 `EnvSpheres3DMini2-RobotMini2` 目录结果为准

## 改动 10. 当前已知限制与调参建议

### 10.1 `<mujoco>` warning 仍然存在，但不阻断流程

如果后续要清理 warning，可以单独准备一份不带 `<mujoco>` 标签的规划专用 URDF，再让 `DifferentiableMini2` 指向该文件。

### 10.2 collision sphere 仍然是工程近似，不是 mesh 精确碰撞

如果后续出现：

- 本应可行的姿态被误判碰撞
- 轨迹明显贴障碍甚至穿模
- RRT 很难找到路

建议优先排查顺序：

1. `mini2_sphere_config.yaml` 中球体位置和半径
2. `obstacle_cutoff_margin`
3. `threshold_start_goal_pos`
4. `EnvSpheres3DMini2` 中障碍物尺寸
5. RRT / GPMP2 参数

## 改动 11. 2026-04-24 Mini2 批量生成结果统计、训练影响与补齐方案

统计目录：

```text
scripts/generate_data/logs/generate_trajectories_mini2_2026-04-23_16-55-24/
  env_id___EnvSpheres3DMini2/
  robot_id___RobotMini2/
```

这里的 `invalid` 按当前最终判定口径统计，既包括碰撞，也包括最后的 joint-limit 过滤。

### 11.1 Mini2 当前统计

当前这一批共：

- `500` 个 context
- 目标轨迹数 `10000`
- `9341` 条 free trajectory
- `659` 条 invalid trajectory
- `461` 个 context 为 `20/20` 全 free
- `11` 个 context 为部分 invalid
- `28` 个 context 为 `20/20` 全 invalid

`20/20` 全 invalid 的 seed：

```text
10, 20, 28, 53, 56, 86, 105, 117, 120, 127, 135, 197,
215, 217, 283, 289, 295, 303, 325, 330, 345, 387, 398,
418, 420, 433, 446, 447
```

部分 invalid 的 seed：

```text
2, 97, 110, 181, 199, 264, 273, 286, 298, 392, 439
```

当前缺失的 free trajectory 总数是 `659`，其中：

- `28` 个全 invalid context 对应损失 `560`
- `11` 个部分 invalid context 额外损失 `99`

### 11.2 与官方数据对照

官方 Panda 3D 数据目录：

```text
data_trajectories/EnvSpheres3D-RobotPanda/
```

统计结果：

- `9640` 条 free trajectory
- `360` 条 invalid trajectory
- `7` 个 `20/20` 全 invalid context

这些全 invalid seed 为：

```text
64, 78, 179, 285, 306, 349, 415
```

当前结论：

- `20/20` 全 invalid context 不是 Mini2 独有现象
- 官方 Panda 3D 数据里也存在同类情况
- 但当前 Mini2 的缺口比官方 Panda 更大

### 11.3 对训练的影响

当前判断：

- 会影响数据规模和困难 context 覆盖
- 不会直接阻断训练

原因：

- 训练集只读取 `trajs-free.pt`
- invalid trajectory 不会进入训练
- 部分 invalid 的 context 仍会贡献剩余 free trajectory
- 全 invalid 的 context 等价于不贡献样本

当前 Mini2 仍然保留了 `9341` 条 free trajectory，因此现在的问题更准确地说是“有效样本数少于预期”，而不是“数据无法训练”。

补充一点：当前 `free=0` 的目录里保存的是 `torch.empty(0)`。当前运行时通常还能继续加载训练集，但正式整理数据时仍建议直接剔除这些目录。

### 11.4 回填方案（已执行）

在不修改 GPMP2 cost 的前提下，本轮实际采用的是“直接补采样并回填原 `free=0` 目录”的方案。

本轮 16 个回填 seed 固定为：
- `10 <- 500`
- `20 <- 501`
- `28 <- 502`
- `53 <- 503`
- `56 <- 504`
- `86 <- 505`
- `105 <- 506`
- `117 <- 507`
- `120 <- 508`
- `127 <- 509`
- `135 <- 510`
- `197 <- 511`
- `215 <- 512`
- `217 <- 513`
- `283 <- 514`
- `289 <- 515`

执行约束：
- 结果写回原 `free=0` 文件夹
- `args.yaml` 中的 `seed` 保持旧 seed
- 新 seed 只用于重新采样和规划随机性

### 11.4.1 运行结果（2026-04-25）

本轮 16 个回填目录已全部从 `free=0` 变为 `free>0`，并且 `args.yaml` 中的 `seed` 已核对为旧目录 seed。

整批统计更新为：
- free trajectory：`9341 -> 9625`，净增 `284`
- invalid trajectory：`659 -> 375`，减少 `284`
- `20/20` 全 invalid context：`28 -> 12`

这 16 个目录的回填结果为：
- `20/20 free`：`20, 53, 56, 86, 105, 117, 120, 127, 197, 215, 217, 283, 289`
- `19/20 free`：`28`
- `4/20 free`：`135`
- `1/20 free`：`10`

当前仍为 `free=0` 的 seed：
- `295, 303, 325, 330, 345, 387, 398, 418, 420, 433, 446, 447`

结论：
- 这轮回填有效，已经把有效 free 数补到 `9625`
- 与官方 Panda 3D 的 `9640` 相比，仅差 `15` 条 free trajectory

## 改动 12. 当前结论

基于 2026-04-24 的排查，当前可以把 Mini2 数据生成状态总结为：

1. Mini2 规划链路已经打通，绝大多数 context 可以生成 free trajectory。
2. 当前批量结果不是不可用，而是有效样本数低于名义目标。
3. `20/20` 全 invalid context 确实存在，但官方 Panda 3D 数据里也有同类现象。
4. 对训练的主要影响是困难 context 覆盖减少，而不是 invalid 轨迹进入训练。
5. 当前已验证“补采样并回填原异常目录”可以显著提升有效样本数。
