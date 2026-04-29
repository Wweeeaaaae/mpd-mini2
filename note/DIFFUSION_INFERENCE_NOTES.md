# Inference Notes

本文只保留当前仍有效的改动和已验证结论。已经证伪、已经被后续实验覆盖、或者只用于中间排查的猜测不再展开。

## 改动范围

涉及文件：

```text
scripts/inference/inference.py
scripts/inference/inference_mini2.py
deps/torch_robotics/torch_robotics/environments/env_spheres_3d_mini2_extra_objects.py
deps/torch_robotics/torch_robotics/environments/__init__.py
deps/torch_robotics/torch_robotics/isaac_gym_envs/motion_planning_envs.py
deps/torch_robotics/torch_robotics/trajectory/utils.py
```

## 当前稳定改动

### 1. 修正 inference 参数名和结果字段

文件：

```text
scripts/inference/inference.py
scripts/inference/inference_mini2.py
```

已修正两类问题：

1. 碰撞引导的插值点参数名  
   原始代码传入的是：

   ```python
   num_interpolated_points=...
   ```

   当前统一改成：

   ```python
   num_interpolated_points_for_collision=...
   ```

   原因是 `GuideManagerTrajectoriesWithVelocity` 实际读取的是后者。旧参数名会被 `**kwargs` 吞掉，因此代码能跑，但自定义插值点数不会生效。

2. 路径长度和平滑度字段写反  
   原始写法把：

   ```python
   'cost_path_length_trajs_final_free'
   'cost_smoothness_trajs_final_free'
   ```

   对应到了错误的变量。当前已经改成字段名与保存值一致，避免后续读取 `results_data_dict.pickle` 时把两类代价解释反。

### 2. 调整 Panda inference 结果目录与视频输出路径

文件：

```text
scripts/inference/inference.py
```

这次 Panda 脚本的主要问题不是“视频保存失败”，而是脚本内部会覆盖用户传入的 `results_dir`。

原始行为：

- 参数表面上提供了 `results_dir`
- 但进入 `experiment(...)` 后，代码又强行写成：

  ```python
  model_dir/results_inference/<seed>
  ```

- 因此视频和 `results_data_dict.pickle` 实际不会保存到 `scripts/inference/logs/...`

当前已改成：

- 默认结果目录为：

  ```text
  scripts/inference/logs/panda/<seed>
  ```

- `results_dir` 不再被脚本内部重写
- `results_data_dict.pickle`
- `*-joint-space-opt-iters.mp4`
- `*-isaac-controller-position.mp4`
- `*-traj-opt-iters.mp4`
- `*-robot-traj.mp4`

  都统一写到同一个 `results_dir` 下

另外，为了避免工作目录变化引起的相对路径问题，Panda 的 Isaac asset 路径也改成了基于仓库根目录的绝对路径，不再依赖 `../../deps/...` 这样的相对写法。

当前 Panda 入口另外补了一个可选参数：

```python
model_dir: str = ''
checkpoint_path: str = ''
```

这样在不修改源码的前提下，也可以显式指定 Panda 模型目录和 checkpoint。

这一轮同时还修掉了一个与目录整理直接相关的加载错误：

- 前面已经把 Panda 改成使用 `model_dir_path`
- 但加载 checkpoint 的地方仍残留旧变量 `model_dir`
- 在未显式传 `model_dir` 时，这会退化成查找相对路径 `checkpoints/...`
- 运行时表现为：

  ```text
  FileNotFoundError: checkpoints/ema_model_current_state_dict.pth
  ```

当前已经统一改成：

- 如果传了 `checkpoint_path`，就直接使用显式路径
- 否则从：

  ```text
  <model_dir_path>/checkpoints/
  ```

  下自动选择 `ema_model_current_state_dict.pth` 或 `model_current_state_dict.pth`

### 3. 新增 Mini2 专用 inference 入口

文件：

```text
scripts/inference/inference_mini2.py
```

Mini2 当前使用独立入口，而不是继续复用 Panda 默认脚本。这样做的原因有三点：

- Mini2 需要独立的模型目录和 checkpoint 选择
- Mini2 需要独立控制 `use_extra_objects`、Isaac 渲染和排障参数
- Mini2 的结果目录需要固定落在 `scripts/inference/logs/mini2/<seed>`

当前入口保留的关键参数包括：

```python
model_dir: str = ''
checkpoint_path: str = ''
use_extra_objects: bool = True
use_guide_on_extra_objects_only: bool = False
use_isaac_gym: bool = True
interpolate_for_isaac_replay: bool = False
num_interpolation_for_isaac_replay: int = 2
isaac_debug_contact_summary: bool = False
isaac_asset_root: str = ''
results_dir: str = '<repo>/scripts/inference/logs/mini2'
```

另外，`results_dir` 不再由脚本内部再追加一次 `/seed`。当前直接使用 `single_experiment_yaml` 生成的 `<results_dir>/<seed>` 目录，避免出现 `.../1/1` 这样的重复层级。

### 4. 整理 Isaac 回放插值逻辑

文件：

```text
scripts/inference/inference.py
scripts/inference/inference_mini2.py
deps/torch_robotics/torch_robotics/trajectory/utils.py
```

当前 Panda / Mini2 两条 Isaac 回放路径都支持两种模式：

- `interpolate_for_isaac_replay=True`
  只在回放入口对关节位置轨迹插值
- `interpolate_for_isaac_replay=False`
  直接使用原始 support points 回放

当前 Panda / Mini2 默认都为 `False`，因为这两条线都已经进入 Isaac 回放排查阶段，不应自动引入额外轨迹点。

`interpolate_traj_via_points(...)` 当前状态：

- 插值权重是标准线性形式  
  `start * (1 - alpha) + end * alpha`
- 保留了 `preserve_start_end` 参数
- Mini2 当前回放路径不使用 `preserve_start_end`

这样处理的目的是把“正确的插值实现”和“是否在 Isaac 回放里启用插值”拆开。函数本身保持正确实现，回放路径默认不使用它。

### 5. Mini2 接入 Isaac Gym 渲染

文件：

```text
deps/torch_robotics/torch_robotics/isaac_gym_envs/motion_planning_envs.py
scripts/inference/inference_mini2.py
```

Mini2 当前不是复用 Panda 的写死分支，而是接入到底层通用 `MotionPlanningIsaacGymEnv`，再通过 `Mini2MotionPlanningIsaacGymEnv` 提供 Mini2 专用包装。

Mini2 对接时显式指定：

```text
asset_root = <torch_robotics>/data/urdf/robots
asset_file = Mini2/urdf/Mini2_isaac.urdf
end_effector_link_name = 6_Link
arm_dof_count = 6
flip_visual_attachments = False
```

其中 `Mini2_isaac.urdf` 不是手写维护文件，而是运行前从 `Mini2.urdf` 自动生成，唯一目的就是把：

```text
package://Mini2/meshes/...
```

改成 Isaac Gym 能稳定加载的相对路径：

```text
../meshes/...
```

### 6. 新增 `EnvSpheres3DMini2ExtraObjects` 并调整当前布局

文件：

```text
deps/torch_robotics/torch_robotics/environments/env_spheres_3d_mini2_extra_objects.py
deps/torch_robotics/torch_robotics/environments/__init__.py
```

新增原因：

- `use_extra_objects=True` 时，Mini2 需要自己的 extra-objects 场景类
- 原仓库只有 Panda 对应的 `EnvSpheres3DExtraObjects`

当前 Mini2 extra objects 配置是：

```text
center [-0.08,  0.14, 0.54], radius 0.08
center [ 0.26, -0.16, 0.30], radius 0.08
center [-0.22, -0.12, 0.22], radius 0.08
center [ 0.06,  0.32, 0.60], radius 0.08
```

这一版布局不是初始版本。当前布局是经过 Isaac 排障后保留下来的稳定版本，目的不是让场景变简单，而是避免在 Mini2 中段扫掠区域形成不必要的局部拥挤。

## 已完成排查

### 1. 轨迹步数与视频显示

已确认：

- 真实轨迹 horizon 是 `64`
- `results_data_dict.pickle` 中保存的 `trajs_final_free.shape` 为 `(B, 64, D)`
- Isaac 视频总帧数包含前置停留帧和末尾停留帧，因此视频总帧不等于纯轨迹步数

因此，视频叠字里出现的较小步数不代表轨迹真的少了一步。这个问题只影响显示理解，不影响轨迹本体或碰撞判断。

### 2. 回放插值不是主因

排查方式：

- 给 Isaac 回放新增 `interpolate_for_isaac_replay` 开关
- 在完全不插值的条件下重跑 Mini2

结论：

- 关闭插值后，Isaac 自由轨迹数只有小幅改善
- 问题没有被根本解决

因此这次问题的主因不是回放插值。当前代码仍保留这个开关，但 Mini2 默认不插值。

### 3. Isaac self-collision 不是主因

排查方式：

- 临时关闭 Isaac 中 actor 的 self-collision
- 保持其余设置不变，重跑同一组 inference

结论：

- 自由轨迹数没有明显回升

因此这次问题的主因不是 self-collision。当前代码已经恢复到原始 self-collision 设置，不再把这个假设作为当前问题的主要解释。

### 4. 接触来源已经被定位

为排查接触来源，在 Isaac 侧增加了可选调试开关：

```python
isaac_debug_contact_summary
```

对应实现位于：

```text
scripts/inference/inference_mini2.py
deps/torch_robotics/torch_robotics/isaac_gym_envs/motion_planning_envs.py
```

打开后，Isaac 会在每条轨迹第一次发生接触时打印：

- 第几步发生接触
- 哪个 env 发生接触
- 接触类型
- 接触的刚体名称

这轮排查得到的稳定结果是：

- 失败几乎全部是 `robot-extra-object`
- 接触的机器人 link 几乎都固定为 `3_Link`
- 接触时间高度集中在 `step=36` 或 `step=37`
- 接触的 extra object 实际对应同一颗局部球体

因此这次问题已经不再是“Isaac 到底撞到了什么”的开放问题，而是明确收敛成：

- `3_Link`
- 某一颗 extra sphere
- 中段执行区间

### 5. 通过调整 extra objects 布局解决了当前主要问题

在定位到接触集中在同一颗 extra sphere 后，对 `EnvSpheres3DMini2ExtraObjects` 做了有针对性的轻微外移。

旧布局：

```text
[-0.05,  0.08, 0.52]
[ 0.22, -0.12, 0.32]
[-0.18, -0.05, 0.28]
[ 0.00,  0.26, 0.58]
```

当前布局：

```text
[-0.08,  0.14, 0.54]
[ 0.26, -0.16, 0.30]
[-0.22, -0.12, 0.22]
[ 0.06,  0.32, 0.60]
```

调整原则：

- 保持障碍物数量不变
- 保持半径 `0.08` 不变
- 重点把集中触发接触的局部球体外移
- 其余球体只做小幅拉开，避免新的局部堆叠

排查结果：

- 调整后，Isaac 中几乎所有轨迹都恢复为可行

因此，这一轮 Mini2 inference 的主要问题已经可以明确归结为：

- `extra objects` 的旧布局在 Mini2 工作空间里过于密集
- 问题表现为 `3_Link` 在中段动作时与同一颗 extra sphere 集中接触
- 调整布局后，主要问题得到解决

## 当前结论

对于当前这套 Mini2 inference 配置，应这样理解：

1. 规划链路本身已经基本可用  
   问题不在“所有轨迹都不合理”，而在 Isaac 场景布局与 Mini2 工作空间的局部冲突。

2. Isaac 回放插值不是这轮问题的主因  
   因此 Mini2 当前默认保持 `interpolate_for_isaac_replay=False`。

3. self-collision 不是这轮问题的主因  
   已做过验证并恢复原设置。

4. 当前最重要的有效修复是 extra objects 布局调整  
   这是这轮排查后真正保留下来的场景级修改。

5. `isaac_debug_contact_summary` 应视为诊断工具  
   默认关闭；只有在下一次需要定位接触来源时再打开。
