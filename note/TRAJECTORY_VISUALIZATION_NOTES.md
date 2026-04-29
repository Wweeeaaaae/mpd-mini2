# Trajectory Visualization Notes

本文按“改动项”整理轨迹可视化相关变更，不再按修改顺序展开。目标是把当前两个可视化脚本的真实行为、各自职责、默认输出和已知限制写清楚。

## 改动范围

涉及文件：

```text
scripts/generate_data/visualize_trajectories.py
scripts/generate_data/visualize_trajectories_mini2.py
```

当前结构可以直接理解为：

- `visualize_trajectories.py`
  通用轨迹可视化入口
- `visualize_trajectories_mini2.py`
  Mini2 专用快捷入口

## 改动 1. 通用可视化脚本从硬编码脚本改成了 CLI 工具

文件：

```text
scripts/generate_data/visualize_trajectories.py
```

旧脚本的问题是：

- `DATA_DIR` 写死在源码里
- 想看别的结果必须手改文件
- 默认直接弹出 `matplotlib` 窗口
- 更像一次性调试脚本，不适合重复使用

当前脚本已经改成命令行入口，支持的核心参数包括：

```text
--data-dir
--logs-dir
--exp-name
--env-id
--robot-id
--device
--prefer
--max-trajs
--show
--no-video
--plot-full-trajs-in-video
--draw-collision-spheres
```

这些参数当前分别解决了以下问题：

- `--data-dir`
  直接指定某个结果目录
- `--logs-dir` + `--exp-name`
  在不手动给目录时自动找最近结果
- `--env-id` + `--robot-id`
  在自动查找时继续过滤实验类型
- `--prefer`
  控制优先展示 `free`、`collision` 或 `all`
- `--show`
  控制是否真的打开交互窗口
- `--plot-full-trajs-in-video`
  控制视频是否恢复旧的全轨迹叠加模式

## 改动 2. 通用脚本支持自动发现最近一次生成结果

文件：

```text
scripts/generate_data/visualize_trajectories.py
```

新增：

```python
find_latest_result_dir(...)
```

它当前扫描的目录形态是：

```text
logs/generate_trajectories_<timestamp>/env_id___*/robot_id___*/*/
```

自动发现结果目录时，会额外检查：

- `args.yaml`
- `metadata.yaml`
- `trajs-free.pt` 或 `trajs-collision.pt`

这项改动带来的实际变化是：

- 不需要再手改 `DATA_DIR`
- 可以直接配合 `launch_generate_trajectories.py` 的时间戳输出目录使用
- 可以通过 `--env-id` 和 `--robot-id` 避免拿错实验

## 改动 3. 通用脚本的轨迹加载逻辑做了容错

文件：

```text
scripts/generate_data/visualize_trajectories.py
```

新增：

```python
load_trajectories(data_dir, tensor_args, prefer)
```

当前行为：

- `trajs-free.pt` 不存在时，先退成空 tensor，而不是立刻报错
- `trajs-collision.pt` 不存在时，也允许继续处理
- `prefer='all'` 时会把已有的 free / collision 轨迹拼起来
- 只有两类轨迹都为空时才真正报错

相比旧版，这是一次明确的“从临时查看脚本，转成可重复使用工具”的改动。

## 改动 4. 默认输出模式改成“保存结果”，而不是依赖图形界面

文件：

```text
scripts/generate_data/visualize_trajectories.py
scripts/generate_data/visualize_trajectories_mini2.py
```

当前统一使用：

```python
if not cli_args.show:
    matplotlib.use('Agg')
```

配合：

```python
save_or_show(fig, output_path, show=...)
```

实际效果是：

- 在服务器、容器、远程终端中也能直接运行
- 默认行为是保存图片和视频
- 只有显式传 `--show` 才会打开 `matplotlib` 窗口

通用脚本的默认输出文件：

- `joint-trajectories.png`
- `robot-trajectories.png`
- `robot-traj.mp4`

Mini2 脚本的默认输出文件：

- `mini2-joint-trajectories.png`
- `mini2-robot-trajectories.png`
- `mini2-robot-traj-000.mp4` 到 `mini2-robot-traj-00N.mp4`

## 改动 5. 视频默认行为已经改成“不叠加整条轨迹”

这是这一轮最重要的可视化行为修正。

旧行为等价于：

```python
plot_trajs=True
```

对机械臂视频来说，这会导致：

- 每一帧都把整条轨迹重新画出来
- 多条轨迹同时存在时，线条大面积重叠
- 机械臂本体和轨迹线混在一起
- 实际运动过程反而更难看清

当前两个脚本都改成：

```python
plot_trajs=args_cli.plot_full_trajs_in_video
```

而 `--plot-full-trajs-in-video` 是 `store_true` 参数，因此默认行为就是：

```python
plot_trajs=False
```

也就是默认视频只保留：

- 当前时刻机械臂
- 障碍物
- 起点
- 终点

如果确实需要旧的“每一帧叠加整条轨迹”模式，再显式传：

```bash
--plot-full-trajs-in-video
```

当前应如何理解这条改动：

- 默认视频更适合看动作过程
- 旧的整条轨迹叠加模式没有删除
- 只是从默认行为改成了显式开关

## 改动 6. 可视化前会重建完整规划上下文，而不是只画 tensor

文件：

```text
scripts/generate_data/visualize_trajectories.py
scripts/generate_data/visualize_trajectories_mini2.py
```

两个脚本现在都会重新加载：

- `args.yaml`
- `metadata.yaml`
- `environment`
- `robot`
- `PlanningTask`

因此当前输出的不只是图片和视频，还包括重新计算的指标：

- `fraction_free`
- `collision_intensity`
- `success`

这项改动的意义是：

- 可视化和原始规划任务上下文保持一致
- 重新查看结果时可以顺手复核碰撞指标
- 出现“文件保存了，但任务语义不对”的情况时更容易排查

## 改动 7. 新增 Mini2 专用可视化脚本，固定 Mini2 的常用路径和校验

文件：

```text
scripts/generate_data/visualize_trajectories_mini2.py
```

这个脚本不是简单复制通用脚本，而是把 Mini2 的几个高频前提直接写死为默认行为。

### 7.1 默认按脚本顶部的 `num` 选择 Mini2 结果目录

当前脚本顶部有一行：

```python
num = 114
```

当不传 `--data-dir` 时，脚本会去 Mini2 日志目录里查找这个编号对应的目录。

这样做的意义是：

- 抽查时只需要改一个数字
- 不再默认落到“最新修改的那个 seed”
- 仍然保留 `--data-dir` 作为显式覆盖入口
- 如果不同批次里同一个 `num` 重复出现，脚本会直接报错，要求显式传 `--data-dir`

### 7.2 加了 Mini2 专属校验

脚本会显式检查：

```python
args['env_id'] == 'EnvSpheres3DMini2'
args['robot_id'] == 'RobotMini2'
```

如果目录指向别的实验，会直接报错，而不是把错误数据画出来。

### 7.3 输出文件名使用 Mini2 前缀

当前输出：

- `mini2-joint-trajectories.png`
- `mini2-robot-trajectories.png`
- `mini2-robot-traj-000.mp4`
- `mini2-robot-traj-001.mp4`
- ...

这样做的目的是：

- 避免 Mini2 与 Panda 结果混在一起时文件名冲突
- 让每条轨迹的视频单独查看，不再把多条轨迹堆进同一个视频

## 改动 8. Mini2 脚本在“结果目录容错”上仍然比通用脚本更严格

文件：

```text
scripts/generate_data/visualize_trajectories_mini2.py
```

Mini2 版本当前的 `load_trajectories(...)` 仍然是直接读取：

```python
trajs-free.pt
trajs-collision.pt
```

这意味着它当前默认假设这两个文件都存在。

与之对比：

- 通用脚本已经支持某一个轨迹文件缺失时继续运行
- Mini2 脚本目前还没有对齐这部分容错能力

不过在视频显示策略上，Mini2 已经与通用脚本保持一致：

- 默认不叠加整条轨迹
- 默认只看当前时刻机械臂
- `--plot-full-trajs-in-video` 才恢复旧行为

- 但视频现在会按轨迹分别导出，而不是合成一个总视频

## 改动 9. 2026-04-25 Mini2 抽查模式与逐轨迹视频导出

文件：

```text
scripts/generate_data/visualize_trajectories_mini2.py
```

本轮改动只针对 Mini2 脚本，目的有两个：

1. 让抽查某个固定编号目录更直接。
2. 避免多条轨迹共用一个视频导致画面过乱。

当前行为：

- 脚本顶部新增 `num = 114`
- 不传 `--data-dir` 时，默认按这个 `num` 去定位 Mini2 结果目录
- 如果多个批次同时匹配这个 `num`，脚本会直接报错，避免误选
- `--max-trajs` 仍然控制最多导出多少条轨迹，默认是 `5`
- 关节图和机器人静态图仍然是前 `max_trajs` 条轨迹的汇总图
- 视频改成逐轨迹导出：
  - `mini2-robot-traj-000.mp4`
  - `mini2-robot-traj-001.mp4`
  - ...

因此现在的 Mini2 可视化更适合做人工抽查：

- 想换目录时，直接改 `num`
- 想限制导出数量时，直接改 `--max-trajs`
- 想看某一条轨迹的视频，不会再被其他轨迹干扰

## 改动 10. 当前结论

当前这套可视化改动可以概括为四点：

1. 通用脚本已经从硬编码脚本整理成了可重复调用的 CLI 工具。
2. Mini2 已经有独立快捷入口，不需要每次手动拼路径。
3. 默认视频显示策略已经修正为“看得清机械臂怎么动”。
4. 可视化过程会重新加载规划上下文并复算指标，不只是把张量画出来。

因此后续不管是检查 Panda 数据、Mini2 数据，还是对照 free / collision 轨迹，这两个入口都已经比旧版更稳定，也更适合持续使用。
