# Train Diffusion Notes

本文只记录训练阶段相关的代码改动及其影响，不保留运行教程、命令示例或与改动无关的说明。

## 改动范围

涉及文件：

```text
scripts/train_diffusion/train.py
scripts/train_diffusion/launch_train_01.py
mpd/datasets/trajectories.py
deps/experiment_launcher/experiment_launcher/utils.py
```

## 改动 1. 修正验证集 subset 传递错误

文件：

```text
scripts/train_diffusion/train.py
```

原实现把：

```python
val_subset=train_subset
```

传进了 `trainer.train(...)`。当前已改为：

```python
val_subset=val_subset
```

影响：

- validation loss 仍然是在 `val_dataloader` 上算，这部分原本没有坏
- 真正被修正的是 validation summary 的抽样来源
- 修正后，验证可视化和验证统计不再误用训练子集

## 改动 2. 训练入口支持显式数据路径

文件：

```text
scripts/train_diffusion/train.py
mpd/datasets/trajectories.py
```

`TrajectoryDataset` 新增：

```python
dataset_path: str = ''
```

读取逻辑变成：

- `dataset_path` 非空时，直接从该目录读取
- `dataset_path` 为空时，仍从 `data_trajectories/<dataset_subdir>` 读取

影响：

- 不再需要把生成结果手动搬到 `data_trajectories`
- 可以直接拿 `scripts/generate_data/logs/...` 下的完整结果目录训练

## 改动 3. `launch_train_01.py` 支持按数据集配置路径

文件：

```text
scripts/train_diffusion/launch_train_01.py
```

原来脚本只维护：

```python
dataset_subdir_l = [...]
```

当前改成：

```python
dataset_configs_l = [
    dict(dataset_subdir='...', dataset_path='...'),
]
```

影响：

- 每个训练任务可以独立指定数据目录
- launcher 不再要求所有数据都预先整理到统一归档目录

## 改动 4. WandB 接入改为真正可用

文件：

```text
deps/experiment_launcher/experiment_launcher/utils.py
scripts/train_diffusion/train.py
scripts/train_diffusion/launch_train_01.py
```

这轮前后实际修了两件事。

### 4.1 `wandb_mode` 现在真正生效

之前训练脚本虽然传了：

```python
wandb_mode = 'online' / 'offline' / 'disabled'
```

但底层 `start_wandb(...)` 实际只看 `wandb_enabled`，导致：

- 训练脚本表面上配置了 `wandb_mode`
- 实际并没有按这个字段控制 WandB 初始化

当前 `start_wandb(...)` 已改成直接识别：

```python
wandb_mode in {'online', 'offline', 'disabled'}
```

影响：

- `online` 会正常同步到 WandB
- `offline` 会只写本地 run 目录
- `disabled` 会完全关闭 WandB 记录

### 4.2 默认不再硬编码不存在的 entity

之前 launcher 里默认写的是：

```python
wandb_entity = 'scoreplan'
```

如果当前登录账号下不存在这个 entity，会直接报：

```text
entity scoreplan not found during upsertBucket
```

当前已改成：

```python
wandb_entity = ''
```

并且只有当 `wandb_entity` 非空时，才显式传给 `wandb.init(...)`。

影响：

- 默认使用当前 `wandb login` 对应的账号
- 不再因为仓库里遗留的 team 名导致 `404 entity not found`

## 改动 5. WandB run 名改成固定的可读实验名

文件：

```text
scripts/train_diffusion/launch_train_01.py
```

当前 launcher 不再让 WandB 自动生成随机英文名，而是按这轮实验的真实差异固定命名。

两条 run 名分别为：

```text
EnvSpheres3DMini2-RobotMini2-unet_3stage_32-64-128
EnvSpheres3DMini2-RobotMini2-unet_4stage_32-64-128-256
```

对应含义：

- `unet_3stage_32-64-128`
  对应 `unet_dim_mults_option = 0`
- `unet_4stage_32-64-128-256`
  对应 `unet_dim_mults_option = 1`

这两个名字直接反映了当前同时运行的两个实验，本质上是在比较：

- 较浅的 3-stage U-Net
- 较深的 4-stage U-Net

同时，WandB group 也改成了统一的：

```text
EnvSpheres3DMini2-RobotMini2-unet_depth_ablation
```

影响：

- WandB 页面上的两个 run 名现在可以直接看出网络规模差异
- 不再出现与实验内容无关的随机英文名
- 两个 run 会被归到同一个结构清晰的 group 下

## 结论

这轮训练相关代码改动可以概括为四点：

1. 修正 validation subset 传错
2. 支持显式数据路径
3. 让 launcher 能按任务配置数据路径
4. 让 WandB 的模式和账号选择真正可用
5. 让 WandB run 名和 group 直接反映当前的 U-Net 深度对比实验
