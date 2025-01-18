import os
import torch

from experiment_launcher import single_experiment_yaml, run_experiment
from mpd import trainer
from mpd.models import UNET_DIM_MULTS, TemporalUnet
from mpd.trainer import get_dataset, get_model, get_loss, get_summary
from mpd.trainer.trainer import get_num_epochs
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


@single_experiment_yaml
def experiment(
    ########################################################################
    #  数据集
    dataset_subdir: str = 'EnvSimple2D-RobotPointMass',
    # dataset_subdir: str = 'EnvSpheres3D-RobotPanda',
    include_velocity: bool = True,

    ########################################################################
    #  扩散模型
    diffusion_model_class: str = 'GaussianDiffusionModel',
    variance_schedule: str = 'exponential',  #噪声由少变多，可选择cosine
    n_diffusion_steps: int = 25,
    predict_epsilon: bool = True,

    #  Unet网络
    unet_input_dim: int = 32,  #输入维度
    unet_dim_mults_option: int = 1,  #各层的维度倍数
    # 1: (1, 2, 4, 8)

    ########################################################################
    #  损失函数
    loss_class: str = 'GaussianDiffusionLoss',

    #  训练参数
    batch_size: int = 32,
    lr: float = 1e-4,
    num_train_steps: int = 500000,   #迭代次数

    use_ema: bool = True,  #使用指数移动平均（ EMA）来平滑模型参数
    use_amp: bool = False,  #使用自动混合精度（ AMP）来训练模型

    #  总结参数
    steps_til_summary: int = 10,  #训练总结频率
    summary_class: str = 'SummaryTrajectoryGeneration',  #生成训练总结的类

    steps_til_ckpt: int = 50000,  #保存模型检查点的频率

    ########################################################################
    device: str = 'cuda',

    debug: bool = True,

    ########################################################################
    #  MANDATORY
    seed: int = 0,
    results_dir: str = 'logs',

    ########################################################################
    #  WandB
    wandb_mode: str = 'disabled',  # "online", "offline" or "disabled"
    wandb_entity: str = 'scoreplan',
    wandb_project: str = 'test_train',
    **kwargs
):
    fix_random_seed(seed)

    device = get_torch_device(device=device)
    tensor_args = {'device': device, 'dtype': torch.float32}

    #  获取训练集
    train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        dataset_class='TrajectoryDataset',
        include_velocity=include_velocity,
        dataset_subdir=dataset_subdir,
        batch_size=batch_size,
        results_dir=results_dir,
        save_indices=True,
        tensor_args=tensor_args
    )

    dataset = train_subset.dataset

    # 模型参数设置
    diffusion_configs = dict(
        variance_schedule=variance_schedule,  # 方差调度策略，控制噪声的增加或减少
        n_diffusion_steps=n_diffusion_steps,  # 扩散步骤数，决定模型执行扩散过程的次数
        predict_epsilon=predict_epsilon,  # 是否预测噪声项ε
    )

    unet_configs = dict(
        state_dim=dataset.state_dim,  # 从数据集中获取状态空间的维度
        n_support_points=dataset.n_support_points,  # 从数据集中获取支持点的数量
        unet_input_dim=unet_input_dim,  # U-Net网络的输入维度
        dim_mults=UNET_DIM_MULTS[unet_dim_mults_option],  # 根据选项获取U-Net各层的维度倍数
    )

    model = get_model(
        model_class=diffusion_model_class,  # 指定使用的扩散模型类别
        model=TemporalUnet(**unet_configs),  # 创建TemporalUnet实例，并传入U-Net配置
        tensor_args=tensor_args,  # 指定设备和数据类型参数
        **diffusion_configs,  # 将扩散模型配置参数传入模型
        **unet_configs  # 将U-Net配置参数传入模型
    )

    # Loss
    loss_fn = val_loss_fn = get_loss(
        loss_class=loss_class
    )

    # Summary
    summary_fn = get_summary(
        summary_class=summary_class,
    )

    # Train
    trainer.train(
        model=model,
        train_dataloader=train_dataloader,
        train_subset=train_subset,
        val_dataloader=val_dataloader,
        val_subset=train_subset,
        epochs=get_num_epochs(num_train_steps, batch_size, len(dataset)),
        model_dir=results_dir,
        summary_fn=summary_fn,
        lr=lr,
        loss_fn=loss_fn,
        val_loss_fn=val_loss_fn,
        steps_til_summary=steps_til_summary,
        steps_til_checkpoint=steps_til_ckpt,
        clip_grad=True,
        use_ema=use_ema,
        use_amp=use_amp,
        debug=debug,
        tensor_args=tensor_args
    )


if __name__ == '__main__':
    # Leave unchanged
    run_experiment(experiment)
