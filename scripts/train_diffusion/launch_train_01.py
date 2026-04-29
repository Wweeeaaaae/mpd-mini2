import os
from itertools import product

from experiment_launcher import Launcher
from experiment_launcher.utils import is_local

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  #设置环境变量，禁用文件锁定

########################################################################################################################
#  启动器launcher（初始化）

LOCAL = is_local()
TEST = False
USE_CUDA = True

N_SEEDS = 1

N_EXPS_IN_PARALLEL = 4  #实验并行运行的数量

N_CORES = N_EXPS_IN_PARALLEL * 4  #总共需要的核心数
MEMORY_SINGLE_JOB = 12000  #每个任务分配的内存
MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
PARTITION = 'gpu' if USE_CUDA else 'amd3,amd2,amd'  #指定计算分区
GRES = 'gpu:1' if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1, gpu:rtx3090:1, gpu:a5000:1  #每个任务请求一个GPU资源
CONDA_ENV = 'mpd'


exp_name = f'train_diffusion'  #存储实验名称

launcher = Launcher(
    exp_name=exp_name,  # 实验名称
    exp_file='train',  # 运行单次实验的Python模块名称
    # project_name='project01234',
    n_seeds=N_SEEDS,  # 实验的随机种子数量
    n_exps_in_parallel=N_EXPS_IN_PARALLEL,  # 每次并行运行的实验数量
    n_cores=N_CORES,  # 所需核心数
    memory_per_core=MEMORY_PER_CORE,  # 每个核心的内存分配
    days=2,  # 最大运行时间（天）
    hours=23,  # 最大运行时间（小时）
    minutes=59,  # 最大运行时间（分钟）
    seconds=0,  # 最大运行时间（秒）
    partition=PARTITION,  # 计算分区
    conda_env=CONDA_ENV,  # Conda环境
    gres=GRES,  # GPU资源配置
    use_timestamp=True  # 是否在实验名称中包含时间戳
)

########################################################################################################################
#  实验参数设置

dataset_configs_l = [
     dict(dataset_subdir='EnvSpheres3DMini2-RobotMini2',
          dataset_path='/home/yuancz/PycharmProjects/mpd-public/scripts/generate_data/logs/'
                        'generate_trajectories_mini2_2026-04-23_16-55-24/'
                        'env_id___EnvSpheres3DMini2/robot_id___RobotMini2'
        ),
]

include_velocity_l = [
    True
]

use_ema_l = [  #使用指数移动平均（EMA）来平滑模型参数
    True
]

variance_schedule_l = [  #方差调度策略，指数衰减
    'exponential'
]

n_diffusion_steps_l = [  #扩散步骤数量
    25,
]

predict_epsilon_l = [
    True
]

dim = 32

unet_dim_mults_option_l = [
    0,
    1
]


batch_size = 128
lr = 3e-4  #学习率


wandb_options = dict(
    wandb_mode='online',  # wandb运行模式 "online"实验数据同步云端, "offline"保存在本地 or "disabled"不记录实验数据
    wandb_entity='',  # 为空时默认使用当前 wandb login 对应的账号
    wandb_project=exp_name   #项目名称
)

########################################################################################################################
#  运行学习过程

for dataset_config, include_velocity, use_ema, variance_schedule, n_diffusion_steps, predict_epsilon, unet_dim_mults_option in \
        product(dataset_configs_l, include_velocity_l, use_ema_l, variance_schedule_l, n_diffusion_steps_l, predict_epsilon_l, unet_dim_mults_option_l):
    #  遍历所有参数的组合
    dataset_subdir = dataset_config['dataset_subdir']
    dataset_path = dataset_config['dataset_path']
    unet_variant_name = {
        0: 'unet_3stage_32-64-128',
        1: 'unet_4stage_32-64-128-256',
    }.get(unet_dim_mults_option, f'unet_option_{unet_dim_mults_option}')
    wandb_group = f'{dataset_subdir}-unet_depth_ablation'
    wandb_run_name = f'{dataset_subdir}-{unet_variant_name}'

    #  添加每个实验的独立配置
    launcher.add_experiment(
        dataset_subdir__=dataset_subdir,  # 数据集子目录
        dataset_path__=dataset_path,  # 数据集路径；为空时仍从 data_trajectories/<dataset_subdir> 读取
        include_velocity__=include_velocity,  # 是否包含速度信息
        use_ema__=use_ema,  # 是否使用EMA
        variance_schedule__=variance_schedule,  # 方差调度策略
        n_diffusion_steps__=n_diffusion_steps,  # 扩散步骤数量
        predict_epsilon__=predict_epsilon,  # 是否预测epsilon
        unet_dim_mults_option__=unet_dim_mults_option,  # U-Net的维度变化选项

        lr=lr,  # 学习率

        batch_size=batch_size,  # 批量大小
        num_train_steps=500000,  # 总训练步数
        steps_til_ckpt=50000,  # 每隔50000步保存模型检查点
        steps_til_summary=20000,  # 每隔20000步生成训练摘要

        **wandb_options,  # 传入Wandb的配置
        wandb_group=wandb_group,
        wandb_run_name=wandb_run_name,
        debug=False,  # 是否启用调试模式
    )

launcher.run(LOCAL, TEST)
