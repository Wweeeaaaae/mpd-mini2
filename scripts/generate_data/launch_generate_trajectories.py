import os
import socket
import numpy as np

from experiment_launcher import Launcher
from experiment_launcher.utils import is_local

########################################################################################################################
# 这里手动选择一个数据生成配置。
# 该脚本不会直接生成轨迹，而是用 Launcher 批量调度 generate_trajectories.py。
# - env_id: 环境类型
# - robot_id: 机器人类型
# - num_contexts: 要生成多少个不同的规划问题实例
# - num_trajectories_per_context: 每个问题实例生成多少条候选轨迹
# - threshold_start_goal_pos: 起点和终点之间的最小距离阈值，避免问题过于简单
# - obstacle_cutoff_margin: 碰撞检测时使用的安全边界

# env_id, robot_id, num_contexts, num_trajectories_per_context, threshold_start_goal_pos, obstacle_cutoff_margin = 'EnvSimple2D', 'RobotPointMass', 500, 20, 1, 0.02
# env_id, robot_id, num_contexts, num_trajectories_per_context, threshold_start_goal_pos, obstacle_cutoff_margin = 'EnvNarrowPassageDense2D', 'RobotPointMass', 500, 20, 1, 0.02
# env_id, robot_id, num_contexts, num_trajectories_per_context, threshold_start_goal_pos, obstacle_cutoff_margin = 'EnvDense2D', 'RobotPointMass', 500, 20, 1, 0.02
env_id, robot_id, num_contexts, num_trajectories_per_context, threshold_start_goal_pos, obstacle_cutoff_margin = 'EnvSpheres3D', 'RobotPanda', 500, 20, 1.83, 0.05  # 1.83 = 7 * np.deg2rad(15)


########################################################################################################################
# LAUNCHER

# 主机名常用于集群环境下的调试或区分不同机器；当前脚本里未进一步使用。
hostname = socket.gethostname()

# 判断当前是本地运行还是集群运行。
# Launcher 会根据这个标志决定是本地直接执行，还是走集群提交流程。
LOCAL = is_local()

# TEST=True 时通常只会做轻量测试；这里关闭，表示按完整配置运行。
TEST = False

# 是否启用 GPU。
# 这份脚本默认走 CPU 并发，因为数据生成主要依赖传统规划器。
# USE_CUDA = True
USE_CUDA = False

# Launcher 会为每个 seed 启动一个独立实验。
# 这里直接把 seed 数量设置为 num_contexts，等价于生成 num_contexts 个不同随机问题。
N_SEEDS = num_contexts

# CPU 模式下可以同时并发多个实验；
# GPU 模式下一般一个实验占一张卡，所以并发数退到 1。
N_EXPS_IN_PARALLEL = 15 if not USE_CUDA else 1

# 为整批并发实验申请的总 CPU 核数。
# N_CORES = N_EXPS_IN_PARALLEL
N_CORES = 8

# 估计单个实验的内存需求，并换算成调度器常用的“每核内存”配置。
MEMORY_SINGLE_JOB = 12000
MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES

# 根据是否使用 GPU 选择对应的分区和资源描述。
PARTITION = 'gpu' if USE_CUDA else 'amd3,amd2,amd'
GRES = 'gpu:1' if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1, gpu:rtx3090:1, gpu:a5000:1
CONDA_ENV = 'mpd-public'

# 这一批实验的名字，会出现在日志和输出目录里。
exp_name = f'generate_trajectories'

# 并行文件系统下 HDF5 文件锁有时会导致问题，这里显式关闭。
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Launcher 是 experiment_launcher 的批处理入口。
# 它负责记录“运行哪个脚本、跑多少个 seed、并行多少、资源多少、时限多久”。
launcher = Launcher(
    exp_name=exp_name,
    exp_file='generate_trajectories',
    # project_name='project01234',
    n_seeds=N_SEEDS,
    n_exps_in_parallel=N_EXPS_IN_PARALLEL,
    n_cores=N_CORES,
    memory_per_core=MEMORY_PER_CORE,
    days=0,
    hours=7,
    minutes=59,
    seconds=0,
    partition=PARTITION,
    conda_env=CONDA_ENV,
    gres=GRES,
    use_timestamp=True
)


########################################################################################################################
# RUN

# 注册一组实验参数。
# 注意这里只 add 一次，但因为 n_seeds=N_SEEDS，最终会展开成多次运行。
# 参数名后的双下划线 `__` 是 experiment_launcher 的约定： 表示这些字段会被注入到目标实验脚本的配置中。
launcher.add_experiment(
    env_id__=env_id,
    robot_id__=robot_id,

    # 每个上下文中希望最终生成的轨迹数量。
    num_trajectories=num_trajectories_per_context,

    # 这两个参数控制起终点采样难度和碰撞边界。
    threshold_start_goal_pos=threshold_start_goal_pos,
    obstacle_cutoff_margin=obstacle_cutoff_margin,

    # 目标脚本里使用的设备类型。
    device='cuda' if USE_CUDA else 'cpu',

    # 表示正式批量生成，而不是调试模式。
    debug=False
)

# 根据 LOCAL / TEST 设置，真正开始本地执行或集群提交。
launcher.run(LOCAL, TEST)
