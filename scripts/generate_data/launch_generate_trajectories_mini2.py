import os
import socket
import numpy as np

from experiment_launcher import Launcher
from experiment_launcher.utils import is_local

########################################################################################################################
# Mini2 数据生成入口。
# 该脚本复用 generate_trajectories.py，只是把实验参数切到 Mini2。

env_id = 'EnvSpheres3DMini2'
robot_id = 'RobotMini2'
num_contexts = 500
num_trajectories_per_context = 20
threshold_start_goal_pos = float(6 * np.deg2rad(15))  # 约 1.57，建议范围 1.0-2.3
obstacle_cutoff_margin = 0.05


########################################################################################################################
# LAUNCHER

hostname = socket.gethostname()
LOCAL = is_local()
TEST = False

# 数据生成主要消耗在传统规划器，默认走 CPU 并发。
USE_CUDA = False

N_SEEDS = num_contexts
N_EXPS_IN_PARALLEL = 15 if not USE_CUDA else 1
N_CORES = 8

MEMORY_SINGLE_JOB = 12000
MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES

PARTITION = 'gpu' if USE_CUDA else 'amd3,amd2,amd'
GRES = 'gpu:1' if USE_CUDA else None
CONDA_ENV = 'mpd'

exp_name = 'generate_trajectories_mini2'

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

launcher = Launcher(
    exp_name=exp_name,
    exp_file='generate_trajectories',
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

launcher.add_experiment(
    env_id__=env_id,
    robot_id__=robot_id,
    num_trajectories=num_trajectories_per_context,
    threshold_start_goal_pos=threshold_start_goal_pos,
    obstacle_cutoff_margin=obstacle_cutoff_margin,
    device='cuda' if USE_CUDA else 'cpu',
    debug=False
)

launcher.run(LOCAL, TEST)
