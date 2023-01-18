import os
from itertools import product

from experiment_launcher import Launcher
from experiment_launcher.utils import is_local

LOCAL = is_local()
TEST = True
USE_CUDA = True

N_SEEDS = 3

if LOCAL:
    JOBLIB_PARALLEL_JOBS = 5  # os.cpu_count()
else:
    JOBLIB_PARALLEL_JOBS = 1

N_CORES = 1
MEMORY_SINGLE_JOB = 1000
MEMORY_PER_CORE = MEMORY_SINGLE_JOB // N_CORES
PARTITION = 'rtx2,rtx' if USE_CUDA else 'amd2,amd'
GRES = 'gpu:1' if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1
CONDA_ENV = 'el'  # None

launcher = Launcher(
    exp_name='test_launcher',
    python_file='test',
    # project_name='project01234',  # for hrz cluster
    n_exps=N_SEEDS,
    joblib_n_jobs=None,  # If experiments_per_gpu are specified, then joblib does not do anything
    experiments_per_gpu=3,
    # n_cores=JOBLIB_PARALLEL_JOBS * N_CORES_JOB,
    n_cores=N_CORES,
    memory_per_core=MEMORY_PER_CORE,
    days=2,
    hours=23,
    minutes=59,
    seconds=0,
    partition=PARTITION,
    conda_env=CONDA_ENV,
    gres=GRES,
    use_timestamp=True,
    compact_dirs=False
)

tensor_sizes_l = [1000000 + i for i in range(5)]

# Optional arguments for Weights and Biases
wandb_options = dict(
    wandb_enabled=False,  # If True, runs wandb. Default is False.
    wandb_entity='joaocorreiacarvalho',
    wandb_project='test_experiment_launcher_config_files'
)

for tensor_size in tensor_sizes_l:
    launcher.add_experiment(
        # A subdirectory will be created for parameters with a trailing double underscore.
        tensor_size__=tensor_size,

        debug=False,

        # **wandb_options,
        # wandb_group=f'test_group-el-{tensor_size}',
    )

launcher.run(LOCAL, TEST)
