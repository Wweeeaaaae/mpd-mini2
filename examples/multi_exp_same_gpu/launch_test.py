import os
from itertools import product

from experiment_launcher import Launcher
from experiment_launcher.utils import is_local

LOCAL = is_local()
LOCAL = False
TEST = False
USE_CUDA = True

N_SEEDS = 1

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
    joblib_n_jobs=None if JOBLIB_PARALLEL_JOBS == 1 else JOBLIB_PARALLEL_JOBS,
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

tensor_sizes_l = 6 * [10000]

for tensor_size in tensor_sizes_l:
    launcher.add_experiment(
        # A subdirectory will be created for parameters with a trailing double underscore.
        tensor_size__=tensor_size,

        debug=False
    )

launcher.run(LOCAL, TEST)
