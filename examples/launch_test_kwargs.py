from itertools import product

from experiment_launcher import Launcher
from experiment_launcher.utils import bool_local_cluster

LOCAL = bool_local_cluster()
TEST = False
USE_CUDA = False

PARTITION = 'amd'  # 'amd', 'rtx', 'test30m' for lichtenberg
GRES = 'gpu:1' if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1

CONDA_ENV = 'el'  # None

N_SEEDS = 12

if LOCAL:
    JOBLIB_PARALLEL_JOBS = 5  # os.cpu_count()
else:
    JOBLIB_PARALLEL_JOBS = 1

N_CORES_JOB = 1
MEMORY_SINGLE_JOB = 1000

launcher = Launcher(exp_name='test_launcher_kwargs',
                    python_file='test_kwargs',
                    # project_name='project01234',
                    n_exps=N_SEEDS,
                    joblib_n_jobs=JOBLIB_PARALLEL_JOBS,
                    n_cores=JOBLIB_PARALLEL_JOBS * N_CORES_JOB,
                    memory_per_core=MEMORY_SINGLE_JOB,
                    days=0,
                    hours=0,
                    minutes=2,
                    seconds=0,
                    partition=PARTITION,
                    conda_env=CONDA_ENV,
                    gres=GRES,
                    use_timestamp=True
                    )

envs = {
    'env_00': {'env_param': 'aa'},
    'env_01': {'env_param': 'bb'}
}

unknown_args_list = [
    dict(integer=10),
    dict(floating=11.0, string='test')
]

for env, unknown_args in zip(envs, unknown_args_list):
    d = envs[env]
    launcher.add_experiment(
        **d,
        **unknown_args,
        # A subdirectory will be created for parameters with a trailing double underscore.
        env__=env,
    )

launcher.run(LOCAL, TEST)
