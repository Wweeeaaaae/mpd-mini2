from itertools import product

from experiment_launcher import Launcher
from experiment_launcher.utils import bool_local_cluster

LOCAL = bool_local_cluster()
TEST = False
USE_CUDA = False

PARTITION = 'test30m'  # 'amd', 'rtx'
GRES = 'gpu:1' if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1

CONDA_ENV = 'py38'  # None

N_SEEDS = 12

JOBLIB_PARALLEL_JOBS = 5  # os.cpu_count() to use all cores
N_CORES_JOB = 2
MEMORY_SINGLE_JOB = 1000

launcher = Launcher(exp_name='test_launcher',
                    python_file='test',
                    # python_file='test_old',
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
a_l = [1, 2, 3]
boolean_param_l = [True, False]
some_default_param = 'b'

for env in envs:
    launcher.add_default_params(
        env_param=envs[env]['env_param'],
        some_default_param=some_default_param
    )

    for a, boolean_param in product(a_l, boolean_param_l):
        launcher.add_experiment(
            env=env,
            a=a,
            boolean_param=boolean_param
        )

launcher.run(LOCAL, TEST)
