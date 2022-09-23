from itertools import product

from experiment_launcher import Launcher
from experiment_launcher.utils import bool_local_cluster

LOCAL = bool_local_cluster()
TEST = False
USE_CUDA = False

PARTITION = 'amd'  # 'amd', 'rtx', 'test30m' for lichtenberg
GRES = 'gpu:1' if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1

CONDA_ENV = 'py38'  # None

N_SEEDS = 12

if LOCAL:
    JOBLIB_PARALLEL_JOBS = 5  # os.cpu_count()
else:
    JOBLIB_PARALLEL_JOBS = 1

N_CORES_JOB = 1
MEMORY_SINGLE_JOB = 1000

launcher = Launcher(exp_name='test_launcher',
                    python_file='test',
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
    d = envs[env]
    for a, boolean_param in product(a_l, boolean_param_l):
        launcher.add_experiment(
            **d,
            some_default_param=some_default_param,

            # A subdirectory will be created for parameters with a trailing double underscore.
            env__=env,
            a__=a,
            boolean_param__=boolean_param,

            env='some_env',  # This value will be overwritten by env__

            # Weights and Biases
            wandb_enabled=False,
            wandb_entity='jaccarvalho',
            wandb_project='test_experiment_launcher',
            wandb_group=f'test_group-{env}-{a}={boolean_param}'
        )

launcher.run(LOCAL, TEST)
