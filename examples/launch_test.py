from itertools import product

from experiment_launcher import Launcher
from experiment_launcher.utils import bool_local_cluster

LOCAL = bool_local_cluster()
TEST = False
USE_CUDA = False

N_SEEDS = 3

if LOCAL:
    JOBLIB_PARALLEL_JOBS = 5  # os.cpu_count()
else:
    JOBLIB_PARALLEL_JOBS = 1

N_CORES = 1
MEMORY_SINGLE_JOB = 1000
MEMORY_PER_CORE = MEMORY_SINGLE_JOB // N_CORES
PARTITION = 'amd2,amd'  # 'amd', 'rtx'
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
    compact_dirs=True
)

envs = {
    'env_00': {'env_param': 'aa'},
    'env_01': {'env_param': 'bb'}
}
a_l = [1, 2, 3]
boolean_param_l = [True, False]
some_default_param = 'b'
# These arguments are kwargs in the experiment function
unknown_args_list = [
    dict(integer_arg=10),
    dict(floating_arg=11.0, string_arg='test')
]

for env in envs:
    d = envs[env]
    for a, boolean_param in product(a_l, boolean_param_l):
        for unknown_args in unknown_args_list:
            launcher.add_experiment(
                # A subdirectory will be created for parameters with a trailing double underscore.
                env__=env,
                a__=a,
                # boolean_param is changing, but we forgot to add the trailing underscores.
                # The launcher adds a random number to the results directory to prevent overwriting previous experiments.
                boolean_param=boolean_param,

                env='some_env',  # This value will be overwritten by env__

                **d,
                some_default_param=some_default_param,

                **unknown_args,

                # Optional arguments for Weights and Biases
                wandb_enabled=False,  # If True, runs wandb. It is False by default.
                wandb_entity='joaocorreiacarvalho',
                wandb_project='test_experiment_launcher',
                wandb_group=f'test_group-el-{env}-{a}={boolean_param}'
            )

launcher.run(LOCAL, TEST)
