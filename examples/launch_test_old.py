from itertools import product
from experiment_launcher import Launcher, bool_local_cluster

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
                    python_file='test_old',
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

a_list = [1, 2, 3]
b_c_list = [11, 12]
boolean_list = [True, False]

launcher.add_default_params(default='b')

for a, b_c, boolean in product(a_list, b_c_list, boolean_list):
    launcher.add_experiment(a=a,
                            b_c=b_c,
                            boolean=boolean)

launcher.run(LOCAL, TEST)
