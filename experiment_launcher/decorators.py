import os
from functools import wraps

from experiment_launcher.utils import save_args


def single_experiment(exp_func):
    @wraps(exp_func)
    def wrapper(*args, **kwargs):
        # Make results directory
        assert 'results_dir' in kwargs and 'seed' in kwargs, "results_dir and seed must be arguments"
        results_dir = os.path.join(kwargs['results_dir'], str(kwargs['seed']))
        os.makedirs(results_dir, exist_ok=True)
        kwargs['results_dir'] = results_dir
        # Save arguments
        save_args(results_dir, kwargs, git_repo_path='./')

        # Run the experiment
        exp_func(*args, **kwargs)

    return wrapper
