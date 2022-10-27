import datetime

import os
from functools import wraps, partial

from experiment_launcher.utils import save_args, fix_random_seed, start_wandb


def pseudo_single_experiment(exp_func, save_args_yaml=False, use_wandb=False):
    @wraps(exp_func)
    def wrapper(*args, **kwargs):
        # Make results directory
        assert 'results_dir' in kwargs and 'seed' in kwargs, "results_dir and seed must be arguments"
        results_dir = os.path.join(kwargs['results_dir'], str(kwargs['seed']))
        os.makedirs(results_dir, exist_ok=True)
        kwargs['results_dir'] = results_dir

        # Save arguments
        save_args(results_dir, kwargs, git_repo_path='./', save_args_yaml=save_args_yaml)
        print(kwargs)

        # Start WandB
        wandb_run = None
        if use_wandb:
            wandb_run = start_wandb(**kwargs)

        # Run the experiment
        exp_func(*args, **kwargs)

        # Clean up
        if use_wandb:
            wandb_run.finish()

    return wrapper


single_experiment = partial(pseudo_single_experiment)

single_experiment_yaml = partial(pseudo_single_experiment, save_args_yaml=True)

single_experiment_wandb = partial(pseudo_single_experiment, save_args_yaml=True, use_wandb=True)
