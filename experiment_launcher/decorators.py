import datetime

import os
from functools import wraps, partial

from experiment_launcher.utils import save_args, start_wandb, create_results_dir


def wrapper_single_experiment(exp_func, save_args_yaml=False):
    @wraps(exp_func)
    def wrapper(
        # Function arguments
        *args,
        **kwargs
    ):
        # Make results directory
        kwargs = create_results_dir(**kwargs)

        # Save arguments
        save_args(kwargs['results_dir'], kwargs, git_repo_path='./', save_args_as_yaml=save_args_yaml, verbose=True)

        # Start WandB
        wandb_run = start_wandb(**kwargs)

        # Run the experiment
        exp_func(*args, **kwargs)

        # Clean up
        wandb_run.finish()

    return wrapper


single_experiment_flat = partial(wrapper_single_experiment)
single_experiment_yaml = partial(wrapper_single_experiment, save_args_yaml=True)
