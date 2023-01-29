import datetime
import logging

import os
import sys
from functools import wraps, partial

from experiment_launcher.utils import save_args, start_wandb, create_results_dir


def wrapper_single_experiment(exp_func, save_args_yaml=False, use_logging=False):
    @wraps(exp_func)
    def wrapper(
        # Function arguments
        *args,
        **kwargs
    ):
        # Make results directory
        kwargs = create_results_dir(**kwargs)

        # Setup logging
        if use_logging:
            logging.basicConfig(level=logging.INFO,
                                filename=os.path.join(kwargs['results_dir'], "logfile"),
                                # stream=sys.stdout,
                                filemode="a+",
                                format="%(asctime)-15s %(levelname)-8s %(message)s")

        # Save arguments
        save_args(kwargs['results_dir'], kwargs, git_repo_path='./', save_args_as_yaml=save_args_yaml, verbose=True)

        # Start WandB
        wandb_run = start_wandb(**kwargs)

        # Run the experiment
        exp_func(*args, **kwargs)

        # Clean up
        wandb_run.finish()

    return wrapper


single_experiment = partial(wrapper_single_experiment)
single_experiment_wandb = partial(wrapper_single_experiment, save_args_yaml=True, use_logging=True)
