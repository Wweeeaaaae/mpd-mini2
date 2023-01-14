import argparse
import os

import wandb

from experiment_launcher import run_experiment
from experiment_launcher.decorators import single_experiment_yaml
from experiment_launcher.launcher import add_launcher_base_args, get_experiment_default_params


# This decorator is not mandatory.
# It creates results_dir as results_dir/seed, and saves the experiment arguments into a file.
@single_experiment_yaml
def experiment(
    #######################################
    env: str = 'env',  # You need to specify the argument type if you use the automatic parser.
    env_param: str = 'aa',
    a: int = 1,
    boolean_param: bool = True,
    some_default_param: str = 'b',

    #######################################
    # MANDATORY
    seed: int = 0,
    results_dir: str = 'logs',

    **kwargs  # accept unknown arguments
):
    # EXPERIMENT
    filename = os.path.join(results_dir, 'log_' + str(seed) + '.txt')
    out_str = f'Running experiment with seed {seed}, env {env}, ' \
              f'env_param {env_param}, a {a}, ' \
              f'boolean_param {boolean_param}, some_default_param {some_default_param}'
    print(out_str)
    with open(filename, 'w') as file:
        file.write('Some logs in a log file.\n')
        file.write(out_str)

    wandb.log({'seed': seed}, step=1)


if __name__ == '__main__':
    # Leave unchanged
    run_experiment(experiment)


