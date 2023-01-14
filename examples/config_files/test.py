import argparse
import os

import wandb
import yaml

from experiment_launcher import run_experiment
from experiment_launcher.decorators import single_experiment_yaml


# This decorator is not mandatory.
# It creates results_dir as results_dir/seed, and saves the experiment arguments into a file.
@single_experiment_yaml
def experiment(
    #######################################
    config_file_path: str = 'configs/configs00.yaml',

    some_default_param: str = 'b',

    debug: bool = True,

    #######################################
    # MANDATORY
    seed: int = 0,
    results_dir: str = 'logs',
    **kwargs  # accept unknown arguments
):
    # EXPERIMENT
    print(f'DEBUG MODE: {debug}')

    with open(config_file_path, 'r') as f:
        configs = yaml.load(f, yaml.Loader)

    print('Config file content:')
    print(configs)

    wandb.log({'seed': seed}, step=1)


if __name__ == '__main__':
    # Leave unchanged
    run_experiment(experiment)


