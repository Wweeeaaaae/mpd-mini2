import wandb
import yaml

from experiment_launcher import run_experiment, single_experiment_wandb


# This decorator is not mandatory.
# It creates results_dir as results_dir/seed, and saves the experiment arguments into a file.
@single_experiment_wandb
def experiment(
    #######################################
    config_file_path: str = './configs/config00.yaml',

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
