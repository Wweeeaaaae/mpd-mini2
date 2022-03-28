import os

from experiment_launcher import run_experiment
from experiment_launcher.decorators import single_experiment


@single_experiment
def experiment(
    env: str = 'env',
    env_param: str = 'aa',
    a: int = 1,
    boolean_param: bool = True,
    some_default_param: str = 'b',

    seed: int = 0,  # This argument is mandatory
    results_dir: str = '/tmp'  # This argument is mandatory
):
    # EXPERIMENT
    filename = os.path.join(results_dir, 'log_' + str(seed) + '.txt')
    os.makedirs(results_dir, exist_ok=True)
    print(f'Running experiment with seed {seed}, '
          f'env {env}, env_param {env_param}, a {a}, '
          f'boolean_param {boolean_param}, some_default_param {some_default_param}')
    with open(filename, 'w') as file:
        file.write('Some logs in a log file.')


if __name__ == '__main__':
    # Leave unchanged
    run_experiment(experiment)
