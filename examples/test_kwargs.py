import os

from experiment_launcher import run_experiment
from experiment_launcher.decorators import single_experiment_flat


@single_experiment_flat
def experiment(
    env: str = 'env',  # You need to specify the argument type if you use the automatic parser.
    env_param: str = 'aa',

    seed: int = 0,  # This argument is mandatory
    results_dir: str = '/tmp',  # This argument is mandatory

    **kwargs  # accept unknown arguments
):
    # EXPERIMENT
    filename = os.path.join(results_dir, 'log_' + str(seed) + '.txt')
    out_str = f'Running experiment with seed {seed}, env {env}, ' \
              f'env_param {env_param}'
    if kwargs:
        out_str += "\nextra args:\n"
        for key, value in kwargs.items():
            out_str += f'{key} -> {value} ({type(value).__name__})\n'

    print(out_str)

    with open(filename, 'w') as file:
        file.write('Some logs in a log file.\n')
        file.write(out_str)


if __name__ == '__main__':
    run_experiment(experiment)

