import argparse
import os

from experiment_launcher import run_experiment
from experiment_launcher.decorators import single_experiment
from experiment_launcher.launcher import add_launcher_base_args, get_experiment_default_params


# This decorator is not mandatory.
# It creates results_dir as results_dir/seed, and saves the experiment arguments into a file.
@single_experiment
def experiment(
    env: str = 'env',  # You need to specify the argument type if you use the automatic parser.
    env_param: str = 'aa',
    a: int = 1,
    boolean_param: bool = True,
    some_default_param: str = 'b',

    seed: int = 0,  # This argument is mandatory
    results_dir: str = '/tmp'  # This argument is mandatory
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


# You can specify your own parser, or use the experiment_launcher parser.
def parse_args():
    parser = argparse.ArgumentParser()

    # Place your experiment arguments here
    arg_test = parser.add_argument_group('Test')
    arg_test.add_argument("--env", type=str)
    arg_test.add_argument("--env-param", type=str)
    arg_test.add_argument("--a", type=int)
    arg_test.add_argument("--boolean-param", action='store_true')
    arg_test.add_argument('--some-default-param', type=str)

    # Leave unchanged
    parser = add_launcher_base_args(parser)
    parser.set_defaults(**get_experiment_default_params(experiment))
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    # Leave unchanged
    run_experiment(experiment)

    # To use your own parser, run instead.
    # args = parse_args()
    # run_experiment(experiment, args)

