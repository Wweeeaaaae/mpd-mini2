import os

from experiment_launcher import run_experiment, single_experiment


@single_experiment
def experiment(
    a: int = 1,
    b_c: int = 1,
    boolean: bool = True,
    default: str = 'dft',

    seed: int = 0,  # This argument is mandatory
    results_dir: str = '/tmp'  # This argument is mandatory
):
    # EXPERIMENT
    filename = os.path.join(results_dir, 'log_' + str(seed) + '.txt')
    os.makedirs(results_dir, exist_ok=True)
    print('Running experiment with seed', str(seed),
          'and parameters a =', str(a),
          'b_c =', str(b_c),
          'boolean =', str(boolean),
          'default =', default)
    with open(filename, 'w') as file:
        file.write('Default parameters:\n')
        file.write('seed: ' + str(seed) + '\n')
        file.write('results_dir: ' + str(results_dir) + '\n')

        file.write('Experiment parameters:\n')
        file.write('a: ' + str(a) + '\n')
        file.write('b-c: ' + str(b_c) + '\n')
        file.write('boolean: ' + str(boolean) + '\n')
        file.write('default: ' + default + '\n')


if __name__ == '__main__':
    # Leave unchanged
    run_experiment(experiment)
