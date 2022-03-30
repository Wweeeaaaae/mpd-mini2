import argparse
import datetime
import inspect
import os
from copy import copy, deepcopy
from distutils.util import strtobool
from importlib import import_module
from itertools import product

import numpy as np
from joblib import Parallel, delayed


class Launcher(object):
    """
    Creates and starts jobs with Joblib or SLURM.

    """

    def __init__(self, exp_name, python_file, n_exps, n_cores=1, memory_per_core=2000,
                 days=0, hours=24, minutes=0, seconds=0,
                 project_name=None, base_dir=None, joblib_n_jobs=None, conda_env=None, gres=None, partition=None,
                 begin=None, use_timestamp=False, max_seeds=10000):
        """
        Constructor.

        Args:
            exp_name (str): name of the experiment
            python_file (str): prefix of the python file that runs a single experiment
            n_exps (int): number of experiments
            n_cores (int): number of cpu cores
            memory_per_core (int): maximum memory per core (slurm will kill the job if this is reached)
            days (int): number of days the experiment can last (in slurm)
            hours (int): number of hours the experiment can last (in slurm)
            minutes (int): number of minutes the experiment can last (in slurm)
            seconds (int): number of seconds the experiment can last (in slurm)
            project_name (str): name of the project for slurm. This is important if you have
                different projects (e.g. in the hhlr cluster)
            base_dir (str): path to directory to save the results (in hhlr results are saved to /work/scratch/$USER)
            joblib_n_jobs (int or None): number of parallel jobs in Joblib
            conda_env (str): name of the conda environment to run the experiments in
            gres (str): request cluster resources. E.g. to add a GPU in the IAS cluster specify gres='gpu:rtx2080:1'
            partition (str): the partition to use in case of slurm execution. If None, no partition is specified.
            begin (str): start the slurm experiment at a given time (see --begin in slurm docs)
            use_timestamp (bool): add a timestamp to the experiment name
            max_seeds (int): interval [1, max_seeds-1] of random seeds to sample from

        """
        self._exp_name = exp_name
        self._python_file = python_file
        self._n_exps = n_exps
        self._n_cores = n_cores
        self._memory_per_core = memory_per_core
        self._duration = Launcher._to_duration(days, hours, minutes, seconds)
        self._project_name = project_name
        assert (joblib_n_jobs is None or joblib_n_jobs > 0), "joblib_n_jobs must be None or > 0"
        self._joblib_n_jobs = joblib_n_jobs
        self._conda_env = conda_env
        self._gres = gres
        self._partition = partition
        self._begin = begin

        self._experiment_list = list()

        if use_timestamp:
            self._exp_name += datetime.datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')

        base_dir = './logs' if base_dir is None else base_dir
        self._exp_dir_local = os.path.join(base_dir, self._exp_name)

        scratch_dir = os.path.join('/work', 'scratch', os.getenv('USER'))
        if os.path.isdir(scratch_dir):
            self._exp_dir_slurm = os.path.join(scratch_dir, self._exp_name)
        else:
            self._exp_dir_slurm = self._exp_dir_local

        if n_exps >= max_seeds:
            max_seeds = n_exps + 1
            print(f"max_seeds must be larger than the number of experiments. Setting max_seeds to {max_seeds}")
        self._max_seeds = max_seeds

    def add_experiment(self, **kwargs):
        self._experiment_list.append(deepcopy(kwargs))

    def run(self, local, test=False):
        if local:
            self._run_joblib(test)
        else:
            self._run_slurm(test)

        self._experiment_list = list()

    def generate_slurm(self):
        project_name_option = ''
        partition_option = ''
        begin_option = ''
        gres_option = ''

        if self._project_name:
            project_name_option = '#SBATCH -A ' + self._project_name + '\n'
        if self._partition:
            partition_option += f'#SBATCH -p {self._partition}\n'
        if self._begin:
            begin_option += f'#SBATCH --begin={self._begin}\n'
        if self._gres:
            print(self._gres)
            gres_option += '#SBATCH --gres=' + str(self._gres) + '\n'

        joblib_seed = ''
        if self._joblib_n_jobs is not None:
            joblib_seed = f"""\
# Joblib seed
aux=$(( $SLURM_ARRAY_TASK_ID + 1 ))
aux=$(( $aux * $3 ))
if (( $aux <= $2 ))
then
  JOBLIB_SEEDS=$3
else
  JOBLIB_SEEDS=$(( $2 % $3 ))
fi
"""
        execution_code = ''
        if self._conda_env:
            if os.path.exists('/home/{}/miniconda3'.format(os.getenv('USER'))):
                execution_code += f'eval \"$(/home/{os.getenv("USER")}/miniconda3/bin/conda shell.bash hook)\"\n'
            elif os.path.exists(f'/home/{os.getenv("USER")}/anaconda3'):
                execution_code += f'eval \"$(/home/{os.getenv("USER")}/anaconda/bin/conda shell.bash hook)\"\n'
            else:
                raise Exception('You do not have a /home/USER/miniconda3 or /home/USER/anaconda3 directories')
            execution_code += f'conda activate {self._conda_env}\n\n'
            execution_code += f'python {self._python_file}.py \\'
        else:
            execution_code += f'python3  {self._python_file}.py \\'

        experiment_args = '\t\t'
        if self._joblib_n_jobs is not None:
            experiment_args += r'${@:4}'
        else:
            experiment_args += r'${@:2}'
        experiment_args += ' \\'

        result_dir_code = '\t\t--results_dir $1'

        joblib_code = ''
        if self._joblib_n_jobs is not None:
            joblib_code = f'\\\n\t\t--joblib_n_jobs $3  '
            N_EXPS = self._n_exps
            N_JOBS = self._joblib_n_jobs
            if N_EXPS < N_JOBS:
                joblib_code += f'--joblib_n_seeds $2 \n'
            elif N_EXPS % N_JOBS == 0:
                joblib_code += f'--joblib_n_seeds $3 \n'
            elif N_EXPS % N_JOBS != 0:
                joblib_code += '--joblib_n_seeds ${JOBLIB_SEEDS} \n'
            else:
                raise NotImplementedError

        code = f"""\
#!/usr/bin/env bash

###############################################################################
# SLURM Configurations

# Optional parameters
{project_name_option}{partition_option}{begin_option}{gres_option}
# Mandatory parameters
#SBATCH -J {self._exp_name}
#SBATCH -a 0-{compute_job_array_joblib(self._n_exps, self._joblib_n_jobs)}
#SBATCH -t {self._duration}
#SBATCH --ntasks 1
#SBATCH --cpus-per-task {self._n_cores}
#SBATCH --mem-per-cpu={self._memory_per_core}
#SBATCH -o {self._exp_dir_slurm}/%A_%a.out
#SBATCH -e {self._exp_dir_slurm}/%A_%a.err

###############################################################################
# Your PROGRAM call starts here
echo "Starting Job $SLURM_JOB_ID, Index $SLURM_ARRAY_TASK_ID"

{joblib_seed}
# Program specific arguments
{execution_code}
{experiment_args}
\t\t--seed $SLURM_ARRAY_TASK_ID \\
{result_dir_code} {joblib_code}
"""
        return code

    def save_slurm(self):
        code = self.generate_slurm()

        os.makedirs(self._exp_dir_slurm, exist_ok=True)
        script_name = "slurm_" + self._exp_name + ".sh"
        full_path = os.path.join(self._exp_dir_slurm, script_name)

        with open(full_path, "w") as file:
            file.write(code)

        return full_path

    def _run_slurm(self, test):
        full_path = self.save_slurm()

        for exp in self._experiment_list:
            exp_new_without_underscore = self.remove_last_underscores_dict(exp)
            command_line_arguments = self._convert_to_command_line(exp_new_without_underscore)
            results_dir = self._generate_results_dir(self._exp_dir_slurm, exp)

            command = "sbatch " + full_path + ' ' + results_dir
            if self._joblib_n_jobs is not None:
                command += ' ' + str(self._n_exps) + ' ' + str(self._joblib_n_jobs)
            command += ' ' + command_line_arguments

            if test:
                print(command)
            else:
                os.system(command)

    def _run_joblib(self, test):
        if not test:
            os.makedirs(self._exp_dir_local, exist_ok=True)

        module = import_module(self._python_file)
        experiment = module.experiment

        if test:
            for exp, i in product(self._experiment_list, range(self._n_exps)):
                results_dir = self._generate_results_dir(self._exp_dir_local, exp)
                params = str(exp).replace('{', '(').replace('}', '').replace(': ', '=').replace('\'', '')
                print('experiment' + params + 'seed=' + str(i) + ', results_dir=' + results_dir + ')')
        else:
            params_dict = get_experiment_default_params(experiment)

            Parallel(n_jobs=self._joblib_n_jobs)(delayed(experiment)(**params)
                                                 for params in self._generate_exp_params(params_dict))

    @staticmethod
    def _generate_results_dir(results_dir, exp):
        for key, value in exp.items():
            if key.endswith('_'):
                subfolder = key + '_' + str(value)
                results_dir = os.path.join(results_dir, subfolder)
        return results_dir

    def _generate_exp_params(self, params_dict):
        seeds = np.arange(self._n_exps)
        for exp, seed in product(self._experiment_list, seeds):
            exp_new_without_underscore = self.remove_last_underscores_dict(exp)
            params_dict.update(exp_new_without_underscore)
            params_dict['seed'] = int(seed)
            params_dict['results_dir'] = self._generate_results_dir(self._exp_dir_local, exp)
            yield params_dict

    @staticmethod
    def remove_last_underscores_dict(exp_dict):
        exp_dict_new = copy(exp_dict)
        for key, value in exp_dict.items():
            if key.endswith('__'):
                exp_dict_new[key[:-2]] = value
                del exp_dict_new[key]
        return exp_dict_new

    @staticmethod
    def _convert_to_command_line(exp):
        command_line = ''
        for key, value in exp.items():
            new_command = '--' + key + ' '

            new_command += str(value) + ' '

            command_line += new_command

        return command_line

    @staticmethod
    def _to_duration(days, hours, minutes, seconds):
        h = "0" + str(hours) if hours < 10 else str(hours)
        m = "0" + str(minutes) if minutes < 10 else str(minutes)
        s = "0" + str(seconds) if seconds < 10 else str(seconds)

        return str(days) + '-' + h + ":" + m + ":" + s

    @property
    def exp_name(self):
        return self._exp_name


def compute_job_array_joblib(n_exps, joblib_n_jobs):
    if joblib_n_jobs is None:
        return n_exps - 1
    else:
        if n_exps % joblib_n_jobs == 0:
            return n_exps // joblib_n_jobs - 1
        else:
            return n_exps // joblib_n_jobs


def get_experiment_default_params(func):
    signature = inspect.signature(func)
    defaults = {}
    for k, v in signature.parameters.items():
        if v.default is not inspect.Parameter.empty:
            defaults[k] = v.default
    return defaults


def translate_experiment_params_to_argparse(parser, func):
    annotation_to_argparse = {
        'str': str,
        'int': int,
        'float': float,
        'bool': bool
    }
    arg_experiments = parser.add_argument_group('Experiment')
    signature = inspect.signature(func)
    for k, v in signature.parameters.items():
        if k not in ['seed', 'results_dir']:
            if v.default is not inspect.Parameter.empty:
                if v.annotation.__name__ in annotation_to_argparse:
                    if v.annotation.__name__ == 'bool':
                        arg_experiments.add_argument(f"--{str(k)}", type=lambda x: bool(strtobool(x)),
                                                     nargs='?', const=v.default, default=v.default)
                    else:
                        arg_experiments.add_argument(f"--{str(k)}", type=annotation_to_argparse[v.annotation.__name__])
                else:
                    raise NotImplementedError(f'{v.annotation.__name__} not found in annotation_to_argparse.')
    return parser


def add_launcher_base_args(parser):
    arg_default = parser.add_argument_group('Default')
    arg_default.add_argument('--seed', type=int)
    arg_default.add_argument('--results_dir', type=str)
    arg_default.add_argument('--joblib_n_jobs', type=int)
    arg_default.add_argument('--joblib_n_seeds', type=int)
    return parser


def parse_args(func):
    parser = argparse.ArgumentParser()

    parser = translate_experiment_params_to_argparse(parser, func)

    parser = add_launcher_base_args(parser)
    parser.set_defaults(**get_experiment_default_params(func))
    args = parser.parse_args()
    return vars(args)


def run_experiment(func, args=None):
    if not args:
        args = parse_args(func)
    joblib_n_jobs = copy(args['joblib_n_jobs'])
    joblib_n_seeds = copy(args['joblib_n_seeds'])
    initial_seed = copy(args['seed'])
    if joblib_n_jobs is not None:
        initial_seed *= joblib_n_jobs
    del args['joblib_n_jobs']
    del args['joblib_n_seeds']

    def generate_joblib_seeds(params_dict):
        final_seed = initial_seed + (1 if joblib_n_seeds is None else joblib_n_seeds)
        seeds = np.arange(initial_seed, final_seed, dtype=int)
        for seed in seeds:
            params_dict['seed'] = int(seed)
            yield params_dict

    Parallel(n_jobs=joblib_n_jobs)(delayed(func)(**params)
                                   for params in generate_joblib_seeds(args))
