import datetime
import json
import logging
import os
import socket
import time

import wandb
import yaml

import git
from git import InvalidGitRepositoryError

import random
from random import randint
try:
    import numpy as np
    import torch
except ImportError:
    pass


def save_args(results_dir, args, git_repo_path=None, seed=None, save_args_as_yaml=False, verbose=False):
    try:
        repo = git.Repo(git_repo_path, search_parent_directories=True)
        args['git_hash'] = repo.head.object.hexsha
        args['git_url'] = repo.remotes.origin.url
    except InvalidGitRepositoryError:
        args['git_hash'] = ''
        args['git_url'] = ''

    if save_args_as_yaml:
        filename = 'args.yaml' if seed is None else f'args-{seed}.yaml'
        with open(os.path.join(results_dir, filename), 'w') as f:
            yaml.dump(args, f, Dumper=yaml.Dumper)
    else:
        filename = 'args.json' if seed is None else f'args-{seed}.json'
        with open(os.path.join(results_dir, filename), 'w') as f:
            json.dump(args, f, indent=2)

    del args['git_hash']
    del args['git_url']

    if verbose:
        print('------------------------------------------------------------------------------------')
        print('--------> Experiment arguments')
        print(json.dumps(args, indent=2))
        print('------------------------------------------------------------------------------------')


def is_local():
    clusters_name_list = ['mn01', 'logc']
    hostname = socket.gethostname()
    for cluster_name in clusters_name_list:
        if cluster_name in hostname:
            return False
    return True


def start_wandb(
    wandb_enabled=False,
    wandb_entity='experiment_launcher',
    wandb_project='test_experiment_launcher',
    wandb_group=None,
    wandb_run_name=None,
    **kwargs
):

    if not wandb_enabled:
        return wandb.init(mode="disabled")

    init = {
        "entity": wandb_entity,
        "project": wandb_project,
        "group": wandb_group,
        "name": wandb_run_name,
        "reinit": True,
        "notes": datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
    }

    return wandb.init(**init)


def fix_random_seed(seed):
    random.seed(seed)

    try:
        np.random.seed(seed)
    except NameError:
        pass

    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
        # torch.backends.cudnn.benchmark = False
    except NameError:
        pass


def random_int_with_n_digits(n):
    range_start = 10 ** (n - 1)
    range_end = (10 ** n) - 1
    return randint(range_start, range_end)


def create_results_dir(results_dir=None, seed=None, **kwargs):
    assert results_dir is not None and seed is not None, "results_dir and seed must exist"
    results_dir_new = os.path.join(results_dir, str(seed))
    os.makedirs(results_dir_new, exist_ok=True)
    kwargs['results_dir'] = results_dir_new
    return kwargs
