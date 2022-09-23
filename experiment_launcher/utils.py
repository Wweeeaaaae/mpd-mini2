import datetime
import json
import os
import socket

import wandb
import yaml

import git
from git import InvalidGitRepositoryError

import random
try:
    import numpy as np
    import torch
except ImportError:
    pass


def save_args(results_dir, args, git_repo_path=None, seed=None, save_args_yaml=False):
    try:
        repo = git.Repo(git_repo_path, search_parent_directories=True)
        args['git_hash'] = repo.head.object.hexsha
        args['git_url'] = repo.remotes.origin.url
    except InvalidGitRepositoryError:
        args['git_hash'] = ''
        args['git_url'] = ''

    if save_args_yaml:
        filename = 'args.yaml' if seed is None else f'args-{seed}.yaml'
        with open(os.path.join(results_dir, filename), 'w') as f:
            yaml.dump(args, f, Dumper=yaml.Dumper)
    else:
        filename = 'args.json' if seed is None else f'args-{seed}.json'
        with open(os.path.join(results_dir, filename), 'w') as f:
            json.dump(args, f, indent=2)

    del args['git_hash']
    del args['git_url']


def bool_local_cluster():
    hostname = socket.gethostname()
    return False if hostname == 'mn01' or 'logc' in hostname else True


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
