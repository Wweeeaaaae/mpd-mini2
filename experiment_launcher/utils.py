import json
import os
import socket

import git
from git import InvalidGitRepositoryError


def save_args(results_dir, args, git_repo_path=None, seed=None):
    try:
        repo = git.Repo(git_repo_path, search_parent_directories=True)
        args['git_hash'] = repo.head.object.hexsha
        args['git_url'] = repo.remotes.origin.url
    except InvalidGitRepositoryError:
        args['git_hash'] = ''
        args['git_url'] = ''

    filename = 'args.json' if seed is None else f'args-{seed}.json'
    with open(os.path.join(results_dir, filename), 'w') as f:
        json.dump(args, f, indent=2)

    del args['git_hash']
    del args['git_url']


def bool_local_cluster():
    hostname = socket.gethostname()
    return False if hostname == 'mn01' or 'logc' in hostname else True
