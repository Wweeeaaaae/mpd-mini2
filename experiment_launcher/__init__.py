try:
    from .launcher import Launcher, get_default_params, run_experiment, add_launcher_base_args, save_args, \
        bool_local_cluster, single_experiment
except ImportError:
    pass

__version__ = '1.3'
