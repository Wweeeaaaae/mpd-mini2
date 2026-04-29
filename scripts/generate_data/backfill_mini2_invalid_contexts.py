import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEPS_DIR = REPO_ROOT / 'deps'

for import_path in (
    REPO_ROOT,
    DEPS_DIR / 'experiment_launcher',
    DEPS_DIR / 'motion_planning_baselines',
    DEPS_DIR / 'torch_robotics',
    DEPS_DIR / 'storm',
):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)

from generate_trajectories import generate_collision_free_trajectories


ENV_ID = 'EnvSpheres3DMini2'
ROBOT_ID = 'RobotMini2'
NUM_TRAJECTORIES = 20
N_SUPPORT_POINTS = 64
DURATION = 5.0
THRESHOLD_START_GOAL_POS = float(6 * np.deg2rad(15))
OBSTACLE_CUTOFF_MARGIN = 0.05

# Use new sampling seeds only for RNG. Results are written back into the original
# invalid context folders, and args.yaml keeps the original folder seed.
BACKFILL_PLAN = [
    (10, 500),
    (20, 501),
    (28, 502),
    (53, 503),
    (56, 504),
    (86, 505),
    (105, 506),
    (117, 507),
    (120, 508),
    (127, 509),
    (135, 510),
    (197, 511),
    (215, 512),
    (217, 513),
    (283, 514),
    (289, 515),
]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_default_logs_root():
    script_dir = Path(__file__).resolve().parent
    return script_dir / 'logs' / 'generate_trajectories_mini2_2026-04-23_16-55-24' / f'env_id___{ENV_ID}' / f'robot_id___{ROBOT_ID}'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Backfill selected Mini2 free=0 contexts by rerunning planning with new sampling seeds '
                    'and writing outputs back into the original context folders.'
    )
    parser.add_argument(
        '--logs-root',
        type=Path,
        default=get_default_logs_root(),
        help='Directory containing the original Mini2 per-seed result folders.',
    )
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cpu',
        help='Torch device used for planning.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only print the 16-seed backfill mapping and target directories.',
    )
    return parser.parse_args()


def build_args_yaml(results_dir, original_seed, device):
    return {
        'debug': False,
        'device': device,
        'duration': DURATION,
        'env_id': ENV_ID,
        'n_support_points': N_SUPPORT_POINTS,
        'num_trajectories': NUM_TRAJECTORIES,
        'obstacle_cutoff_margin': OBSTACLE_CUTOFF_MARGIN,
        'results_dir': str(results_dir),
        'robot_id': ROBOT_ID,
        'seed': original_seed,
        'threshold_start_goal_pos': THRESHOLD_START_GOAL_POS,
    }


def write_yaml(path, payload):
    with open(path, 'w') as handle:
        yaml.dump(payload, handle, Dumper=yaml.Dumper)


def main():
    cli_args = parse_args()

    print('Mini2 backfill plan:')
    for target_seed, sampling_seed in BACKFILL_PLAN:
        print(f'  folder seed {target_seed} <- sampling seed {sampling_seed}')

    if cli_args.dry_run:
        return

    tensor_args = {'device': cli_args.device, 'dtype': torch.float32}

    for original_seed, sampling_seed in BACKFILL_PLAN:
        results_dir = cli_args.logs_root / str(original_seed)
        os.makedirs(results_dir, exist_ok=True)

        print('\n============================================================')
        print(f'Backfilling folder seed {original_seed} with sampling seed {sampling_seed}')
        print(f'Results dir: {results_dir}')

        # Use the new seed only for RNG. Persist the original folder seed in args.yaml.
        seed_everything(sampling_seed)

        args_yaml = build_args_yaml(results_dir, original_seed, cli_args.device)
        write_yaml(results_dir / 'args.yaml', args_yaml)

        num_coll, num_free = generate_collision_free_trajectories(
            ENV_ID,
            ROBOT_ID,
            NUM_TRAJECTORIES,
            str(results_dir),
            threshold_start_goal_pos=THRESHOLD_START_GOAL_POS,
            obstacle_cutoff_margin=OBSTACLE_CUTOFF_MARGIN,
            n_support_points=N_SUPPORT_POINTS,
            duration=DURATION,
            tensor_args=tensor_args,
            debug=False,
        )

        metadata = {
            'env_id': ENV_ID,
            'robot_id': ROBOT_ID,
            'num_trajectories': NUM_TRAJECTORIES,
            'num_trajectories_generated': num_coll + num_free,
            'num_trajectories_generated_coll': num_coll,
            'num_trajectories_generated_free': num_free,
        }
        write_yaml(results_dir / 'metadata.yaml', metadata)

        print(f'Finished folder seed {original_seed}: free={num_free}, invalid={num_coll}')


if __name__ == '__main__':
    main()
