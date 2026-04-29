import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
import yaml

from mpd.utils.loading import load_params_from_yaml
from torch_robotics import environments, robots
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_utils.torch_utils import get_torch_device
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOGS_DIR = SCRIPT_DIR / 'logs'
# Manual Mini2 spot-check target. Change this folder number directly when you
# want to inspect another context without passing --data-dir.
num = 6


def find_mini2_result_by_num(logs_dir, result_num):
    candidates = []
    if not logs_dir.exists():
        return None

    pattern = (
        f'generate_trajectories_mini2_*/'
        f'env_id___EnvSpheres3DMini2/robot_id___RobotMini2/{result_num}/metadata.yaml'
    )
    for metadata_path in logs_dir.glob(pattern):
        result_dir = metadata_path.parent
        if (
            (result_dir / 'args.yaml').exists()
            and ((result_dir / 'trajs-free.pt').exists() or (result_dir / 'trajs-collision.pt').exists())
        ):
            candidates.append(result_dir)

    if not candidates:
        return None

    if len(candidates) > 1:
        candidates_str = '\n'.join(str(path) for path in sorted(candidates))
        raise RuntimeError(
            f'Multiple Mini2 result directories matched num={result_num}:\n{candidates_str}\n'
            f'Pass --data-dir explicitly to avoid ambiguity.'
        )

    return candidates[0]


def load_trajectories(data_dir, tensor_args, prefer='free'):
    trajs_free = torch.load(data_dir / 'trajs-free.pt', map_location=tensor_args['device']).to(**tensor_args)
    trajs_collision = torch.load(data_dir / 'trajs-collision.pt', map_location=tensor_args['device']).to(**tensor_args)

    if prefer == 'free':
        trajs = trajs_free if trajs_free.nelement() > 0 else trajs_collision
    elif prefer == 'collision':
        trajs = trajs_collision if trajs_collision.nelement() > 0 else trajs_free
    elif prefer == 'all':
        parts = [x for x in (trajs_free, trajs_collision) if x.nelement() > 0]
        trajs = torch.cat(parts, dim=0) if parts else torch.empty(0, **tensor_args)
    else:
        raise ValueError(f'Unknown trajectory preference: {prefer}')

    if trajs.nelement() == 0:
        raise RuntimeError(f'No trajectories found in {data_dir}')

    return trajs, trajs_free, trajs_collision


def save_or_show(fig, path=None, show=False):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)
        print(f'Saved: {path}')
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Visualize Mini2 generated planning trajectories.')
    parser.add_argument('--data-dir', type=Path, default=None,
                        help='Directory containing args.yaml, metadata.yaml, trajs-free.pt and trajs-collision.pt.')
    parser.add_argument('--logs-dir', type=Path, default=DEFAULT_LOGS_DIR,
                        help='Logs directory used with the manual num selection when --data-dir is omitted.')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                        help='Torch device for loading trajectories and collision checks.')
    parser.add_argument('--prefer', default='free', choices=['free', 'collision', 'all'],
                        help='Which trajectories to visualize.')
    parser.add_argument('--max-trajs', type=int, default=5,
                        help='Maximum number of trajectories to render.')
    parser.add_argument('--show', action='store_true',
                        help='Open matplotlib windows in addition to saving images/videos.')
    parser.add_argument('--no-video', action='store_true',
                        help='Skip mp4 animation generation.')
    parser.add_argument('--plot-full-trajs-in-video', action='store_true',
                        help='Overlay full trajectories in every video frame. Disabled by default so motion is easier to see.')
    parser.add_argument('--draw-collision-spheres', action='store_true',
                        help='Draw Mini2 collision spheres on rendered robot poses.')
    args_cli = parser.parse_args()

    if not args_cli.show:
        matplotlib.use('Agg')

    data_dir = args_cli.data_dir
    if data_dir is None:
        data_dir = find_mini2_result_by_num(args_cli.logs_dir, num)
        if data_dir is None:
            raise FileNotFoundError(
                f'No Mini2 result directory found under {args_cli.logs_dir} for num={num}. '
                f'Edit the num variable near the top of this script, or pass --data-dir explicitly.'
            )
    data_dir = data_dir.resolve()

    args = load_params_from_yaml(data_dir / 'args.yaml')
    metadata = load_params_from_yaml(data_dir / 'metadata.yaml')

    print(f'\n-------------- DATA DIR --------------')
    print(data_dir)
    print(f'\n-------------- METADATA --------------')
    print(yaml.dump(metadata))
    print(f'--------------------------------------\n')

    if args['env_id'] != 'EnvSpheres3DMini2' or args['robot_id'] != 'RobotMini2':
        raise ValueError(
            f'This script is for EnvSpheres3DMini2/RobotMini2, got '
            f"{args['env_id']}/{args['robot_id']}"
        )

    device = get_torch_device(args_cli.device)
    tensor_args = {'device': device, 'dtype': torch.float32}

    env_class = getattr(environments, args['env_id'])
    env = env_class(tensor_args=tensor_args)

    robot_class = getattr(robots, args['robot_id'])
    robot = robot_class(tensor_args=tensor_args)

    task = PlanningTask(
        env=env,
        robot=robot,
        obstacle_cutoff_margin=args['obstacle_cutoff_margin'],
        tensor_args=tensor_args,
    )

    trajs, trajs_free, trajs_collision = load_trajectories(data_dir, tensor_args, prefer=args_cli.prefer)
    trajs = trajs[:args_cli.max_trajs]

    print(f'Loaded trajectories: {trajs.shape}')
    print(f'Free trajectories: {tuple(trajs_free.shape)}')
    print(f'Collision trajectories: {tuple(trajs_collision.shape)}')
    print(f'fraction_free: {task.compute_fraction_free_trajs(trajs):.4f}')
    print(f'collision_intensity: {task.compute_collision_intensity_trajs(trajs):.4f}')
    print(f'success: {task.compute_success_free_trajs(trajs)}')

    planner_visualizer = PlanningVisualizer(task=task)

    pos_trajs = robot.get_position(trajs)
    start_state_pos = pos_trajs[0][0]
    goal_state_pos = pos_trajs[0][-1]

    fig_joint, _ = planner_visualizer.plot_joint_space_state_trajectories(
        trajs=trajs,
        pos_start_state=start_state_pos,
        pos_goal_state=goal_state_pos,
        vel_start_state=torch.zeros_like(start_state_pos),
        vel_goal_state=torch.zeros_like(goal_state_pos),
    )
    save_or_show(fig_joint, data_dir / 'mini2-joint-trajectories.png', show=args_cli.show)

    fig_robot, _ = planner_visualizer.render_robot_trajectories(
        trajs=trajs,
        start_state=start_state_pos,
        goal_state=goal_state_pos,
        render_planner=False,
        draw_links_spheres=args_cli.draw_collision_spheres,
    )
    save_or_show(fig_robot, data_dir / 'mini2-robot-trajectories.png', show=args_cli.show)

    if not args_cli.no_video:
        for traj_idx, traj in enumerate(trajs):
            video_path = data_dir / f'mini2-robot-traj-{traj_idx:03d}.mp4'
            planner_visualizer.animate_robot_trajectories(
                trajs=traj.unsqueeze(0),
                start_state=start_state_pos,
                goal_state=goal_state_pos,
                plot_trajs=args_cli.plot_full_trajs_in_video,
                video_filepath=str(video_path),
                n_frames=traj.shape[0],
                anim_time=args['duration'],
                draw_links_spheres=args_cli.draw_collision_spheres,
            )


if __name__ == '__main__':
    main()
