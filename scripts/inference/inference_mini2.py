import os
import pickle
from math import ceil
from pathlib import Path

try:
    from torch_robotics.isaac_gym_envs.motion_planning_envs import (
        Mini2MotionPlanningIsaacGymEnv,
        MotionPlanningController,
    )
    ISAAC_GYM_IMPORT_ERROR = None
except ImportError as exc:
    Mini2MotionPlanningIsaacGymEnv = None
    MotionPlanningController = None
    ISAAC_GYM_IMPORT_ERROR = exc

import einops
import matplotlib.pyplot as plt
import torch
from einops._torch_specific import allow_ops_in_compiled_graph

from experiment_launcher import single_experiment_yaml, run_experiment
from mp_baselines.planners.costs.cost_functions import CostCollision, CostComposite, CostGPTrajectory
from mpd.models import TemporalUnet, UNET_DIM_MULTS
from mpd.models.diffusion_models.guides import GuideManagerTrajectoriesWithVelocity
from mpd.models.diffusion_models.sample_functions import guide_gradient_steps, ddpm_sample_fn
from mpd.trainer import get_dataset, get_model
from mpd.utils.loading import load_params_from_yaml
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device, freeze_torch_model_params
from torch_robotics.trajectory.metrics import compute_smoothness, compute_path_length, compute_variance_waypoints
from torch_robotics.trajectory.utils import interpolate_traj_via_points
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer

allow_ops_in_compiled_graph()

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
TRAINED_MODELS_DIR = REPO_ROOT / 'data_trained_models'
DEFAULT_RESULTS_DIR = SCRIPT_DIR / 'logs' / 'mini2'
DEFAULT_MINI2_MODEL_DIR = (
    REPO_ROOT / 'scripts' / 'train_diffusion' / 'logs' / 'train_diffusion_2026-04-27_11-42-10' /
    'dataset_subdir___EnvSpheres3DMini2-RobotMini2' /
    'dataset_path___-home-yuancz-PycharmProjects-mpd-public-scripts-generate_data-logs-'
    'generate_trajectories_mini2_2026-04-23_16-55-24-env_id___EnvSpheres3DMini2-robot_id___RobotMini2' /
    'include_velocity___True' /
    'use_ema___True' /
    'variance_schedule___exponential' /
    'n_diffusion_steps___25' /
    'predict_epsilon___True' /
    'unet_dim_mults_option___0' /
    '0'
)


def resolve_model_dir(model_id, model_dir):
    if model_dir:
        return Path(model_dir).expanduser().resolve()

    candidates = [
        TRAINED_MODELS_DIR / model_id,
        DEFAULT_MINI2_MODEL_DIR,
    ]
    for candidate in candidates:
        if (candidate / 'args.yaml').exists() and (candidate / 'checkpoints').exists():
            return candidate
    return candidates[0]


@single_experiment_yaml
def experiment(
    ####################################################################################################################
    model_id: str = 'EnvSpheres3DMini2-RobotMini2',
    model_dir: str = '',
    checkpoint_path: str = '',

    planner_alg: str = 'mpd',
    use_extra_objects: bool = True,
    use_guide_on_extra_objects_only: bool = False,

    n_samples: int = 50,
    start_guide_steps_fraction: float = 0.25,
    n_guide_steps: int = 5,
    n_diffusion_steps_without_noise: int = 5,

    weight_grad_cost_collision: float = 1e-2,
    weight_grad_cost_smoothness: float = 1e-7,
    factor_num_interpolated_points_for_collision: float = 1.5,
    obstacle_cutoff_margin: float = 0.05,
    trajectory_duration: float = 5.0,

    ########################################################################
    device: str = 'cuda',
    debug: bool = True,
    render: bool = True,
    use_isaac_gym: bool = True,
    interpolate_for_isaac_replay: bool = False,
    num_interpolation_for_isaac_replay: int = 2,
    isaac_debug_contact_summary: bool = False,
    isaac_asset_root: str = '',

    ########################################################################
    seed: int = 8,
    results_dir: str = str(DEFAULT_RESULTS_DIR),

    ########################################################################
    **kwargs
):
    fix_random_seed(seed)

    if use_guide_on_extra_objects_only and not use_extra_objects:
        raise ValueError('use_guide_on_extra_objects_only=True requires use_extra_objects=True')

    device = get_torch_device(device)
    tensor_args = {'device': device, 'dtype': torch.float32}

    print('##########################################################################################################')
    print(f'Model -- {model_id}')
    print(f'Algorithm -- {planner_alg}')

    run_prior_only = False
    run_prior_then_guidance = False
    if planner_alg == 'mpd':
        pass
    elif planner_alg == 'diffusion_prior_then_guide':
        run_prior_then_guidance = True
    elif planner_alg == 'diffusion_prior':
        run_prior_only = True
    else:
        raise NotImplementedError

    model_dir_path = resolve_model_dir(model_id, model_dir)
    if not (model_dir_path / 'args.yaml').exists():
        raise FileNotFoundError(
            f'Model directory must contain args.yaml: {model_dir_path}\n'
            f'Pass model_dir=<path-to-training-run> to override the default.'
        )

    results_dir_path = Path(results_dir).expanduser().resolve()
    os.makedirs(results_dir_path, exist_ok=True)

    args = load_params_from_yaml(model_dir_path / 'args.yaml')

    train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        dataset_class='TrajectoryDataset',
        use_extra_objects=use_extra_objects,
        obstacle_cutoff_margin=obstacle_cutoff_margin,
        **args,
        tensor_args=tensor_args
    )
    dataset = train_subset.dataset
    n_support_points = dataset.n_support_points
    env = dataset.env
    robot = dataset.robot
    task = dataset.task

    dt = trajectory_duration / n_support_points
    robot.dt = dt

    diffusion_configs = dict(
        variance_schedule=args['variance_schedule'],
        n_diffusion_steps=args['n_diffusion_steps'],
        predict_epsilon=args['predict_epsilon'],
    )
    unet_configs = dict(
        state_dim=dataset.state_dim,
        n_support_points=dataset.n_support_points,
        unet_input_dim=args['unet_input_dim'],
        dim_mults=UNET_DIM_MULTS[args['unet_dim_mults_option']],
    )
    diffusion_model = get_model(
        model_class=args['diffusion_model_class'],
        model=TemporalUnet(**unet_configs),
        tensor_args=tensor_args,
        **diffusion_configs,
        **unet_configs
    )

    if checkpoint_path:
        checkpoint_path_resolved = Path(checkpoint_path).expanduser().resolve()
    else:
        checkpoint_name = 'ema_model_current_state_dict.pth' if args['use_ema'] else 'model_current_state_dict.pth'
        checkpoint_path_resolved = model_dir_path / 'checkpoints' / checkpoint_name

    diffusion_model.load_state_dict(
        torch.load(checkpoint_path_resolved, map_location=tensor_args['device'])
    )
    diffusion_model.eval()
    model = diffusion_model

    freeze_torch_model_params(model)
    model = torch.compile(model)
    model.warmup(horizon=n_support_points, device=device)

    n_tries = 100
    start_state_pos, goal_state_pos = None, None
    for _ in range(n_tries):
        q_free = task.random_coll_free_q(n_samples=2)
        start_state_pos = q_free[0]
        goal_state_pos = q_free[1]

        if torch.linalg.norm(start_state_pos - goal_state_pos) > dataset.threshold_start_goal_pos:
            break

    if start_state_pos is None or goal_state_pos is None:
        raise ValueError(
            f'No collision free configuration was found\n'
            f'start_state_pos: {start_state_pos}\n'
            f'goal_state_pos:  {goal_state_pos}\n'
        )

    print(f'start_state_pos: {start_state_pos}')
    print(f'goal_state_pos: {goal_state_pos}')

    hard_conds = dataset.get_hard_conditions(torch.vstack((start_state_pos, goal_state_pos)), normalize=True)
    context = None

    if use_guide_on_extra_objects_only:
        collision_fields = task.get_collision_fields_extra_objects()
    else:
        collision_fields = task.get_collision_fields()

    cost_collision_l = []
    weights_grad_cost_l = []
    for collision_field in collision_fields:
        cost_collision_l.append(
            CostCollision(
                robot,
                n_support_points,
                field=collision_field,
                sigma_coll=1.0,
                tensor_args=tensor_args
            )
        )
        weights_grad_cost_l.append(weight_grad_cost_collision)

    cost_smoothness_l = [
        CostGPTrajectory(
            robot,
            n_support_points,
            dt,
            sigma_gp=1.0,
            tensor_args=tensor_args
        )
    ]
    weights_grad_cost_l.append(weight_grad_cost_smoothness)

    cost_func_list = [*cost_collision_l, *cost_smoothness_l]
    cost_composite = CostComposite(
        robot,
        n_support_points,
        cost_func_list,
        weights_cost_l=weights_grad_cost_l,
        tensor_args=tensor_args
    )

    guide = GuideManagerTrajectoriesWithVelocity(
        dataset,
        cost_composite,
        clip_grad=True,
        interpolate_trajectories_for_collision=True,
        num_interpolated_points_for_collision=ceil(n_support_points * factor_num_interpolated_points_for_collision),
        tensor_args=tensor_args,
    )

    t_start_guide = ceil(start_guide_steps_fraction * model.n_diffusion_steps)
    sample_fn_kwargs = dict(
        guide=None if run_prior_then_guidance or run_prior_only else guide,
        n_guide_steps=n_guide_steps,
        t_start_guide=t_start_guide,
        noise_std_extra_schedule_fn=lambda x: 0.5,
    )

    with TimerCUDA() as timer_model_sampling:
        trajs_normalized_iters = model.run_inference(
            context,
            hard_conds,
            n_samples=n_samples,
            horizon=n_support_points,
            return_chain=True,
            sample_fn=ddpm_sample_fn,
            **sample_fn_kwargs,
            n_diffusion_steps_without_noise=n_diffusion_steps_without_noise,
        )
    print(f't_model_sampling: {timer_model_sampling.elapsed:.3f} sec')
    t_total = timer_model_sampling.elapsed

    if run_prior_then_guidance:
        n_post_diffusion_guide_steps = (t_start_guide + n_diffusion_steps_without_noise) * n_guide_steps
        with TimerCUDA() as timer_post_model_sample_guide:
            trajs = trajs_normalized_iters[-1]
            trajs_post_diff_l = []
            for _ in range(n_post_diffusion_guide_steps):
                trajs = guide_gradient_steps(
                    trajs,
                    hard_conds=hard_conds,
                    guide=guide,
                    n_guide_steps=1,
                    unnormalize_data=False,
                )
                trajs_post_diff_l.append(trajs)

            chain = torch.stack(trajs_post_diff_l, dim=1)
            chain = einops.rearrange(chain, 'b post_diff_guide_steps h d -> post_diff_guide_steps b h d')
            trajs_normalized_iters = torch.cat((trajs_normalized_iters, chain))
        print(f't_post_diffusion_guide: {timer_post_model_sample_guide.elapsed:.3f} sec')
        t_total = timer_model_sampling.elapsed + timer_post_model_sample_guide.elapsed

    trajs_iters = dataset.unnormalize_trajectories(trajs_normalized_iters)
    trajs_final = trajs_iters[-1]
    trajs_final_coll, trajs_final_coll_idxs, trajs_final_free, trajs_final_free_idxs, _ = task.get_trajs_collision_and_free(
        trajs_final, return_indices=True
    )

    print('\n----------------METRICS----------------')
    print(f't_total: {t_total:.3f} sec')

    success_free_trajs = task.compute_success_free_trajs(trajs_final)
    fraction_free_trajs = task.compute_fraction_free_trajs(trajs_final)
    collision_intensity_trajs = task.compute_collision_intensity_trajs(trajs_final)

    print(f'success: {success_free_trajs}')
    print(f'percentage free trajs: {fraction_free_trajs * 100:.2f}')
    print(f'percentage collision intensity: {collision_intensity_trajs * 100:.2f}')

    traj_final_free_best = None
    idx_best_traj = None
    cost_best_free_traj = None
    cost_smoothness = None
    cost_path_length = None
    cost_all = None
    variance_waypoint_trajs_final_free = None
    if trajs_final_free is not None:
        cost_smoothness = compute_smoothness(trajs_final_free, robot)
        print(f'cost smoothness: {cost_smoothness.mean():.4f}, {cost_smoothness.std():.4f}')

        cost_path_length = compute_path_length(trajs_final_free, robot)
        print(f'cost path length: {cost_path_length.mean():.4f}, {cost_path_length.std():.4f}')

        cost_all = cost_path_length + cost_smoothness
        idx_best_traj = torch.argmin(cost_all).item()
        traj_final_free_best = trajs_final_free[idx_best_traj]
        cost_best_free_traj = torch.min(cost_all).item()
        print(f'cost best: {cost_best_free_traj:.3f}')

        variance_waypoint_trajs_final_free = compute_variance_waypoints(trajs_final_free, robot)
        print(f'variance waypoint: {variance_waypoint_trajs_final_free:.4f}')

    print('\n--------------------------------------\n')

    results_data_dict = {
        'trajs_iters': trajs_iters,
        'trajs_final_coll': trajs_final_coll,
        'trajs_final_coll_idxs': trajs_final_coll_idxs,
        'trajs_final_free': trajs_final_free,
        'trajs_final_free_idxs': trajs_final_free_idxs,
        'success_free_trajs': success_free_trajs,
        'fraction_free_trajs': fraction_free_trajs,
        'collision_intensity_trajs': collision_intensity_trajs,
        'idx_best_traj': idx_best_traj,
        'traj_final_free_best': traj_final_free_best,
        'cost_best_free_traj': cost_best_free_traj,
        'cost_path_length_trajs_final_free': cost_path_length,
        'cost_smoothness_trajs_final_free': cost_smoothness,
        'cost_all_trajs_final_free': cost_all,
        'variance_waypoint_trajs_final_free': variance_waypoint_trajs_final_free,
        't_total': t_total,
        'model_dir': str(model_dir_path),
        'checkpoint_path': str(checkpoint_path_resolved),
    }
    with open(results_dir_path / 'results_data_dict.pickle', 'wb') as handle:
        pickle.dump(results_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if render:
        planner_visualizer = PlanningVisualizer(task=task)
        base_file_name = Path(os.path.basename(__file__)).stem
        pos_trajs_iters = robot.get_position(trajs_iters)

        planner_visualizer.animate_opt_iters_joint_space_state(
            trajs=trajs_iters,
            pos_start_state=start_state_pos,
            pos_goal_state=goal_state_pos,
            vel_start_state=torch.zeros_like(start_state_pos),
            vel_goal_state=torch.zeros_like(goal_state_pos),
            traj_best=traj_final_free_best,
            video_filepath=str(results_dir_path / f'{base_file_name}-joint-space-opt-iters.mp4'),
            n_frames=max((2, len(trajs_iters))),
            anim_time=5,
        )

        if use_isaac_gym:
            if ISAAC_GYM_IMPORT_ERROR is not None:
                raise ImportError(
                    'use_isaac_gym=True but Isaac Gym could not be imported. '
                    'Set use_isaac_gym=False to skip Isaac rendering.'
                ) from ISAAC_GYM_IMPORT_ERROR

            n_first_steps = 10
            n_last_steps = 10
            trajs_for_isaac = trajs_final_free if trajs_final_free is not None and len(trajs_final_free) > 0 else trajs_final
            trajs_pos = robot.get_position(trajs_for_isaac).movedim(1, 0)
            if interpolate_for_isaac_replay:
                print(f'Isaac replay interpolation enabled: num_interpolation={num_interpolation_for_isaac_replay}')
                trajs_pos = interpolate_traj_via_points(
                    trajs_pos.movedim(0, 1),
                    num_interpolation=num_interpolation_for_isaac_replay,
                ).movedim(1, 0)
            else:
                print('Isaac replay interpolation disabled: using original support points directly')

            isaac_env_kwargs = dict(
                controller_type='position',
                num_envs=trajs_pos.shape[1],
                all_robots_in_one_env=True,
                color_robots=False,
                show_goal_configuration=True,
                sync_with_real_time=True,
                show_collision_spheres=False,
                debug_contact_summary=isaac_debug_contact_summary,
                dt=dt,
                **results_data_dict,
            )
            if isaac_asset_root:
                isaac_env_kwargs['asset_root'] = isaac_asset_root

            motion_planning_isaac_env = Mini2MotionPlanningIsaacGymEnv(
                env,
                robot,
                task,
                **isaac_env_kwargs,
            )

            motion_planning_controller = MotionPlanningController(motion_planning_isaac_env)
            motion_planning_controller.run_trajectories(
                trajs_pos,
                start_states_joint_pos=trajs_pos[0],
                goal_state_joint_pos=trajs_pos[-1][0],
                n_first_steps=n_first_steps,
                n_last_steps=n_last_steps,
                visualize=True,
                render_viewer_camera=True,
                make_video=True,
                video_path=str(results_dir_path / f'{base_file_name}-isaac-controller-position.mp4'),
                make_gif=False,
            )
        else:
            planner_visualizer.animate_opt_iters_robots(
                trajs=pos_trajs_iters,
                start_state=start_state_pos,
                goal_state=goal_state_pos,
                traj_best=traj_final_free_best,
                video_filepath=str(results_dir_path / f'{base_file_name}-traj-opt-iters.mp4'),
                n_frames=max((2, len(trajs_iters))),
                anim_time=5,
            )

            planner_visualizer.animate_robot_trajectories(
                trajs=pos_trajs_iters[-1],
                start_state=start_state_pos,
                goal_state=goal_state_pos,
                plot_trajs=True,
                video_filepath=str(results_dir_path / f'{base_file_name}-robot-traj.mp4'),
                n_frames=pos_trajs_iters[-1].shape[1],
                anim_time=trajectory_duration,
            )

        plt.show()


if __name__ == '__main__':
    run_experiment(experiment)
