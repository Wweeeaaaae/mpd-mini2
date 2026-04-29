from torch_robotics.isaac_gym_envs.motion_planning_envs import PandaMotionPlanningIsaacGymEnv, MotionPlanningController

import os
import pickle
from math import ceil
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from experiment_launcher import single_experiment_yaml, run_experiment
from mp_baselines.planners.costs.cost_functions import CostCollision, CostComposite, CostGPTrajectory
from mpd.models import TemporalUnet, UNET_DIM_MULTS
from mpd.models.diffusion_models.guides import GuideManagerTrajectoriesWithVelocity
from mpd.models.diffusion_models.sample_functions import guide_gradient_steps, ddpm_sample_fn
from mpd.trainer import get_dataset, get_model
from mpd.utils.loading import load_params_from_yaml
from torch_robotics.robots import RobotPanda
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device, freeze_torch_model_params
from torch_robotics.trajectory.metrics import compute_smoothness, compute_path_length, compute_variance_waypoints
from torch_robotics.trajectory.utils import interpolate_traj_via_points
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer
#导入所需库
allow_ops_in_compiled_graph()
#einops的操作被允许在编译图中执行

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
TRAINED_MODELS_DIR = REPO_ROOT / 'data_trained_models'
DEFAULT_RESULTS_DIR = SCRIPT_DIR / 'logs' / 'panda'
#模型存储目录

@single_experiment_yaml  #装饰器
def experiment(
    ########################################################################################################################
    # Experiment configuration
    # model_id: str = 'EnvDense2D-RobotPointMass',
    # model_id: str = 'EnvNarrowPassageDense2D-RobotPointMass',
    # model_id: str = 'EnvSimple2D-RobotPointMass',
    model_id: str = 'EnvSpheres3D-RobotPanda',
    model_dir: str = '',
    checkpoint_path: str = '',

    # planner_alg: str = 'diffusion_prior',
    # planner_alg: str = 'diffusion_prior_then_guide',
    planner_alg: str = 'mpd',  #规划算法

    use_guide_on_extra_objects_only: bool = False,  #是否仅在额外对象使用引导

    n_samples: int = 50,

    start_guide_steps_fraction: float = 0.25,  #完成多少比例的扩散步骤之后开始引导阶段
    n_guide_steps: int = 5,
    n_diffusion_steps_without_noise: int = 5,  #无噪声扩散步骤

    weight_grad_cost_collision: float = 1e-2,
    weight_grad_cost_smoothness: float = 1e-7,  #碰撞和平滑性成本的梯度权重

    factor_num_interpolated_points_for_collision: float = 1.5,  #用于碰撞的插值点数量因子

    trajectory_duration: float = 5.0,  # currently fixed轨迹持续时间

    ########################################################################
    device: str = 'cuda',

    debug: bool = True,

    render: bool = True,  #渲染结果
    interpolate_for_isaac_replay: bool = False,
    num_interpolation_for_isaac_replay: int = 2,

    ########################################################################
    # MANDATORY
    seed: int = 30,
    results_dir: str = str(DEFAULT_RESULTS_DIR),  #结果目录scripts/inference/logs/panda/<seed>

    ########################################################################
    **kwargs
):
    ########################################################################################################################
    fix_random_seed(seed)  #固定随机种子，确保实验可重复性

    device = get_torch_device(device)
    tensor_args = {'device': device, 'dtype': torch.float32}

    ########################################################################################################################
    print(f'##########################################################################################################')
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
        raise NotImplementedError  #规划算法设置模式

    ########################################################################################################################
    model_dir_path = Path(model_dir).expanduser().resolve() if model_dir else (TRAINED_MODELS_DIR / model_id)
    results_dir_path = Path(results_dir).expanduser().resolve()
    results_dir_path.mkdir(parents=True, exist_ok=True)

    args = load_params_from_yaml(str(model_dir_path / "args.yaml"))  #加载模型参数

    ########################################################################################################################
    #获取训练和验证数据集，从中提取出环境、机器人和任务等相关信息
    train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        dataset_class='TrajectoryDataset',
        use_extra_objects=True,
        obstacle_cutoff_margin=0.05,  #碰撞检测边界值
        **args,  #解包模型参数data_trained_models/EnvSpheres3D-RobotPanda/args.yaml给get_dataset
        tensor_args=tensor_args
    )
    dataset = train_subset.dataset
    n_support_points = dataset.n_support_points
    env = dataset.env
    robot = dataset.robot
    task = dataset.task  #从train_subset中提取出实际的数据集对象（dataset）、环境（env）、机器人（robot）和任务（task）

    dt = trajectory_duration / n_support_points  #两个时间点之间的时间间隔

    # set robot's dt
    robot.dt = dt

    ########################################################################################################################
    # Load prior model
    #data_trained_models/EnvSpheres3D-RobotPanda/args.yaml
    #  扩散模型配置
    diffusion_configs = dict(
        variance_schedule=args['variance_schedule'],  #调度策略 exponential指数衰减
        n_diffusion_steps=args['n_diffusion_steps'],
        predict_epsilon=args['predict_epsilon'],  #模型是否需要预测噪声项
    )
    #  U-Net配置
    unet_configs = dict(
        state_dim=dataset.state_dim,  #UNet状态空间维度，支持点数，输入维度，维度乘数配置
        n_support_points=dataset.n_support_points,
        unet_input_dim=args['unet_input_dim'],
        dim_mults=UNET_DIM_MULTS[args['unet_dim_mults_option']],
    )
    #  加载已经预训练模型
    diffusion_model = get_model(
        model_class=args['diffusion_model_class'],
        model=TemporalUnet(**unet_configs),  #TemporalUnet 类是一个复杂的深度学习模型，专为处理时间序列数据设计。
        tensor_args=tensor_args,
        **diffusion_configs,
        **unet_configs
    )
    #  加载预训练模型权重参数（普通模型权重，EMA平滑模型权重）
    if checkpoint_path:
        checkpoint_path_resolved = Path(checkpoint_path).expanduser().resolve()
    else:
        checkpoint_name = 'ema_model_current_state_dict.pth' if args['use_ema'] else 'model_current_state_dict.pth'
        checkpoint_path_resolved = model_dir_path / 'checkpoints' / checkpoint_name

    diffusion_model.load_state_dict(
        torch.load(checkpoint_path_resolved, map_location=tensor_args['device'])
    )
    diffusion_model.eval()  #设定评估模式，给定相同输入，产生相同输出
    model = diffusion_model

    freeze_torch_model_params(model)  #冻结模型参数
    model = torch.compile(model)  #编译模型
    model.warmup(horizon=n_support_points, device=device)  #模型预热

    ########################################################################################################################
    #  随机生成初始和最终关节配置
    n_tries = 100  #最大尝试次数
    start_state_pos, goal_state_pos = None, None
    for _ in range(n_tries):   #循环尝试生成两个无碰撞关节配置
        q_free = task.random_coll_free_q(n_samples=2)
        start_state_pos = q_free[0]
        goal_state_pos = q_free[1]

        if torch.linalg.norm(start_state_pos - goal_state_pos) > dataset.threshold_start_goal_pos:  #计算起始位置和目标位置之间的距离
            break

    if start_state_pos is None or goal_state_pos is None:  #错误处理
        raise ValueError(f"No collision free configuration was found\n"
                         f"start_state_pos: {start_state_pos}\n"
                         f"goal_state_pos:  {goal_state_pos}\n")

    print(f'start_state_pos: {start_state_pos}')
    print(f'goal_state_pos: {goal_state_pos}')

    ########################################################################################################################
    #  执行运动规划推理

    ########
    #  规范化起始和目标位置，将起始位置和目标位置堆叠起来并规范化
    hard_conds = dataset.get_hard_conditions(torch.vstack((start_state_pos, goal_state_pos)), normalize=True)
    context = None

    ########
    #  设定规划成本

    #  获取碰撞字段
    cost_collision_l = []
    weights_grad_cost_l = []  # 碰撞成本梯度权重，为便于指导，weight_cost_l 是梯度乘数（梯度剪切后的值）
    if use_guide_on_extra_objects_only:  #是否仅在额外对象上使用引导
        collision_fields = task.get_collision_fields_extra_objects()
    else:
        collision_fields = task.get_collision_fields()

    #  为每个碰撞字段创建一个 CostCollision 对象
    for collision_field in collision_fields:
        cost_collision_l.append(
            CostCollision(
                robot, n_support_points,  #机器人的物理和运动学属性,轨迹中的支持点数量
                field=collision_field,  #当前的碰撞字段
                sigma_coll=1.0,  #碰撞成本函数中参数，用于控制碰撞成本的敏感度
                tensor_args=tensor_args  #包含了设备和数据类型的参数
            )
        )
        weights_grad_cost_l.append(weight_grad_cost_collision)  #碰撞成本的梯度权重，定值

    #  计算轨迹平滑度的成本函数
    cost_smoothness_l = [
        CostGPTrajectory(
            robot, n_support_points, dt, sigma_gp=1.0,  #时间间隔，高斯过程中参数控制平滑度成本的敏感度，
            tensor_args=tensor_args
        )
    ]
    weights_grad_cost_l.append(weight_grad_cost_smoothness)  #平滑度成本的梯度权重 ，定值

    ####### 总成本函数构成
    cost_func_list = [
        *cost_collision_l,
        *cost_smoothness_l
    ]

    cost_composite = CostComposite(  #CostComposite对象包括
        robot, n_support_points, cost_func_list,
        weights_cost_l=weights_grad_cost_l,  #所有成本函数梯度权重的列表
        tensor_args=tensor_args  #设备和数据类型的参数
    )

    ########
    #  引导管理器，作用是在轨迹优化过程中考虑速度信息，保证轨迹的平滑性和动态可行性
    guide = GuideManagerTrajectoriesWithVelocity(
            dataset,  #数据集对象
            cost_composite,  #复合成本函数对象（碰撞成本，轨迹平滑成本）
            clip_grad=True,  #是否在优化过程中对梯度进行裁剪
            interpolate_trajectories_for_collision=True,  #计算碰撞成本时对轨迹进行插值
            num_interpolated_points_for_collision=ceil(n_support_points * factor_num_interpolated_points_for_collision),  #计算插值点的数量
            tensor_args=tensor_args,
    )

    t_start_guide = ceil(start_guide_steps_fraction * model.n_diffusion_steps)  #计算开始引导阶段的时间
    sample_fn_kwargs = dict(  #控制运动规划推理过程中的样本生成
        guide=None if run_prior_then_guidance or run_prior_only else guide,
        n_guide_steps=n_guide_steps,
        t_start_guide=t_start_guide,
        noise_std_extra_schedule_fn=lambda x: 0.5,  #在额外的噪声计划中设置噪声的标准差
    )

    ########
    #  使用扩散模型或条件变分自编码器（CVAE）进行轨迹样本生成
    with TimerCUDA() as timer_model_sampling:  #创建一个 TimerCUDA 对象来测量模型采样过程中的时间消耗
        trajs_normalized_iters = model.run_inference(  #！run_inference 来生成轨迹样本
            context, hard_conds,  #上下文信息，硬性条件（起始和目标状态等）
            n_samples=n_samples, horizon=n_support_points,
            return_chain=True,  #是否返回整个采样过程的链
            sample_fn=ddpm_sample_fn,  #采样函数
            **sample_fn_kwargs,
            n_diffusion_steps_without_noise=n_diffusion_steps_without_noise,  #无噪声扩散步骤的数量
            # ddim=True
        )
    print(f't_model_sampling: {timer_model_sampling.elapsed:.3f} sec')
    t_total = timer_model_sampling.elapsed  #记录总时间

    ########
    #  在无扩散的情况下运行额外的引导步骤，来优化样本
    if run_prior_then_guidance:
        n_post_diffusion_guide_steps = (t_start_guide + n_diffusion_steps_without_noise) * n_guide_steps  #计算引导阶段步骤数量
        with TimerCUDA() as timer_post_model_sample_guide:
            trajs = trajs_normalized_iters[-1]
            trajs_post_diff_l = []
            for i in range(n_post_diffusion_guide_steps):  #执行引导步骤
                trajs = guide_gradient_steps(
                    trajs,
                    hard_conds=hard_conds,
                    guide=guide,
                    n_guide_steps=1,
                    unnormalize_data=False
                )
                trajs_post_diff_l.append(trajs)  #更新后轨迹样本添加

            chain = torch.stack(trajs_post_diff_l, dim=1)  #将引导步骤的轨迹堆叠成一个张量
            chain = einops.rearrange(chain, 'b post_diff_guide_steps h d -> post_diff_guide_steps b h d')  #重新排列轨迹张量
            trajs_normalized_iters = torch.cat((trajs_normalized_iters, chain))  #轨迹与扩散过程轨迹拼接
        print(f't_post_diffusion_guide: {timer_post_model_sample_guide.elapsed:.3f} sec')
        t_total = timer_model_sampling.elapsed + timer_post_model_sample_guide.elapsed

    #  将生成的轨迹样本反规范化，重新评估轨迹的碰撞和自由状态
    trajs_iters = dataset.unnormalize_trajectories(trajs_normalized_iters)

    trajs_final = trajs_iters[-1]
    trajs_final_coll, trajs_final_coll_idxs, trajs_final_free, trajs_final_free_idxs, _ = task.get_trajs_collision_and_free(trajs_final, return_indices=True)

    ########################################################################################################################
    #  评估运动规划的指标
    print(f'\n----------------METRICS----------------')
    print(f't_total: {t_total:.3f} sec')

    success_free_trajs = task.compute_success_free_trajs(trajs_final)  #计算成功无碰撞轨迹的数量
    fraction_free_trajs = task.compute_fraction_free_trajs(trajs_final)  #计算无碰撞轨迹的比例
    collision_intensity_trajs = task.compute_collision_intensity_trajs(trajs_final)  #计算轨迹的碰撞强度（waypoint碰撞比例）

    print(f'success: {success_free_trajs}')
    print(f'percentage free trajs: {fraction_free_trajs*100:.2f}')
    print(f'percentage collision intensity: {collision_intensity_trajs*100:.2f}')

    #  通过平滑度，轨迹长度指标评估最优轨迹
    traj_final_free_best = None
    idx_best_traj = None
    cost_best_free_traj = None
    cost_smoothness = None
    cost_path_length = None
    cost_all = None
    variance_waypoint_trajs_final_free = None
    if trajs_final_free is not None:
        cost_smoothness = compute_smoothness(trajs_final_free, robot)  #平滑度
        print(f'cost smoothness: {cost_smoothness.mean():.4f}, {cost_smoothness.std():.4f}')

        cost_path_length = compute_path_length(trajs_final_free, robot)  #轨迹长度
        print(f'cost path length: {cost_path_length.mean():.4f}, {cost_path_length.std():.4f}')

        #  评估出最佳轨迹
        cost_all = cost_path_length + cost_smoothness
        idx_best_traj = torch.argmin(cost_all).item()  #找出最佳轨迹索引
        traj_final_free_best = trajs_final_free[idx_best_traj]
        cost_best_free_traj = torch.min(cost_all).item()  #获取最佳轨迹的总成本
        print(f'cost best: {cost_best_free_traj:.3f}')

        # 计算航点（waypoint）方差
        variance_waypoint_trajs_final_free = compute_variance_waypoints(trajs_final_free, robot)
        print(f'variance waypoint: {variance_waypoint_trajs_final_free:.4f}')

    print(f'\n--------------------------------------\n')

    ########################################################################################################################
    #  保存运动规划数据
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
        't_total': t_total
    }
    with open(results_dir_path / 'results_data_dict.pickle', 'wb') as handle:
        pickle.dump(results_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ########################################################################################################################
    #  渲染结果
    if render:
        # Render
        planner_visualizer = PlanningVisualizer(
            task=task,
        )

        base_file_name = Path(os.path.basename(__file__)).stem  #获取当前文件的名称

        pos_trajs_iters = robot.get_position(trajs_iters)  #提取轨迹位置信息

        planner_visualizer.animate_opt_iters_joint_space_state(  #创建关节空间优化迭代的动画
            trajs=trajs_iters,
            pos_start_state=start_state_pos, pos_goal_state=goal_state_pos,
            vel_start_state=torch.zeros_like(start_state_pos), vel_goal_state=torch.zeros_like(goal_state_pos),
            traj_best=traj_final_free_best,
            video_filepath=str(results_dir_path / f'{base_file_name}-joint-space-opt-iters.mp4'),
            n_frames=max((2, len(trajs_iters))),
            anim_time=5
        )

        if isinstance(robot, RobotPanda):  #检查 robot 是否是 RobotPanda 类的一个实例
            # visualize in Isaac Gym
            # POSITION CONTROL
            # add initial positions for better visualization
            n_first_steps = 10  #轨迹开始的前 10 个时间步的位置将被特别标记，增强可视化效果
            n_last_steps = 10  #轨迹结束的最后 10 个时间步的位置将被特别标记

            trajs_pos = robot.get_position(trajs_final_free).movedim(1, 0)  #提取最后迭代位置信息，速度信息
            trajs_vel = robot.get_velocity(trajs_final_free).movedim(1, 0)

            if interpolate_for_isaac_replay:
                print(f'Isaac replay interpolation enabled: num_interpolation={num_interpolation_for_isaac_replay}')
                trajs_pos = interpolate_traj_via_points(
                    trajs_pos.movedim(0, 1),
                    num_interpolation=num_interpolation_for_isaac_replay,
                ).movedim(1, 0)  #轨迹插值
            else:
                print('Isaac replay interpolation disabled: using original support points directly')

            #  设置IsaacGym仿真环境
            motion_planning_isaac_env = PandaMotionPlanningIsaacGymEnv(
                env, robot, task,
                asset_root=str(REPO_ROOT / "deps" / "isaacgym" / "assets"),  #指定 Isaac Gym 资产（如机器人模型）的根目录
                controller_type='position',  #设置控制器的类型，这里使用位置控制器。
                num_envs=trajs_pos.shape[1],  #设置仿真环境中的环境数量，由批次大小设计
                all_robots_in_one_env=True,  #所有机器人是否在同一个环境中
                color_robots=False,
                show_goal_configuration=True,  #是否显示目标配置
                sync_with_real_time=True,  #仿真是否与实时同步
                show_collision_spheres=False,  #是否显示碰撞球体
                dt=dt,
                **results_data_dict,
                # show_collision_spheres=True
            )

            #  在仿真环境中生成轨迹
            motion_planning_controller = MotionPlanningController(motion_planning_isaac_env)
            motion_planning_controller.run_trajectories(
                trajs_pos,
                start_states_joint_pos=trajs_pos[0], goal_state_joint_pos=trajs_pos[-1][0],
                n_first_steps=n_first_steps,
                n_last_steps=n_last_steps,
                visualize=True,
                render_viewer_camera=True,
                make_video=True,
                video_path=str(results_dir_path / f'{base_file_name}-isaac-controller-position.mp4'),
                make_gif=False
            )
        else:
            # visualize in the planning environment
            #  轨迹迭代动画
            planner_visualizer.animate_opt_iters_robots(
                trajs=pos_trajs_iters, start_state=start_state_pos, goal_state=goal_state_pos,
                traj_best=traj_final_free_best,
                video_filepath=str(results_dir_path / f'{base_file_name}-traj-opt-iters.mp4'),
                n_frames=max((2, len(trajs_iters))),
                anim_time=5
            )

            #  最后一个优化迭代中的轨迹动画
            planner_visualizer.animate_robot_trajectories(
                trajs=pos_trajs_iters[-1], start_state=start_state_pos, goal_state=goal_state_pos,
                plot_trajs=True,
                video_filepath=str(results_dir_path / f'{base_file_name}-robot-traj.mp4'),
                # n_frames=max((2, pos_trajs_iters[-1].shape[1]//10)),
                n_frames=pos_trajs_iters[-1].shape[1],
                anim_time=trajectory_duration
            )

        plt.show()


if __name__ == '__main__':
    # Leave unchanged
    run_experiment(experiment)
