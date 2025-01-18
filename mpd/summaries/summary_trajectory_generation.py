import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from mpd.models import build_context
from mpd.summaries.summary_base import SummaryBase

#  使用扩散模型生成、评估和可视化机器人的运动轨迹
class SummaryTrajectoryGeneration(SummaryBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def summary_fn(self, train_step=None, model=None, datasubset=None, prefix='', debug=False, **kwargs):
        #  prefix：日志记录的前缀，用于区分不同的实验或记录
        dataset = datasubset.dataset

        # ------------------------------------------------------------------------------------
        #  从 dataset 随机选择一条轨迹
        trajectory_id = np.random.choice(datasubset.indices)
        task_id = dataset.map_trajectory_id_to_task_id[trajectory_id]

        data_normalized = dataset[trajectory_id]  #从数据集中获取与选定轨迹ID对应的归一化数据
        context = build_context(model, dataset, data_normalized)

        # ------------------------------------------------------------------------------------
        #  采用扩散/cvae 模型生成样本轨迹
        n_samples = 25  #生成25条新轨迹
        horizon = dataset.n_support_points
        hard_conds = data_normalized['hard_conds']

        trajs_normalized = model.run_inference(
            context, hard_conds,
            n_samples=n_samples, horizon=horizon,
            deterministic_steps=0  #确定性步骤的数量
        )

        #  对扩散模型的轨迹样本进行反归一化处理
        trajs = dataset.unnormalize(trajs_normalized, dataset.field_key_traj)

        # ------------------------------------------------------------------------------------
        #  记录轨迹的数据信息
        wandb.log({f'{prefix}percentage free trajs': dataset.task.compute_fraction_free_trajs(trajs)}, step=train_step)  #无碰撞轨迹的百分比
        wandb.log({f'{prefix}percentage collision intensity': dataset.task.compute_collision_intensity_trajs(trajs)}, step=train_step)  ##碰撞强度的百分比
        wandb.log({f'{prefix}success': dataset.task.compute_success_free_trajs(trajs)}, step=train_step)  ##成功且无碰撞的轨迹比例

        # ------------------------------------------------------------------------------------
        #  渲染

        #  渲染关节空间轨迹和机器人空间轨迹
        fig_joint_trajs_dataset, _, fig_robot_trajs_dataset, _ = dataset.render(
            task_id=task_id,
            render_joint_trajectories=True
        )

        #  获取扩散轨迹位置信息
        pos_trajs = dataset.robot.get_position(trajs)
        start_state_pos = pos_trajs[0][0]
        goal_state_pos = pos_trajs[0][-1]

        #  绘制关节空间轨迹
        fig_joint_trajs_diffusion, _ = dataset.planner_visualizer.plot_joint_space_state_trajectories(
            trajs=pos_trajs,
            pos_start_state=start_state_pos, pos_goal_state=goal_state_pos,
            vel_start_state=torch.zeros_like(start_state_pos), vel_goal_state=torch.zeros_like(goal_state_pos),
            linestyle='dashed'
        )

        #  绘制机器人空间轨迹
        fig_robot_trajs_diffusion = None
        # fig_robot_trajs_diffusion, _ = dataset.planner_visualizer.render_robot_trajectories(
        #     trajs=pos_trajs, start_state=start_state_pos, goal_state=goal_state_pos,
        #     linestyle='dashed'
        # )

        #  轨迹的可视化和记录
        if fig_joint_trajs_dataset is not None:
            wandb.log({f"{prefix}joint trajectories DATASET": wandb.Image(fig_joint_trajs_dataset)}, step=train_step)
        if fig_robot_trajs_dataset is not None:
            wandb.log({f"{prefix}robot trajectories DATASET": wandb.Image(fig_robot_trajs_dataset)}, step=train_step)

        if fig_joint_trajs_diffusion is not None:
            wandb.log({f"{prefix}joint trajectories DIFFUSION": wandb.Image(fig_joint_trajs_diffusion)}, step=train_step)
        if fig_robot_trajs_diffusion is not None:
            wandb.log({f"{prefix}robot trajectories DIFFUSION": wandb.Image(fig_robot_trajs_diffusion)}, step=train_step)

        if debug:
            plt.show()

        if fig_joint_trajs_dataset is not None:
            plt.close(fig_joint_trajs_dataset)
        if fig_robot_trajs_dataset is not None:
            plt.close(fig_robot_trajs_dataset)
        if fig_joint_trajs_diffusion is not None:
            plt.close(fig_joint_trajs_diffusion)
        if fig_robot_trajs_diffusion is not None:
            plt.close(fig_robot_trajs_diffusion)

