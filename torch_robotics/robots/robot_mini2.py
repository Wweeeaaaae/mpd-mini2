from collections import OrderedDict

import einops
import numpy as np
import torch

from torch_robotics.robots.robot_base import RobotBase
from torch_robotics.torch_kinematics_tree.geometrics.skeleton import get_skeleton_from_model
from torch_robotics.torch_kinematics_tree.geometrics.utils import link_pos_from_link_tensor, link_quat_from_link_tensor, \
    link_rot_from_link_tensor
from torch_robotics.torch_kinematics_tree.models.robot_tree import convert_link_dict_to_tensor
from torch_robotics.torch_kinematics_tree.models.robots import DifferentiableMini2
from torch_robotics.torch_planning_objectives.fields.distance_fields import interpolate_points_v1
from torch_robotics.visualizers.plot_utils import plot_coordinate_frame


class RobotMini2(RobotBase):

    def __init__(self, use_collision_spheres=True, tensor_args=None, **kwargs):
        self.link_name_ee = '6_Link'
        self.diff_robot = DifferentiableMini2(
            device=tensor_args['device'],
            use_collision_spheres=use_collision_spheres,
        )

        self.jl_lower, self.jl_upper, _, _ = self.diff_robot.get_joint_limit_array()
        q_limits = torch.tensor(np.array([self.jl_lower, self.jl_upper]), **tensor_args)

        if use_collision_spheres:
            link_names_for_object_collision_checking = self.diff_robot.link_collision_names
            link_margins_for_object_collision_checking = torch.tensor(
                self.diff_robot.link_collision_margins,
                **tensor_args,
            ).view(-1, 1)
        else:
            link_names_for_object_collision_checking = [
                'base_link',
                '1_Link',
                '2_Link',
                '3_Link',
                '4_Link',
                '5_Link',
                '6_Link',
            ]
            # Fallback coarse radii estimated from Mini2 STL link cross-sections.
            link_margins_for_object_collision_checking = [
                0.07,
                0.075,
                0.09,
                0.065,
                0.06,
                0.06,
                0.045,
            ]
        link_idxs_for_object_collision_checking = [
            self.diff_robot._name_to_idx_map[link_name]
            for link_name in link_names_for_object_collision_checking
        ]

        link_names_pairs_for_self_collision_checking = OrderedDict({
            '4_Link': ['base_link'],
            '5_Link': ['base_link', '1_Link'],
            '6_Link': ['base_link', '1_Link', '2_Link'],
        })
        link_names_for_self_collision_checking = []
        for link_name, paired_links in link_names_pairs_for_self_collision_checking.items():
            link_names_for_self_collision_checking.append(link_name)
            link_names_for_self_collision_checking.extend(paired_links)
        link_names_for_self_collision_checking = sorted(set(link_names_for_self_collision_checking))
        link_idxs_for_self_collision_checking = [
            self.diff_robot._name_to_idx_map[link_name]
            for link_name in link_names_for_self_collision_checking
        ]

        super().__init__(
            name='RobotMini2',
            q_limits=q_limits,
            link_names_for_object_collision_checking=link_names_for_object_collision_checking,
            link_margins_for_object_collision_checking=link_margins_for_object_collision_checking,
            link_idxs_for_object_collision_checking=link_idxs_for_object_collision_checking,
            num_interpolated_points_for_object_collision_checking=len(link_names_for_object_collision_checking) * 1,
            link_names_for_self_collision_checking=link_names_for_self_collision_checking,
            link_names_pairs_for_self_collision_checking=link_names_pairs_for_self_collision_checking,
            link_idxs_for_self_collision_checking=link_idxs_for_self_collision_checking,
            num_interpolated_points_for_self_collision_checking=21,
            self_collision_margin_robot=0.06,
            use_collision_spheres=use_collision_spheres,
            tensor_args=tensor_args,
            **kwargs
        )

    def fk_map_collision_impl(self, q, **kwargs):
        q_orig_shape = q.shape
        if len(q_orig_shape) == 3:
            batch_size, horizon, q_dim = q_orig_shape
            q = einops.rearrange(q, 'b h d -> (b h) d')
        elif len(q_orig_shape) == 2:
            batch_size = q.shape[0]
            horizon = 1
        else:
            raise NotImplementedError

        link_pose_dict = self.diff_robot.compute_forward_kinematics_all_links(q, return_dict=True)
        link_tensor = convert_link_dict_to_tensor(link_pose_dict, self.diff_robot.get_link_names())
        link_pos = link_pos_from_link_tensor(link_tensor)

        if len(q_orig_shape) == 3:
            link_pos = einops.rearrange(link_pos, '(b h) l c -> b h l c', b=batch_size, h=horizon)

        return link_pos

    def get_EE_pose(self, q):
        return self.diff_robot.compute_forward_kinematics_all_links(q, link_list=[self.link_name_ee])

    def get_EE_position(self, q):
        ee_pose = self.get_EE_pose(q)
        return link_pos_from_link_tensor(ee_pose)

    def get_EE_orientation(self, q, rotation_matrix=True):
        ee_pose = self.get_EE_pose(q)
        if rotation_matrix:
            return link_rot_from_link_tensor(ee_pose)
        return link_quat_from_link_tensor(ee_pose)

    def render(self, ax, q=None, color='blue', arrow_length=0.12, arrow_alpha=1.0, arrow_linewidth=2.0,
               draw_links_spheres=False, **kwargs):
        skeleton = get_skeleton_from_model(self.diff_robot, q, self.diff_robot.get_link_names())
        skeleton.draw_skeleton(ax=ax, color=color)

        fks_dict = self.diff_robot.compute_forward_kinematics_all_links(q.unsqueeze(0), return_dict=True)

        if draw_links_spheres:
            from torch_robotics.environments.primitives import MultiSphereField

            link_tensor = convert_link_dict_to_tensor(fks_dict, self.link_names_for_object_collision_checking)
            link_pos = link_pos_from_link_tensor(link_tensor)
            link_pos = interpolate_points_v1(link_pos, self.num_interpolated_points_for_object_collision_checking).squeeze(0)
            spheres = MultiSphereField(
                link_pos,
                self.link_margins_for_object_collision_checking_robot_tensor.view(-1, 1),
                tensor_args=self.tensor_args,
            )
            spheres.render(ax, color='red', cmap='Reds', **kwargs)

        plot_coordinate_frame(
            ax,
            fks_dict[self.link_name_ee],
            tensor_args=self.tensor_args,
            arrow_length=arrow_length,
            arrow_alpha=arrow_alpha,
            arrow_linewidth=arrow_linewidth,
        )

    def render_trajectories(self, ax, trajs=None, start_state=None, goal_state=None, colors=['gray'], **kwargs):
        if trajs is not None:
            trajs_pos = self.get_position(trajs)
            for traj, color in zip(trajs_pos, colors):
                for q in traj:
                    self.render(ax, q, color, **kwargs, arrow_length=0.08, arrow_alpha=0.5, arrow_linewidth=1.0)
        if start_state is not None:
            self.render(ax, start_state, color='green')
        if goal_state is not None:
            self.render(ax, goal_state, color='purple')
