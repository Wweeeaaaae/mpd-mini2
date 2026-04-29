import torch
from matplotlib import pyplot as plt

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiSphereField
from torch_robotics.robots import RobotMini2
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes


class EnvSpheres3DMini2(EnvBase):

    def __init__(self, name='EnvSpheres3DMini2', tensor_args=None, **kwargs):
        spheres = MultiSphereField(
            torch.tensor([
                [-0.25,  0.25, 0.52],
                [-0.30, -0.24, 0.34],
                [-0.34,  0.08, 0.16],
                [ 0.34,  0.24, 0.16],
                [ 0.38,  0.18, 0.43],
                [ 0.42, -0.28, 0.24],
                [ 0.18, -0.32, 0.43],
                [ 0.28,  0.05, 0.58],
                [ 0.05, -0.30, 0.16],
                [ 0.05,  0.34, 0.30],
            ]),
            # 障碍物中心建议在 x/y [-0.45, 0.45]、z [0.05, 0.60] 内调节，以匹配 Mini2 约 0.6 m 的工作范围。
            torch.tensor([
                0.10,
                0.10,
                0.10,
                0.10,
                0.10,
                0.10,
                0.09,
                0.09,
                0.09,
                0.09,
            ]),
            # 障碍物半径建议在 [0.07, 0.12] 内调节；任务太简单就增大，难以采到有效配置就减小。
            tensor_args=tensor_args
        )

        obj_field = ObjectField([spheres], 'spheres_mini2')
        obj_list = [obj_field]

        super().__init__(
            name=name,
            limits=torch.tensor([[-0.75, -0.75, -0.15], [0.75, 0.75, 0.85]], **tensor_args),
            # 工作空间边界建议在 x/y +/-[0.65, 0.90]、z [-0.20, 0.90] 内调节；如果频繁触碰边界就适当放宽。
            obj_fixed_list=obj_list,
            tensor_args=tensor_args,
            **kwargs
        )

    def get_gpmp2_params(self, robot=None):
        params = dict(
            opt_iters=300,                  # 建议范围 [250, 500]。
            num_samples=64,                 # 建议范围 [32, 128]。
            sigma_start=1e-3,
            sigma_gp=5e-2,                  # 建议范围 [3e-2, 1e-1]；通常更小会让轨迹更平滑。
            sigma_goal_prior=1e-3,
            sigma_coll=8e-5,                # 建议范围 [5e-5, 2e-4]；修改后需要重新检查避障成功率。
            step_size=7.5e-1,               # 建议范围 [3e-1, 1.0]；优化不稳定时减小。
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.1,
            sigma_start_sample=1e-3,
            sigma_goal_sample=1e-3,
            solver_params={
                'delta': 1e-2,
                'trust_region': True,
                'sparse_computation': False,
                'sparse_computation_block_diag': False,
                'method': 'cholesky',
            },
            stop_criteria=0.05,             # 建议范围 [0.03, 0.10]。
        )
        if isinstance(robot, RobotMini2):
            return params
        else:
            raise NotImplementedError

    def get_rrt_connect_params(self, robot=None):
        params = dict(
            n_iters=10000,                  # 建议范围 [10000, 30000]；规划成功率低时增大。
            step_size=torch.pi / 36,        # 建议范围 [pi/45, pi/24]。
            n_radius=torch.pi / 5,          # 建议范围 [pi/6, pi/4]。
            n_pre_samples=30000,            # 建议范围 [20000, 50000]。
            max_time=180,                   # 建议范围 [120, 300] 秒；困难场景可增大。
        )
        if isinstance(robot, RobotMini2):
            return params
        else:
            raise NotImplementedError


if __name__ == '__main__':
    env = EnvSpheres3DMini2(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=DEFAULT_TENSOR_ARGS
    )
    fig, ax = create_fig_and_axes(env.dim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    env.render(ax)
    plt.show()

    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)
    env.render_grad_sdf(ax, fig)
    plt.show()
