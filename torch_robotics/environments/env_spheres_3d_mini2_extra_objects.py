import torch
from matplotlib import pyplot as plt

from torch_robotics.environments.env_spheres_3d_mini2 import EnvSpheres3DMini2
from torch_robotics.environments.primitives import ObjectField, MultiSphereField
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes


class EnvSpheres3DMini2ExtraObjects(EnvSpheres3DMini2):

    def __init__(self, tensor_args=None, **kwargs):
        # Mini2 has a smaller workspace and shorter links than Panda, so the extra
        # obstacles are scaled down and placed closer to the base workspace.
        # The layout below is slightly spread out to reduce local crowding around
        # the mid-arm sweep while keeping the same obstacle count and radii.
        obj_extra_list = [
            MultiSphereField(
                torch.tensor([
                    [-0.08, 0.14, 0.54],
                    [0.26, -0.16, 0.30],
                    [-0.22, -0.12, 0.22],
                    [0.06, 0.32, 0.60],
                ]),
                torch.tensor([
                    0.08,
                    0.08,
                    0.08,
                    0.08,
                ]),
                tensor_args=tensor_args,
            ),
        ]

        super().__init__(
            name=self.__class__.__name__,
            obj_extra_list=[ObjectField(obj_extra_list, 'extra-objects')],
            tensor_args=tensor_args,
            **kwargs
        )


if __name__ == '__main__':
    env = EnvSpheres3DMini2ExtraObjects(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=DEFAULT_TENSOR_ARGS,
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
