import torch
from OpenGL.wrapper import none_or_pass

from mpd.models import build_context


class GaussianDiffusionLoss:

    def __init__(self):
        pass

    @staticmethod
    def loss_fn(diffusion_model, input_dict, dataset, step=None):
        """
        Loss function for training diffusion-based generative models.
        """
        traj_normalized = input_dict[f'{dataset.field_key_traj}_normalized']  #获取归一化后的轨迹

        context = build_context(diffusion_model, dataset, input_dict)

        hard_conds = input_dict.get('hard_conds', {})
        loss, info = diffusion_model.loss(traj_normalized, context, hard_conds)

        loss_dict = {'diffusion_loss': loss}

        return loss_dict, info
