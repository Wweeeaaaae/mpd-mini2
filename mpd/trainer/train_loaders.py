import os

import torch
from torch.utils.data import DataLoader, random_split

from mpd import models, losses, datasets, summaries
from mpd.utils import model_loader, pretrain_helper
from torch_robotics.torch_utils.torch_utils import freeze_torch_model_params


@model_loader
#加载预训练模型或创建新模型，是否冻结模型参数
def get_model(model_class=None, checkpoint_path=None,
              freeze_loaded_model=False,
              tensor_args=None,
              **kwargs):

    if checkpoint_path is not None:
        model = torch.load(checkpoint_path)
        if freeze_loaded_model:
            freeze_torch_model_params(model)
    else:
        ModelClass = getattr(models, model_class)
        model = ModelClass(**kwargs).to(tensor_args['device'])

    return model


# @model_loader
# def get_model(model_class=None, marginal_prob_sigma=None, device=None, checkpoint_path=None, submodules=None,
#               **kwargs):
#     if marginal_prob_sigma is not None:
#         marginal_prob = MarginalProb(sigma=marginal_prob_sigma)
#         kwargs['marginal_prob_get_std'] = marginal_prob.get_std_fn
#
#     if submodules is not None:
#         for key, value in submodules.items():
#             kwargs[key] = get_model(**value)
#     Model = getattr(models, model_class)
#     model = Model(**kwargs).to(device)
#
#     if checkpoint_path is not None:
#         model.load_state_dict(torch.load(checkpoint_path))
#     if "pretrained_dir" in kwargs and kwargs["pretrained_dir"] is not None:
#         for param in model.parameters():
#             param.requires_grad = False
#     return model

@pretrain_helper
def get_pretrain_model(model_class=None, device=None, checkpoint_path=None, **kwargs):
    Model = getattr(models, model_class)
    model = Model(**kwargs).to(device)

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))

    return model


def build_module(model_class=None, submodules=None, **kwargs):
    if submodules is not None:
        for key, value in submodules.items():
            kwargs[key] = build_module(**value)

    Model = getattr(models, model_class)
    model = Model(**kwargs)

    return model


def get_loss(loss_class=None, **kwargs):
    LossClass = getattr(losses, loss_class)
    loss = LossClass(**kwargs)
    loss_fn = loss.loss_fn  #提取实际的损失函数
    return loss_fn


def get_dataset(dataset_class=None,
                dataset_subdir=None,
                batch_size=2,
                val_set_size=0.05,
                results_dir=None,
                save_indices=False,  #保存索引
                **kwargs):
    DatasetClass = getattr(datasets, dataset_class)
    print('\n---------------Loading data')
    full_dataset = DatasetClass(dataset_subdir=dataset_subdir, **kwargs)
    #dataset_subdir: EnvSpheres3D-RobotPanda 从data_trained_models/EnvSpheres3D-RobotPanda/args.yaml
    print(full_dataset)

    # 划分训练集和验证集，创建训练数据加载器，验证数据加载器
    train_subset, val_subset = random_split(full_dataset, [1-val_set_size, val_set_size])
    train_dataloader = DataLoader(train_subset, batch_size=batch_size)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size)

    if save_indices:
        #  保存训练集和验证集的索引（用于以后的评估）
        torch.save(train_subset.indices, os.path.join(results_dir, f'train_subset_indices.pt'))
        torch.save(val_subset.indices, os.path.join(results_dir, f'val_subset_indices.pt'))

    return train_subset, train_dataloader, val_subset, val_dataloader


def get_summary(summary_class=None, **kwargs):
    if summary_class is None:
        return None
    SummaryClass = getattr(summaries, summary_class)
    summary_fn = SummaryClass(**kwargs).summary_fn
    return summary_fn


# def get_sampler(sampler_class=None, **kwargs):
#     diffusion_coeff = DiffusionCoefficient(sigma=marginal_prob_sigma)
#     Sampler = getattr(samplers, sampler_class)
#     sampler = Sampler(marginal_prob_get_std_fn=marginal_prob.get_std_fn,
#                       diffusion_coeff_fn=diffusion_coeff,
#                       sde_sigma=marginal_prob_sigma,
#                       **kwargs)
#     return sampler
