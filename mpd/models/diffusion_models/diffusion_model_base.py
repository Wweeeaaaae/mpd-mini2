"""
Adapted from https://github.com/jannerm/diffuser
"""
import abc
import time
from collections import namedtuple
from copy import copy

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from abc import ABC

from torch.nn import DataParallel

from mpd.models.diffusion_models.helpers import cosine_beta_schedule, Losses, exponential_beta_schedule
from mpd.models.diffusion_models.sample_functions import extract, apply_hard_conditioning, guide_gradient_steps, \
    ddpm_sample_fn
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import to_numpy

#  生成相同时间步长 ℹ 函数
def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t

#  构建上下文字典 context
def build_context(model, dataset, input_dict):
    # input_dict is already normalized
    context = None
    if model.context_model is not None:
        context = dict()
        # 从已有可变环境中提取 env 环境
        if dataset.variable_environment:
            env_normalized = input_dict[f'{dataset.field_key_env}_normalized']
            context['env'] = env_normalized

        # 从已有可变环境中提取 tasks 任务
        task_normalized = input_dict[f'{dataset.field_key_task}_normalized']
        context['tasks'] = task_normalized
    return context

#  高斯扩散模型
class GaussianDiffusionModel(nn.Module, ABC):
    #  初始化模型参数，注册相关参数
    def __init__(self,
                 model=None,
                 variance_schedule='exponential',  #扩散过程中方差变化的策略
                 n_diffusion_steps=100,
                 clip_denoised=True,  #是否对去噪后的输出进行裁剪，以保持数值稳定性
                 predict_epsilon=False,  #模型是否预测噪声（epsilon）而不是直接预测去噪状态X0
                 loss_type='l2',  #训练模型的损失函数类型，'l2' 表示使用均方误差
                 context_model=None,
                 **kwargs):
        super().__init__()

        self.model = model

        self.context_model = context_model

        self.n_diffusion_steps = n_diffusion_steps

        self.state_dim = self.model.state_dim

        #  由不同变分推断计算扩散过程中的 betas 参数 【exponential（指数衰减）和cosine（余弦退火）】
        if variance_schedule == 'cosine':
            betas = cosine_beta_schedule(n_diffusion_steps, s=0.008, a_min=0, a_max=0.999)
        elif variance_schedule == 'exponential':
            betas = exponential_beta_schedule(n_diffusion_steps, beta_start=1e-4, beta_end=1.0)
        else:
            raise NotImplementedError

        #  计算 alphas 参数
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)  #alphas 序列的累积乘积
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])  #表示每一步之前的数据保留比例

        #  属性赋值
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        #  注册缓冲区，存储扩散模型中的一些中间计算结果
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))  #存储 alpha_t 的平方根
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))  #存储 1-alpha_t 的平方根
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))  #存储 1-alpha_t 的对数
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))  #存储 1/alpha_t 的平方根
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))  #存储 （1/alpha_t）-1 的平方根

        #  计算后验分布 q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)  #后验方差beta_t * [1 - alpha_t-1 /(1 - alpha_t)]
        self.register_buffer('posterior_variance', posterior_variance)

        ##  裁剪的对数后验方差（扩散链的开始处为零，直接计算对数会导致数值不稳定）
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))  #对数后验方差
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))  #后验均值系数 1 beta_t * [sqrt (alpha_t-1) / (1 - alpha_t)]
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))  #后验均值系数 2  [(1 - alpha_t-1) * sqrt (alpha_t)] / (1 - alpha_t)

        ##  获取损失系数并初始化目标函数
        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#
    #  ？？？？？
    def predict_noise_from_start(self, x_t, t, x0):
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return x0
        else:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
            ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    #  推断给定时间步的后验分布  q(x_{t-1} | x_t, x_0) 参数
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (  #估计后验均值
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)  #估计后验方差
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)  #裁剪的后验对数方差
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    #  计算给定输入 x 在特定时间步 t 下的预测上一时间步的均值（mean）和方差（variance）
    def p_mean_variance(self, x, hard_conds, context, t):
        if context is not None:
            context = self.context_model(context)

        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, context))  #预测出无噪声数据

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)  #去噪裁剪，将x_recon张量中的每个元素限制在-1.0和1.0之间
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance  #均值（model_mean），方差（posterior_variance），和对数方差（posterior_log_variance）

    @torch.no_grad()
    #  实际上的反向扩散过程实现函数
    def p_sample_loop(self, shape, hard_conds, context=None, return_chain=False,
                      sample_fn=ddpm_sample_fn,  #采样的函数DDPM
                      n_diffusion_steps_without_noise=0,  #不添加噪声的扩散步数
                      **sample_kwargs):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)  #初始化一个随机噪声张量 x
        x = apply_hard_conditioning(x, hard_conds)  #硬性条件应用到 X

        chain = [x] if return_chain else None

        #  从最后一个扩散步骤开始，反向迭代到第一个步骤
        for i in reversed(range(-n_diffusion_steps_without_noise, self.n_diffusion_steps)):
            t = make_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, hard_conds, context, t, **sample_kwargs)  #预测上一时间步样本状态
            x = apply_hard_conditioning(x, hard_conds)  #硬性条件应用

            if return_chain:
                chain.append(x)

        if return_chain:
            chain = torch.stack(chain, dim=1)
            return x, chain

        return x

    @torch.no_grad()
    def ddim_sample(
        self, shape, hard_conds,
        context=None, return_chain=False,
        t_start_guide=torch.inf,
        guide=None,
        n_guide_steps=1,  #引导阶段的步骤数
        **sample_kwargs,
    ):
        #  Adapted from https://github.com/ezhang7423/language-control-diffusion/blob/63cdafb63d166221549968c662562753f6ac5394/src/lcd/models/diffusion.py#L226
        #  设置采样参数
        device = self.betas.device
        batch_size = shape[0]
        total_timesteps = self.n_diffusion_steps
        sampling_timesteps = self.n_diffusion_steps // 5  #采样时间步数
        eta = 0.  #噪声调节参数

        #  生成时间对
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(0, total_timesteps - 1, steps=sampling_timesteps + 1, device=device)  #生成时间序列
        times = torch.cat((torch.tensor([-1], device=device), times))  #开始处添加时间点
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # 生成时间对[(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device=device)  #初始化一个随机噪声样本x
        x = apply_hard_conditioning(x, hard_conds)  #应用硬性条件

        chain = [x] if return_chain else None  #初始化采样链

        for time, time_next in time_pairs:
            #  生成时间步
            t = make_timesteps(batch_size, time, device)
            t_next = make_timesteps(batch_size, time_next, device)

            #  逆向过程中的模型输出（给定状态 X 和噪声水平 t ）
            model_out = self.model(x, t, context)

            #  预测起始状态和噪声
            x_start = self.predict_start_from_noise(x, t=t, noise=model_out)
            pred_noise = self.predict_noise_from_start(x, t=t, x0=model_out)

            #  判断逆扩散过程的最后步骤
            if time_next < 0:
                x = x_start
                x = apply_hard_conditioning(x, hard_conds)
                if return_chain:
                    chain.append(x)
                break

            #  从缓冲区提取alpha 下一时刻alpha，计算sigma,c 更新样本x
            alpha = extract(self.alphas_cumprod, t, x.shape)
            alpha_next = extract(self.alphas_cumprod, t_next, x.shape)

            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()

            x = x_start * alpha_next.sqrt() + c * pred_noise

            # 使用引导
            if guide is not None:
                if torch.all(t_next < t_start_guide):
                    x = guide_gradient_steps(
                        x,
                        hard_conds=hard_conds,
                        guide=guide,
                        **sample_kwargs
                    )

            # 增加噪声
            noise = torch.randn_like(x)
            x = x + sigma * noise
            x = apply_hard_conditioning(x, hard_conds)

            if return_chain:
                chain.append(x)

        if return_chain:
            chain = torch.stack(chain, dim=1)
            return x, chain

        return x

    @torch.no_grad()
    #  根据给定硬性条件生成样本
    def conditional_sample(self, hard_conds, horizon=None, batch_size=1, ddim=False, **sample_kwargs):
        #  ddim采样方式
        '''
            hard conditions : hard_conds : { (time, state), ... }
        '''
        horizon = horizon or self.horizon  #设置轨迹长度
        shape = (batch_size, horizon, self.state_dim)  #样本数量、轨迹长度和状态空间的维度

        if ddim:
            return self.ddim_sample(shape, hard_conds, **sample_kwargs)

        return self.p_sample_loop(shape, hard_conds, **sample_kwargs)

    def forward(self, cond, *args, **kwargs):
        raise NotImplementedError
        return self.conditional_sample(cond, *args, **kwargs)

    @torch.no_grad()
    def warmup(self, horizon=64, device='cuda'):
        shape = (2, horizon, self.state_dim)
        x = torch.randn(shape, device=device)
        t = make_timesteps(2, 1, device)
        self.model(x, t, context=None)  #前向传播

    @torch.no_grad()
    #  扩散模型生成样本过程（inference）
    def run_inference(self, context=None, hard_conds=None, n_samples=1, return_chain=False, **diffusion_kwargs):
        # context and hard_conds must be normalized
        hard_conds = copy(hard_conds)  #硬性条件
        context = copy(context)

        #  复制 hard_conds 和 context 重复 n_samples 次
        for k, v in hard_conds.items():
            new_state = einops.repeat(v, 'd -> b d', b=n_samples)
            hard_conds[k] = new_state

        if context is not None:
            for k, v in context.items():
                context[k] = einops.repeat(v, 'd -> b d', b=n_samples)

        #  从扩散模型采样
        samples, chain = self.conditional_sample(
            hard_conds, context=context, batch_size=n_samples, return_chain=True, **diffusion_kwargs
        )

        # chain: [ n_samples x (n_diffusion_steps + 1) x horizon x (state_dim)]
        # extract normalized trajectories
        trajs_chain_normalized = chain

        # 重新排列trajs: [ (n_diffusion_steps + 1) x n_samples x horizon x state_dim ]
        trajs_chain_normalized = einops.rearrange(trajs_chain_normalized, 'b diffsteps h d -> diffsteps b h d')

        if return_chain:
            return trajs_chain_normalized

        # eturn_chain 为 False，返回最后一个扩散步骤的样本。
        return trajs_chain_normalized[-1]

    # ------------------------------------------ training ------------------------------------------#
    #  前向扩散过程逐步将噪声引入数据sample
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, context, t, hard_conds):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_hard_conditioning(x_noisy, hard_conds)

        # context model
        if context is not None:
            context = self.context_model(context)

        # diffusion model
        x_recon = self.model(x_noisy, t, context)  #无噪声数据
        x_recon = apply_hard_conditioning(x_recon, hard_conds)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, context, *args):
        batch_size = x.shape[0]
        t = torch.randint(0, self.n_diffusion_steps, (batch_size,), device=x.device).long()
        return self.p_losses(x, context, t, *args)

