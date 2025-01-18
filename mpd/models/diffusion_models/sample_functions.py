import torch
from matplotlib import pyplot as plt


def apply_hard_conditioning(x, conditions):
    for t, val in conditions.items():
        x[:, t, :] = val.clone()
    return x


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


@torch.no_grad()
def ddpm_sample_fn(
        model, x, hard_conds, context, t,
        guide=None,
        n_guide_steps=1,
        scale_grad_by_std=False,
        t_start_guide=torch.inf,
        noise_std_extra_schedule_fn=None,  # 'linear'
        debug=False,
        **kwargs
):
    t_single = t[0]  #时间步张量 t 中提取第一个元素
    if t_single < 0:  #处理负时间步
        t = torch.zeros_like(t)

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, hard_conds=hard_conds, context=context, t=t)  #预测上一时间步的的均值和对数方差
    x = model_mean  #直接赋值

    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)  #裁剪的对数后验方差
    model_std = torch.exp(0.5 * model_log_variance)  #标准差
    model_var = torch.exp(model_log_variance)  #方差

    #  引导步骤
    if guide is not None and t_single < t_start_guide:
        x = guide_gradient_steps(
            x,
            hard_conds=hard_conds,
            guide=guide,
            n_guide_steps=n_guide_steps,
            scale_grad_by_std=scale_grad_by_std,
            model_var=model_var,
            debug=False,
        )

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    #  为了获得更平滑的结果，我们可以在整个扩散过程中对噪声标准偏差进行衰减
    #  this is roughly equivalent to using a temperature in the prior distribution
    #  是否提供了额外的噪声标准差调度函数
    if noise_std_extra_schedule_fn is None:
        noise_std = 1.0
    else:
        noise_std = noise_std_extra_schedule_fn(t_single)  #设置为0.5

    values = None
    return x + model_std * noise * noise_std, values


def guide_gradient_steps(
    x,
    hard_conds=None,
    guide=None,
    n_guide_steps=1, scale_grad_by_std=False,
    model_var=None,
    debug=False,
    **kwargs
):
    for _ in range(n_guide_steps):
        grad_scaled = guide(x)

        if scale_grad_by_std:
            grad_scaled = model_var * grad_scaled

        x = x + grad_scaled
        x = apply_hard_conditioning(x, hard_conds)

    return x
