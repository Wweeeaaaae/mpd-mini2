import copy
from math import ceil

import numpy as np
import os
import time
import torch
import wandb
from collections import defaultdict
from tqdm.autonotebook import tqdm

from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import dict_to_device, DEFAULT_TENSOR_ARGS, to_numpy


def get_num_epochs(num_train_steps, batch_size, dataset_len):
    return ceil(num_train_steps * batch_size / dataset_len)


def save_models_to_disk(models_prefix_l, epoch, total_steps, checkpoints_dir=None):
    for model, prefix in models_prefix_l:
        if model is not None:
            save_model_to_disk(model, epoch, total_steps, checkpoints_dir, prefix=f'{prefix}_')
            for submodule_key, submodule_value in model.submodules.items():
                save_model_to_disk(submodule_value, epoch, total_steps, checkpoints_dir,
                                   prefix=f'{prefix}_{submodule_key}_')


def save_model_to_disk(model, epoch, total_steps, checkpoints_dir=None, prefix='model_'):
    # If the model is frozen we do not save it again, since the parameters did not change
    if hasattr(model, 'is_frozen') and model.is_frozen:
        return

    torch.save(model.state_dict(), os.path.join(checkpoints_dir, f'{prefix}current_state_dict.pth'))
    torch.save(model.state_dict(), os.path.join(checkpoints_dir, f'{prefix}epoch_{epoch:04d}_iter_{total_steps:06d}_state_dict.pth'))
    torch.save(model, os.path.join(checkpoints_dir, f'{prefix}current.pth'))
    torch.save(model, os.path.join(checkpoints_dir, f'{prefix}epoch_{epoch:04d}_iter_{total_steps:06d}.pth'))


def save_losses_to_disk(train_losses, val_losses, checkpoints_dir=None):
    np.save(os.path.join(checkpoints_dir, f'train_losses.npy'), train_losses)
    np.save(os.path.join(checkpoints_dir, f'val_losses.npy'), val_losses)


class EarlyStopper:
    # https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch

    def __init__(self, patience=10, min_delta=0):
        self.patience = patience  # use -1 to deactivate it
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = torch.inf

    def early_stop(self, validation_loss):
        if self.patience == -1:
            return
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class EMA:
    """
    https://github.com/jannerm/diffuser
    (empirical) exponential moving average parameters
    """

    def __init__(self, beta=0.995):
        super().__init__()
        self.beta = beta

    #  更新EMA模型的平均参数
    def update_model_average(self, ema_model, current_model):
        for ema_params, current_params in zip(ema_model.parameters(), current_model.parameters()):
            old_weight, up_weight = ema_params.data, current_params.data
            ema_params.data = self.update_average(old_weight, up_weight)  #对更新参数进行加权平均

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def do_summary(
        summary_fn,
        train_steps_current,
        model,
        batch_dict,
        loss_info,
        datasubset,
        **kwargs
):
    if summary_fn is None:
        return

    with torch.no_grad():  #暂停梯度自动计算
        model.eval()  #设置为评估模式

        summary_fn(train_steps_current,
                   model,
                   batch_dict=batch_dict,
                   loss_info=loss_info,
                   datasubset=datasubset,
                   **kwargs
                   )

    # set model to training mode
    model.train()


def train(model=None, train_dataloader=None, epochs=None, lr=None, steps_til_summary=None, model_dir=None, loss_fn=None,
          train_subset=None,
          summary_fn=None, steps_til_checkpoint=None,
          val_dataloader=None, val_subset=None,
          clip_grad=False,
          clip_grad_max_norm=1.0,
          val_loss_fn=None,
          optimizers=None, steps_per_validation=10, max_steps=None,
          use_ema: bool = True,
          ema_decay: float = 0.995, step_start_ema: int = 1000, update_ema_every: int = 10,
          use_amp=False,
          early_stopper_patience=-1,  #-1表示不使用早停机制
          debug=False,
          tensor_args=DEFAULT_TENSOR_ARGS,
          **kwargs
          ):

    print(f'\n------- TRAINING STARTED -------\n')

    ema_model = None
    if use_ema:

        ema = EMA(beta=ema_decay)  #ema_decay参数是EMA衰减率
        ema_model = copy.deepcopy(model)  #创建独立模型副本

    #  模型优化器
    if optimizers is None:
        optimizers = [torch.optim.Adam(lr=lr, params=model.parameters())]  #Adam优化器

    #  自动混合精度（AMP）
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    #  是否有验证数据加载器
    if val_dataloader is not None:
        assert val_loss_fn is not None, "If validation set is passed, have to pass a validation loss_fn!"

    ##  构建保存目录
    os.makedirs(model_dir, exist_ok=True)

    summaries_dir = os.path.join(model_dir, 'summaries')  #总结信息路径
    os.makedirs(summaries_dir, exist_ok=True)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')  #模型检查点信息路径
    os.makedirs(checkpoints_dir, exist_ok=True)

    #  早停机制
    early_stopper = EarlyStopper(patience=early_stopper_patience, min_delta=0)  #  early_stopper_patience=-1 不使用早停，min_delta=0 参数设置了验证损失的最小变化量

    stop_training = False
    train_steps_current = 0

    #  训练前保存模型
    save_models_to_disk([(model, 'model'), (ema_model, 'ema_model')], 0, 0, checkpoints_dir)

    with tqdm(total=len(train_dataloader) * epochs, mininterval=1 if debug else 60) as pbar:  #创建进度条，total总迭代次数=批次总数*训练周期数，mininterval进度条更新间隔
        train_losses_l = []
        validation_losses_l = []
        for epoch in range(epochs):  #开启一个训练周期
            model.train()  #设置为训练模式
            for step, train_batch_dict in enumerate(train_dataloader):  #遍历数据加载器批次和内容
                ####################################################################################################
                #  计算训练损失
                ####################################################################################################
                with TimerCUDA() as t_training_loss:
                    train_batch_dict = dict_to_device(train_batch_dict, tensor_args['device'])  #将批次数据 train_batch_dict 中的所有张量移动到指定的设备上

                    #  计算损失
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                        train_losses, train_losses_info = loss_fn(model, train_batch_dict, train_subset.dataset)

                    train_loss_batch = 0.
                    train_losses_log = {}
                    for loss_name, loss in train_losses.items():
                        single_loss = loss.mean()  #计算损失均值，评估模型在该损失上的整体性能。
                        train_loss_batch += single_loss  #当前批次所有损失总和
                        train_losses_log[loss_name] = to_numpy(single_loss).item()  #转化存储

                ####################################################################################################
                #  生成总结信息
                if train_steps_current % steps_til_summary == 0:  #是否到达总结信息频率
                    # TRAINING
                    print(f"\n-----------------------------------------")
                    print(f"train_steps_current: {train_steps_current}")
                    print(f"t_training_loss: {t_training_loss.elapsed:.4f} sec")
                    print(f"Total training loss {train_loss_batch:.4f}")
                    print(f"Training losses {train_losses}")

                    train_losses_l.append((train_steps_current, train_losses_log))  #记录当前训练步骤损失

                    with TimerCUDA() as t_training_summary:
                        do_summary(  #总结
                            summary_fn,
                            train_steps_current,
                            ema_model if ema_model is not None else model,
                            train_batch_dict,
                            train_losses_info,
                            train_subset,
                            prefix='TRAINING ',  #为总结信息添加前缀，以区分训练和验证总结。
                            debug=debug,
                            tensor_args=tensor_args
                        )
                    print(f"t_training_summary: {t_training_summary.elapsed:.4f} sec")

                    ################################################################################################
                    #  计算验证损失和生成验证总结
                    validation_losses_log = {}
                    if val_dataloader is not None:
                        with TimerCUDA() as t_validation_loss:
                            print("Running validation...")
                            val_losses = defaultdict(list)
                            total_val_loss = 0.
                            for step_val, batch_dict_val in enumerate(val_dataloader):  #遍历验证数据批次：
                                batch_dict_val = dict_to_device(batch_dict_val, tensor_args['device'])  #将批次数据移动到设备
                                val_loss, val_loss_info = loss_fn(
                                    model, batch_dict_val, val_subset.dataset, step=train_steps_current)
                                # train_losses, train_losses_info = loss_fn(model, train_batch_dict, train_subset.dataset)
                                for name, value in val_loss.items():
                                    single_loss = to_numpy(value)  #将损失值转换为 NumPy 数组，添加到 val_losses 列表中
                                    val_losses[name].append(single_loss)
                                    total_val_loss += np.mean(single_loss).item()

                                #  判断验证步骤达到指定值
                                if step_val == steps_per_validation:
                                    break

                            #  计算平均损失
                            validation_losses = {}
                            for loss_name, loss in val_losses.items():
                                single_loss = np.mean(loss).item()
                                validation_losses[f'VALIDATION {loss_name}'] = single_loss
                            print("... finished validation.")

                        print(f"t_validation_loss: {t_validation_loss.elapsed:.4f} sec")
                        print(f"Validation losses {validation_losses}")

                        #  记录验证损失
                        validation_losses_log = validation_losses
                        validation_losses_l.append((train_steps_current, validation_losses_log))

                        # 在生成验证（validation）摘要信息时，只使用验证数据集的一个批次（batch）数据来进行。
                        with TimerCUDA() as t_validation_summary:
                            do_summary(
                                summary_fn,
                                train_steps_current,
                                ema_model if ema_model is not None else model,
                                batch_dict_val,
                                val_loss_info,
                                val_subset,
                                prefix='VALIDATION ',
                                debug=debug,
                                tensor_args=tensor_args
                            )
                        print(f"t_valididation_summary: {t_validation_summary.elapsed:.4f} sec")

                    wandb.log({**train_losses_log, **validation_losses_log}, step=train_steps_current)

                ####################################################################################################
                #  早停
                if early_stopper.early_stop(total_val_loss):
                    print(f'Early stopped training at {train_steps_current} steps.')
                    stop_training = True

                ####################################################################################################
                #  反向传播和优化器，模型参数更新和学习
                with TimerCUDA() as t_training_optimization:
                    for optim in optimizers:
                        optim.zero_grad()  #梯度清零

                    scaler.scale(train_loss_batch).backward()  #自动缩放梯度，执行反向传播，为损失求取梯度

                    if clip_grad:  #启用梯度剪切
                        for optim in optimizers:
                            scaler.unscale_(optim)  #取消梯度缩放
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=clip_grad_max_norm if isinstance(clip_grad, bool) else clip_grad  #最大范数
                        )

                    for optim in optimizers:
                        scaler.step(optim)  #更新模型参数

                    scaler.update()  #更新Gradsscaler内部状态

                    if ema_model is not None:
                        if train_steps_current % update_ema_every == 0:  #是否到达更新EMA的间隔
                            if train_steps_current < step_start_ema:  #是否到达开始更新EMA的起始步骤数
                                ema_model.load_state_dict(model.state_dict())  #将EMA模型参数重置为当前模型参数
                            ema.update_model_average(ema_model, model)  #更新EMA模型参数  EMAParams←β⋅EMAParams+(1−β)⋅ModelParams

                if train_steps_current % steps_til_summary == 0:  #定期打印时间和日志
                    print(f"t_training_optimization: {t_training_optimization.elapsed:.4f} sec")

                ####################################################################################################
                # SAVING
                ####################################################################################################
                pbar.update(1)   #训练完成了一个批次，进度条向前移动一个单位
                train_steps_current += 1  #更新完成步骤数

                if (steps_til_checkpoint is not None) and (train_steps_current % steps_til_checkpoint == 0):  #保存模型检查点
                    save_models_to_disk([(model, 'model'), (ema_model, 'ema_model')],
                                        epoch, train_steps_current, checkpoints_dir)
                    save_losses_to_disk(train_losses_l, validation_losses_l, checkpoints_dir)  #模型，损失函数保存到磁盘

                if stop_training or (max_steps is not None and train_steps_current == max_steps):
                    break

            if max_steps is not None and train_steps_current == max_steps:  #同步跳出循环
                break

        #  训练结束时更新 ema 模型
        if ema_model is not None:
            # update ema
            if train_steps_current < step_start_ema:
                # reset parameters ema
                ema_model.load_state_dict(model.state_dict())
            ema.update_model_average(ema_model, model)

        #  训练结束时保存模型和损失
        save_models_to_disk([(model, 'model'), (ema_model, 'ema_model')],
                            epoch, train_steps_current, checkpoints_dir)
        save_losses_to_disk(train_losses_l, validation_losses_l, checkpoints_dir)

        print(f'\n------- TRAINING FINISHED -------')
