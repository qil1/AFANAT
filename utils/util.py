import os
import torch
import numpy as np
import random
from torch.optim.lr_scheduler import _LRScheduler


class GradualWarmupScheduler(_LRScheduler):
    """
    继承_LRScheduler类，需要自定义实现get_lr方法和step方法；可以参考其源码理解
    """

    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = warm_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = None  # _LRScheduler 成员  上一个epoch
        self.base_lrs = None  # _LRScheduler 成员 [group['initial_lr'] for group in optimizer.param_groups]
        super().__init__(optimizer)

    def get_lr(self):
        """
        返回每个参数组的当前lr形成的列表
        """
        if self.last_epoch >= self.total_epoch:  # 如果tol_epoch为20，那么从下标为22的epoch开始after_scheduler
            if self.after_scheduler:  # 如果有after_acheduler
                assert self.finished is False
                self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                self.finished = True
                return self.after_scheduler.base_lrs  # self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]  # 如果没有，后面的lr就不会变了
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                self.base_lrs]  # 线性warm_up

    def step(self, epoch=None, metrics=None):
        """
        更新optimizer的参数组的lr，默认做法：
            先更新last_epoch+=1  得到当前epoch  last_epoch初始值为-1，epoch下标从0开始
            再获取当前lr 【调用自定义实现的self.get_lr()】
            再使optimizer的参数组的lr变为当前lr  group['lr'] = lr
        """
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def cal_mpjpe_every_frame(gt32, pred32):
    return torch.sum(torch.mean(torch.norm(gt32 - pred32, dim=3), dim=2), dim=0) * 1000.


def cal_total_model_param(model_list):
    total = 0.
    for model in model_list:
        model_para = sum([param.nelement() for param in model.parameters()])
        total += model_para

    print("Number of parameter: %.2fM" % (total / 1e6))


def seed_torch(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
