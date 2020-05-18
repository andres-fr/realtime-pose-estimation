# -*- coding:utf-8 -*-


"""
This module contains optimizers and error metrics for gradient descent
optimization.
"""


import torch
#
from .third_party.fp16_utils.fp16_optimizer import FP16_Optimizer


# #############################################################################
# # GLOBALS
# #############################################################################


# #############################################################################
# # OPTIMIZERS
# #############################################################################
def get_sgd_optimizer(params,
                      initial_lr=0.001,
                      momentum=0,  # 0.9
                      weight_decay=0.0001,  # 0.0001
                      nesterov=False,
                      half_precision=True):
    """
    :returns: An instance of ``torch.optim.SGD`` with the required parameters.
      Optionally, if ``half_precision`` is true, the instance is wrapped using
      ``FP16_Optimizer``, from the third party software. This allows it to
      transparently optimize models that contain half precision parameters.
    """
    opt = torch.optim.SGD(params, lr=initial_lr, momentum=momentum,
                          weight_decay=weight_decay, nesterov=nesterov)
    if half_precision:
        opt = FP16_Optimizer(opt, static_loss_scale=1.0,
                             dynamic_loss_scale=False)
    return opt


class SgdrScheduler:
    """
    A reseteable ``torch.optim.lr_scheduler.CosineAnnealingLR``.
    See the constructor docstring for more details.
    """

    def __init__(self, optimizer, max_lr=1, min_lr=0, period=100,
                 scale_max_lr=1.0, scale_min_lr=1.0, scale_period=1.0):
        """
        :param optimizer: A ``torch.optim`` instance to be passed to the
          ``torch.optim.lr_scheduler``

        The main shape of this scheduler is a cosine function starting at 0
        degrees and finishing at 90 (i.e. a smooth descending function) after
        the given ``period`` number of steps.
        The max value at 0 degrees is ``max_lr``, and the min is ``min_lr``.
        Once it reaches the bottom, it repeats the cycle, but the parameters
        ``max_lr, min_lr`` and ``period`` are rescaled by the given ``scale``
        factors (e.g. if ``scale_period=2.0``, every next cycle will take twice
        as many steps as the prior one). This can be useful to e.g. make the
        learning rate cycle slower and smoother as training progresses.
        """
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.period = period
        #
        self.scale_max_lr = scale_max_lr
        self.scale_min_lr = scale_min_lr
        self.scale_period = scale_period
        #
        self.scheduler = self._new_scheduler()
        self.step_count = 0

    def _new_scheduler(self):
        """
        Since built-in schedulers don't have a reset mechanism, we create a
        new one from scratch every cycle following a factory pattern, as
        suggested here::

          discuss.pytorch.org/t/resetting-scheduler-and-optimizer-learning-rate
        """
        for g in self.optimizer.param_groups:
            g["lr"] = self.max_lr
        #
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.period, eta_min=self.min_lr)
        return scheduler

    def step(self):
        """
        Call this method after ``optimizer.step()`` to advance the scheduling
        of the learning rate.
        """
        self.scheduler.step()
        self.step_count += 1
        #
        if self.step_count % self.period < 1:
            self.step_count = 0
            self.max_lr *= self.scale_max_lr
            self.min_lr *= self.scale_min_lr
            self.period *= self.scale_period
            #
            for g in self.optimizer.param_groups:
                g["initial_lr"] = self.max_lr
            self.scheduler = self._new_scheduler()


# #############################################################################
# # LOSS FUNCTIONS
# #############################################################################
class MaskedMseLoss(torch.nn.Module):
    """
    An MSELoss, where, if a mask is provided, the MSE inputs are be multiplied
    by the mask. This has the effect that if e.g. the mask is boolean, all
    false entries will not propagate a gradient and won't contribute to the
    learning.
    """
    def __init__(self):
        """
        """
        super().__init__()
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, pred, gt, mask=None):
        """
        """
        if mask is not None:
            pred = pred * mask
            gt = gt * mask
        return self.loss_fn(pred, gt)


class MaskedBceWithLogits(torch.nn.Module):
    """
    """
    def __init__(self, pos_weight=1, device="cpu"):
        """
        """
        super().__init__()
        self.pos_weight = torch.ones(1) * pos_weight
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.to(device)

    def forward(self, pred, gt, mask=None):
        """
        """
        if mask is not None:
            pred = pred * mask
            gt = gt * mask
        return self.loss_fn(pred, gt)


class DistillationLoss(torch.nn.Module):
    """
    Given the student predictions, teacher predictions and ground truth,
    this loss equals::

      loss = alpha*MSE(teacher, gt) + (1 - alpha)*MSE(student, gt)

    Alpha represents the 'mix' between GT and teacher, and is expected to
    be between 0 and 1. It is the teacher's relative weight (i.e. 1=teacher)
    """

    def __init__(self):
        """
        """
        super().__init__()
        self.teacher_loss_fn = MaskedMseLoss()
        self.gt_loss_fn = MaskedMseLoss()

    def forward(self, student_pred, teacher_pred, gt,
                alpha=0.5, mask=None):
        """
        """
        assert 0 <= alpha <= 1, "alpha must be in range [0, 1]"
        teacher_loss = self.teacher_loss_fn(student_pred, teacher_pred, mask)
        gt_loss = self.gt_loss_fn(student_pred, gt, mask)
        result = alpha * teacher_loss + (1 - alpha) * gt_loss
        return result


class DistillationLossKeypointMining(DistillationLoss):
    """
    Inspired in http://arxiv.org/abs/1711.07319, section 4.2.3. Pick_top=8
    """
    def forward(self, student_pred, teacher_pred, gt, alpha=0.5, mask=None,
                background_factor=0, pick_top=None):
        """
        """

        # TODO: if
        # 2. make connected component analysis of all separate components
        # 3. Compute the square error for each pixel
        # 4.
        assert 0 <= background_factor <= 1
        if mask is not None:
            with torch.no_grad():
                bg_mask = gt.cpu() == 0
                mask[bg_mask] *= background_factor
        #
        if pick_top is not None:
            with torch.no_grad():
                # make connected component analysis of all separate components
                # compute square distill error for each pixel
                # average error for each connected component and sort descending
                # all entries below highest "PICK_TOP" entries
                raise NotImplementedError
        #
        # import matplotlib
        # from .helpers import plot_arrays
        # matplotlib.use('TkAgg')
        # plot_arrays(gt[0].max(dim=0)[0].cpu(), teacher_pred[0].max(dim=0)[0].cpu(), mask[0].max(dim=0)[0].cpu())
        # breakpoint()
        return super().forward(student_pred, teacher_pred, gt, alpha, mask)


class DistillationBceLossKeypointMining(DistillationLoss):
    """
    """
    def __init__(self, teacher_pos_weight=1, gt_pos_weight=1, device="cpu"):
        """
        """
        super().__init__()
        self.teacher_loss_fn = MaskedBceWithLogits(teacher_pos_weight, device)
        self.gt_loss_fn = MaskedBceWithLogits(gt_pos_weight, device)

    def forward(self, student_pred, teacher_pred, gt, alpha=0.5, mask=None,
                background_factor=0, pick_top=None):
        """
        only difference with super is that here targets MUST be between 0
        and 1, so we normalize them if needed
        """
        assert 0 <= background_factor <= 1
        # normalize GT/teacher maps if needed
        with torch.no_grad():
            if gt.min() < 0:
                gt = gt - gt.min()
            if gt.max() > 1:
                gt = gt / gt.max()
            if teacher_pred.min() < 0:
                teacher_pred = teacher_pred - teacher_pred.min()
            if teacher_pred.max() > 1:
                teacher_pred = teacher_pred / teacher_pred.max()
        #
        if mask is not None:
            with torch.no_grad():
                bg_mask = gt.cpu() == 0
                mask[bg_mask] *= background_factor
        #
        if pick_top is not None:
            with torch.no_grad():
                raise NotImplementedError
        #
        return super().forward(student_pred, teacher_pred, gt, alpha, mask)
