# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modifications copyright (c) 2026, Vinay Saji Mathew.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""TrigFlow diffusion loss and learning rate schedule for the training loop."""

import numpy as np
import torch
import dnnlib
from torch_utils import persistence
from torch_utils import training_stats


@persistence.persistent_class
class TrigFlowLoss:
    """TrigFlow diffusion loss with adaptive weighting."""

    def __init__(self, P_mean=-0.8, P_std=1.6, sigma_data=0.5, augment_pipe=None, augment_kwargs=None):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.augment_pipe = augment_pipe
        if self.augment_pipe is None and augment_kwargs is not None:
            self.augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs)

    def __call__(self, net, images, labels=None):
        batch = images.shape[0]
        net_kwargs = {}
        if self.augment_pipe is not None:
            images, net_kwargs['augment_labels'] = self.augment_pipe(images)

        # ln(sigma_data * tan t) ~ N(P_mean, P_std^2), as in EDM.
        tau = torch.randn([batch, 1, 1, 1], device=images.device) * self.P_std + self.P_mean
        t = torch.atan(tau.exp() / self.sigma_data)

        z = torch.randn_like(images) * self.sigma_data
        x_t = torch.cos(t) * images + torch.sin(t) * z
        v_t = torch.cos(t) * z - torch.sin(t) * images

        _, logvar, F_theta = net(
            x_t,
            t.flatten(),
            labels,
            **net_kwargs,
            return_logvar=True,
            return_F=True,
        )
        resid = self.sigma_data * F_theta - v_t   # [B, C, H, W]

        # Adaptive weighting (Lu & Song, 2025) with w = -logvar. The paper's 1/D is
        # dropped: the per-pixel sum keeps fp16 gradients from underflowing.
        loss = resid.square() / logvar.exp() + logvar

        with torch.no_grad():
            mse = resid.reshape(batch, -1).square().mean(dim=1)
            training_stats.report('Loss/mse_raw', mse.mean())
            training_stats.report('Loss/logvar', logvar.mean())
        return loss


def trigflow_learning_rate_schedule(cur_nimg, batch_size, ref_lr=100e-4, ref_batches=35e3, rampup_Mimg=10):
    """EDM2 learning rate schedule with TrigFlow defaults."""
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
    if rampup_Mimg > 0:
        lr *= min(cur_nimg / (rampup_Mimg * 1e6), 1)
    return lr
