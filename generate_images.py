# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the given model."""

import os
import re
import warnings
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist

warnings.filterwarnings('ignore', '`resume_download` is deprecated')
warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`')
warnings.filterwarnings('ignore', '1Torch was not compiled with flash attention')

# Modules referenced by distilled network pickles.
import training.networks_edm
import training.networks_edm2
import training.distillation.consistency.loss_cd
import training.distillation.moment_matching.loss_mm

#----------------------------------------------------------------------------
# Configuration presets.

model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions'

config_presets = {
    'edm2-img512-xs-fid':              dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.135.pkl'),      # fid = 3.53
    'edm2-img512-xs-dino':             dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.200.pkl'),      # fd_dinov2 = 103.39
    'edm2-img512-s-fid':               dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.130.pkl'),       # fid = 2.56
    'edm2-img512-s-dino':              dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.190.pkl'),       # fd_dinov2 = 68.64
    'edm2-img512-m-fid':               dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.100.pkl'),       # fid = 2.25
    'edm2-img512-m-dino':              dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.155.pkl'),       # fd_dinov2 = 58.44
    'edm2-img512-l-fid':               dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.085.pkl'),       # fid = 2.06
    'edm2-img512-l-dino':              dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.155.pkl'),       # fd_dinov2 = 52.25
    'edm2-img512-xl-fid':              dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.085.pkl'),      # fid = 1.96
    'edm2-img512-xl-dino':             dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.155.pkl'),      # fd_dinov2 = 45.96
    'edm2-img512-xxl-fid':             dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.070.pkl'),     # fid = 1.91
    'edm2-img512-xxl-dino':            dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.150.pkl'),     # fd_dinov2 = 42.84
    'edm2-img64-s-fid':                dnnlib.EasyDict(net=f'{model_root}/edm2-img64-s-1073741-0.075.pkl'),        # fid = 1.58
    'edm2-img64-m-fid':                dnnlib.EasyDict(net=f'{model_root}/edm2-img64-m-2147483-0.060.pkl'),        # fid = 1.43
    'edm2-img64-l-fid':                dnnlib.EasyDict(net=f'{model_root}/edm2-img64-l-1073741-0.040.pkl'),        # fid = 1.33
    'edm2-img64-xl-fid':               dnnlib.EasyDict(net=f'{model_root}/edm2-img64-xl-0671088-0.040.pkl'),       # fid = 1.33
    'edm2-img512-xs-guid-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.045.pkl',       gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.045.pkl', guidance=1.40), # fid = 2.91
    'edm2-img512-xs-guid-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.150.pkl',       gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.150.pkl', guidance=1.70), # fd_dinov2 = 79.94
    'edm2-img512-s-guid-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.025.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.025.pkl', guidance=1.40), # fid = 2.23
    'edm2-img512-s-guid-dino':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.085.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.085.pkl', guidance=1.90), # fd_dinov2 = 52.32
    'edm2-img512-m-guid-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.030.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.030.pkl', guidance=1.20), # fid = 2.01
    'edm2-img512-m-guid-dino':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.015.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=2.00), # fd_dinov2 = 41.98
    'edm2-img512-l-guid-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.015.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.20), # fid = 1.88
    'edm2-img512-l-guid-dino':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.035.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.035.pkl', guidance=1.70), # fd_dinov2 = 38.20
    'edm2-img512-xl-guid-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.020.pkl',       gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.020.pkl', guidance=1.20), # fid = 1.85
    'edm2-img512-xl-guid-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.030.pkl',       gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.030.pkl', guidance=1.70), # fd_dinov2 = 35.67
    'edm2-img512-xxl-guid-fid':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.015.pkl',      gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.20), # fid = 1.81
    'edm2-img512-xxl-guid-dino':       dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.015.pkl',      gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.70), # fd_dinov2 = 33.09
    'edm2-img512-s-autog-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.070.pkl',        gnet=f'{model_root}/edm2-img512-xs-0134217-0.125.pkl',        guidance=2.10), # fid = 1.34
    'edm2-img512-s-autog-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.120.pkl',        gnet=f'{model_root}/edm2-img512-xs-0134217-0.165.pkl',        guidance=2.45), # fd_dinov2 = 36.67
    'edm2-img512-xxl-autog-fid':       dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.075.pkl',      gnet=f'{model_root}/edm2-img512-m-0268435-0.155.pkl',         guidance=2.05), # fid = 1.25
    'edm2-img512-xxl-autog-dino':      dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.130.pkl',      gnet=f'{model_root}/edm2-img512-m-0268435-0.205.pkl',         guidance=2.30), # fd_dinov2 = 24.18
    'edm2-img512-s-uncond-autog-fid':  dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-uncond-2147483-0.070.pkl', gnet=f'{model_root}/edm2-img512-xs-uncond-0134217-0.110.pkl', guidance=2.85), # fid = 3.86
    'edm2-img512-s-uncond-autog-dino': dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-uncond-2147483-0.090.pkl', gnet=f'{model_root}/edm2-img512-xs-uncond-0134217-0.125.pkl', guidance=2.90), # fd_dinov2 = 90.39
    'edm2-img64-s-autog-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img64-s-1073741-0.045.pkl',         gnet=f'{model_root}/edm2-img64-xs-0134217-0.110.pkl',         guidance=1.70), # fid = 1.01
    'edm2-img64-s-autog-dino':         dnnlib.EasyDict(net=f'{model_root}/edm2-img64-s-1073741-0.105.pkl',         gnet=f'{model_root}/edm2-img64-xs-0134217-0.175.pkl',         guidance=2.20), # fd_dinov2 = 31.85
}

#----------------------------------------------------------------------------
# EDM sampler from the paper
# "Elucidating the Design Space of Diffusion-Based Generative Models",
# extended to support classifier-free guidance.

def edm_sampler(
    net, noise, labels=None, gnet=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like,
):
    # Guided denoiser.
    def denoise(x, t):
        Dx = net(x, t, labels).to(dtype)
        if guidance == 1:
            return Dx
        ref_Dx = gnet(x, t, labels).to(dtype)
        return ref_Dx.lerp(Dx, guidance)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = noise.to(dtype) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step.
        d_cur = (x_hat - denoise(x_hat, t_hat)) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            d_prime = (x_next - denoise(x_next, t_next)) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

#----------------------------------------------------------------------------
# TrigFlow sampler from the paper "Simplifying, Stabilizing and Scaling
# Continuous-Time Consistency Models" (Lu & Song, 2025). order=1 is DDIM,
# order=2 is single-step DPM-Solver-2S with r = 1. The corrector is skipped
# on the last step, so NFE = 2 * num_steps - 1.

def trigflow_sampler(
    net, noise, labels=None, gnet=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    sigma_data=0.5, order=2,
    dtype=torch.float32, randn_like=torch.randn_like,
):
    del randn_like, S_churn, S_min, S_max, S_noise

    def flow_and_eps(x, t_scalar):
        # Returns sigma_d * F and eps = sin(t) * x + cos(t) * sigma_d * F.
        t = torch.full([x.shape[0]], t_scalar, device=x.device, dtype=dtype)
        _, F = net(x, t, labels, return_F=True)
        F = F.to(dtype)
        if guidance != 1:
            _, ref_F = gnet(x, t, labels, return_F=True)
            ref_F = ref_F.to(dtype)
            F = ref_F.lerp(F, guidance)
        sigma_dF = sigma_data * F
        sin_t = torch.sin(t_scalar)
        cos_t = torch.cos(t_scalar)
        eps = sin_t * x + cos_t * sigma_dF
        return sigma_dF, eps

    # Karras sigma schedule, mapped to t = arctan(sigma / sigma_data).
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sigma_steps = torch.cat([sigma_steps, torch.zeros_like(sigma_steps[:1])]) # sigma_N = 0
    t_steps = torch.atan(sigma_steps / sigma_data)

    # Same starting point as EDM, mapped by x_t = cos(t) * x_sigma.
    x = noise.to(dtype) * sigma_steps[0] * torch.cos(t_steps[0])
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        delta = t_cur - t_next
        sigma_dF_cur, eps_cur = flow_and_eps(x, t_cur)

        # DDIM step.
        x_pred = torch.cos(delta) * x - torch.sin(delta) * sigma_dF_cur

        # DPM-Solver-2S correction, skipped on the last step.
        if order >= 2 and i < num_steps - 1:
            _, eps_next = flow_and_eps(x_pred, t_next)
            x = x_pred - (torch.sin(delta) / (2 * torch.cos(t_cur))) * (eps_next - eps_cur)
        else:
            x = x_pred

    return x

#----------------------------------------------------------------------------
# EDM Heun sampler on a TrigFlow network, as a cross-check for trigflow_sampler.
# D(x_sigma, sigma) = cos(t) * x_t - sin(t) * sigma_d * F(x_t / sigma_d, t),
# with t = arctan(sigma / sigma_data) and x_t = cos(t) * x_sigma.

class _TrigFlowEDMDenoiser:
    """Adapts a TrigFlow precond to the EDM denoiser interface D(x, sigma, labels)."""

    def __init__(self, net, sigma_data, dtype=torch.float32):
        self.net = net
        self.sigma_data = float(sigma_data)
        self.dtype = dtype
        self.img_channels = net.img_channels
        self.img_resolution = net.img_resolution
        self.label_dim = int(getattr(net, 'label_dim', 0))

    def __call__(self, x_sigma, sigma, labels=None):
        sigma = torch.as_tensor(sigma, device=x_sigma.device, dtype=self.dtype)
        if sigma.ndim == 0:
            sigma = sigma.expand(x_sigma.shape[0])
        sigma_b = sigma.reshape(-1, 1, 1, 1)
        t_b = torch.atan(sigma_b / self.sigma_data)
        x_t = torch.cos(t_b) * x_sigma
        _, F = self.net(x_t, t_b.flatten(), labels, return_F=True)
        F = F.to(self.dtype)
        return torch.cos(t_b) * x_t - torch.sin(t_b) * self.sigma_data * F


def trigflow_edm_heun_sampler(
    net, noise, labels=None, gnet=None,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    sigma_data=0.5, dtype=torch.float32, randn_like=torch.randn_like,
    **kwargs,
):
    del kwargs
    denoiser = _TrigFlowEDMDenoiser(net, sigma_data, dtype)
    gdenoiser = _TrigFlowEDMDenoiser(gnet, sigma_data, dtype) if gnet is not None else None
    return edm_sampler(
        denoiser, noise, labels=labels, gnet=gdenoiser,
        num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho,
        guidance=guidance, S_churn=0, dtype=dtype, randn_like=randn_like,
    )

#----------------------------------------------------------------------------
# sCM sampler. Extra steps restart from the fixed time t_mid.

def scm_sampler(
    net, noise, labels=None, gnet=None,
    num_steps=1, t_mid=1.1,
    sigma_max=80, sigma_data=0.5,
    dtype=torch.float32, randn_like=torch.randn_like, **kwargs,
):
    del gnet, kwargs
    device = noise.device
    t_max = torch.atan(torch.tensor(float(sigma_max) / float(sigma_data), device=device, dtype=dtype))

    def consistency_fn(x, t):
        # f(x_t, t) = cos(t) * x_t - sin(t) * sigma_d * F.
        t_b = torch.full([x.shape[0]], float(t), device=device, dtype=dtype)
        _, F = net(x, t_b, labels, return_F=True)
        F = F.to(dtype)
        t_t = torch.tensor(float(t), device=device, dtype=dtype)
        return torch.cos(t_t) * x - torch.sin(t_t) * sigma_data * F

    x = noise.to(dtype) * float(sigma_max) * torch.cos(t_max)
    x0 = consistency_fn(x, float(t_max.item()))
    if int(num_steps) <= 1:
        return x0
    # Re-noise to t_mid with fresh noise; a deterministic re-projection cannot improve x0.
    for _ in range(int(num_steps) - 1):
        z = randn_like(x0)
        t_t = torch.tensor(float(t_mid), device=device, dtype=dtype)
        x_t = torch.cos(t_t) * x0 + torch.sin(t_t) * float(sigma_data) * z
        x0 = consistency_fn(x_t, float(t_mid))
    return x0

#----------------------------------------------------------------------------
# MS-sCD sampler: one step per segment, from t_{k+1} to t_k, with the network
# conditioned on both. M = num_segments, or num_steps if not given.

def ms_scd_sampler(
    net, noise, labels=None, gnet=None,
    num_steps=8, num_segments=None,
    boundary_schedule='uniform_t',
    sigma_max=80, sigma_data=0.5,
    dtype=torch.float32, randn_like=torch.randn_like, **kwargs,
):
    del randn_like, gnet, kwargs
    from training.segment_schedule import build_boundaries, t_max_from_sigma

    M = int(num_segments) if num_segments is not None else int(num_steps)
    device = noise.device
    t_max = t_max_from_sigma(sigma_max, sigma_data)
    boundaries = build_boundaries(M, t_max, schedule=boundary_schedule, device=device, dtype=dtype)

    x = noise.to(dtype) * float(sigma_max) * torch.cos(torch.tensor(t_max, device=device, dtype=dtype))
    for k in range(M - 1, -1, -1):
        t_from = float(boundaries[k + 1].item())
        t_to = float(boundaries[k].item())
        delta = t_from - t_to
        t_from_b = torch.full([x.shape[0]], t_from, device=device, dtype=dtype)
        t_k_b = torch.full([x.shape[0]], t_to, device=device, dtype=dtype)
        _, F = net(x, t_from_b, labels, t_k=t_k_b, return_F=True)
        F = F.to(dtype)
        d = torch.tensor(delta, device=device, dtype=dtype)
        x = torch.cos(d) * x - torch.sin(d) * sigma_data * F
    return x

#----------------------------------------------------------------------------
# Euler sampler for MSCD students (NFE = num_steps).

def euler_sampler(
    net, noise, labels=None, gnet=None,
    num_steps=8, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like,
    step_sigmas=None, **kwargs,
):
    def denoise(x, t):
        Dx = net(x, t, labels).to(dtype)
        if guidance == 1:
            return Dx
        ref_Dx = gnet(x, t, labels).to(dtype)
        return ref_Dx.lerp(Dx, guidance)

    if step_sigmas is not None:
        t_steps = torch.as_tensor(step_sigmas, dtype=dtype, device=noise.device)
    else:
        step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
    x_next = noise.to(dtype) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur
        d_cur = (x_hat - denoise(x_hat, t_hat)) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
    return x_next

#----------------------------------------------------------------------------
# Ancestral sampler for moment-matching students (Salimans et al., 2024).

def ancestral_sampler(
    net, noise, labels=None, gnet=None, randn_like=torch.randn_like,
    num_steps=8, sigma_min=0.002, sigma_max=80, rho=7, **kwargs,
):
    from training.distillation.moment_matching.momentmatching_ops import (
        time_to_sigma, sample_conditional_posterior,
    )
    sigma_min = max(sigma_min, float(getattr(net, 'sigma_min', sigma_min)))
    sigma_max = min(sigma_max, float(getattr(net, 'sigma_max', sigma_max)))

    step_indices = torch.arange(num_steps + 1, dtype=torch.float64, device=noise.device)
    t_grid = step_indices / num_steps                       # [0, 1/k, ..., 1]
    sigma_grid = time_to_sigma(t_grid, sigma_min, sigma_max, rho)
    sigma_grid = sigma_grid.flip(0)                          # [sigma_max, ..., sigma_min]
    if hasattr(net, 'round_sigma'):
        sigma_grid = net.round_sigma(sigma_grid)

    x_next = noise.to(torch.float64) * sigma_grid[0]
    for i in range(num_steps):
        sig_t = sigma_grid[i]
        sig_s = sigma_grid[i + 1]
        x_hat = net(x_next, sig_t, labels).to(torch.float64)
        if i == num_steps - 1:
            x_next = x_hat
        else:
            sig_t_batch = sig_t.expand(x_next.shape[0])
            sig_s_batch = sig_s.expand(x_next.shape[0])
            x_next = sample_conditional_posterior(x_next, x_hat, sig_t_batch, sig_s_batch, randn_like=randn_like)
    return x_next

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Generate images for the given seeds in a distributed fashion.
# Returns an iterable that yields
# dnnlib.EasyDict(images, labels, noise, batch_idx, num_batches, indices, seeds)

def generate_images(
    net,                                        # Main network. Path, URL, or torch.nn.Module.
    gnet                = None,                 # Guiding network. None = same as main network.
    encoder             = None,                 # Instance of training.encoders.Encoder. None = load from network pickle.
    outdir              = None,                 # Where to save the output images. None = do not save.
    subdirs             = False,                # Create subdirectory for every 1000 seeds?
    seeds               = range(16, 24),        # List of random seeds.
    class_idx           = None,                 # Class label. None = select randomly.
    max_batch_size      = 32,                   # Maximum batch size for the diffusion model.
    encoder_batch_size  = 4,                    # Maximum batch size for the encoder. None = default.
    verbose             = True,                 # Enable status prints?
    device              = torch.device('cuda'), # Which compute device to use.
    sampler_fn          = edm_sampler,          # Which sampler function to use.
    **sampler_kwargs,                           # Additional arguments for the sampler function.
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load main network.
    if isinstance(net, str):
        if verbose:
            dist.print0(f'Loading main network from {net} ...')
        with dnnlib.util.open_url(net, verbose=(verbose and dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        net = data['ema'].to(device)
        if encoder is None:
            encoder = data.get('encoder', None)
            if encoder is None:
                encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder')
    assert net is not None

    # Load guidance network.
    if isinstance(gnet, str):
        if verbose:
            dist.print0(f'Loading guiding network from {gnet} ...')
        with dnnlib.util.open_url(gnet, verbose=(verbose and dist.get_rank() == 0)) as f:
            gnet = pickle.load(f)['ema'].to(device)
    if gnet is None:
        gnet = net

    # Initialize encoder.
    assert encoder is not None
    if verbose:
        dist.print0(f'Setting up {type(encoder).__name__}...')
    encoder.init(device)
    if encoder_batch_size is not None and hasattr(encoder, 'batch_size'):
        encoder.batch_size = encoder_batch_size

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide seeds into batches.
    num_batches = max((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    if verbose:
        dist.print0(f'Generating {len(seeds)} images...')

    # Return an iterable over the batches.
    class ImageIterable:
        def __len__(self):
            return len(rank_batches)

        def __iter__(self):
            # Loop over batches.
            for batch_idx, indices in enumerate(rank_batches):
                r = dnnlib.EasyDict(images=None, labels=None, noise=None, batch_idx=batch_idx, num_batches=len(rank_batches), indices=indices)
                r.seeds = [seeds[idx] for idx in indices]
                if len(r.seeds) > 0:

                    # Pick noise and labels.
                    rnd = StackedRandomGenerator(device, r.seeds)
                    r.noise = rnd.randn([len(r.seeds), net.img_channels, net.img_resolution, net.img_resolution], device=device)
                    r.labels = None
                    if net.label_dim > 0:
                        r.labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[len(r.seeds)], device=device)]
                        if class_idx is not None:
                            r.labels[:, :] = 0
                            r.labels[:, class_idx] = 1

                    # Generate images.
                    latents = dnnlib.util.call_func_by_name(func_name=sampler_fn, net=net, noise=r.noise,
                        labels=r.labels, gnet=gnet, randn_like=rnd.randn_like, **sampler_kwargs)
                    r.images = encoder.decode(latents)

                    # Save images.
                    if outdir is not None:
                        for seed, image in zip(r.seeds, r.images.permute(0, 2, 3, 1).cpu().numpy()):
                            image_dir = os.path.join(outdir, f'{seed//1000*1000:06d}') if subdirs else outdir
                            os.makedirs(image_dir, exist_ok=True)
                            PIL.Image.fromarray(image, 'RGB').save(os.path.join(image_dir, f'{seed:06d}.png'))

                # Yield results.
                torch.distributed.barrier() # keep the ranks in sync
                yield r

    return ImageIterable()

#----------------------------------------------------------------------------
# Samplers selectable with --sampler, also used by calculate_metrics.py.

SAMPLER_REGISTRY = {
    'edm': edm_sampler,
    'trigflow': trigflow_sampler,
    'scm': scm_sampler,
    'euler': euler_sampler,
    'ancestral': ancestral_sampler,
    'ms_scd': ms_scd_sampler,
}

SAMPLER_CHOICES = list(SAMPLER_REGISTRY.keys())

def resolve_sampler(sampler_name, opts):
    """Return the sampler and drop the options in opts that it does not take."""
    if sampler_name not in SAMPLER_REGISTRY:
        raise click.ClickException(f'Invalid sampler "{sampler_name}"')
    sampler_fn = SAMPLER_REGISTRY[sampler_name]
    if sampler_name in ('edm', 'euler', 'ancestral'):
        opts.pop('order', None)
        opts.pop('sigma_data', None)
    if sampler_name != 'ms_scd':
        opts.pop('boundary_schedule', None)
    if sampler_name != 'scm':
        opts.pop('t_mid', None)
    if sampler_name != 'euler':
        opts.pop('step_sigmas', None)
    elif opts.get('step_sigmas', None) is not None:
        opts['step_sigmas'] = parse_float_list(opts['step_sigmas'])
    return sampler_fn

def parse_float_list(s):
    """Parse a comma-separated list of floats, e.g. '80,1.1,0'."""
    if s is None or (isinstance(s, str) and s.strip() == ''):
        return None
    if isinstance(s, (list, tuple)):
        return [float(x) for x in s]
    return [float(x) for x in s.split(',')]

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------
# Command line interface.

@click.command()
@click.option('--preset',                   help='Configuration preset', metavar='STR',                             type=str, default=None)
@click.option('--net',                      help='Main network pickle filename', metavar='PATH|URL',                type=str, default=None)
@click.option('--gnet',                     help='Guiding network pickle filename', metavar='PATH|URL',             type=str, default=None)
@click.option('--outdir',                   help='Where to save the output images', metavar='DIR',                  type=str, required=True)
@click.option('--subdirs',                  help='Create subdirectory for every 1000 seeds',                        is_flag=True)
@click.option('--seeds',                    help='List of random seeds (e.g. 1,2,5-10)', metavar='LIST',            type=parse_int_list, default='16-19', show_default=True)
@click.option('--class', 'class_idx',       help='Class label  [default: random]', metavar='INT',                   type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                               type=click.IntRange(min=1), default=32, show_default=True)

@click.option('--steps', 'num_steps',       help='Number of sampling steps', metavar='INT',                         type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--sampler',                  help='Sampler family', metavar='edm|trigflow|scm|euler|ancestral|ms_scd',   type=click.Choice(SAMPLER_CHOICES), default='edm', show_default=True)
@click.option('--boundary-schedule', 'boundary_schedule', help='MS-sCD segment boundary schedule', metavar='STR',  type=click.Choice(['uniform_t', 'uniform_log_snr']), default='uniform_t', show_default=True)
@click.option('--order',                    help='Order for trigflow sampler', metavar='INT',                       type=click.IntRange(min=1, max=2), default=2, show_default=True)
@click.option('--t_mid', 't_mid',           help='Intermediate t for 2-step sCM sampler', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True), default=1.1, show_default=True)
@click.option('--step-sigmas', 'step_sigmas', help='Explicit sigma grid for euler, e.g. 80,1.1,0', metavar='LIST', type=str, default=None)
@click.option('--sigma_data',               help='Data std for trigflow sampler', metavar='FLOAT',                  type=click.FloatRange(min=0, min_open=True), default=0.5, show_default=True)
@click.option('--sigma_min',                help='Lowest noise level', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=0.002, show_default=True)
@click.option('--sigma_max',                help='Highest noise level', metavar='FLOAT',                            type=click.FloatRange(min=0, min_open=True), default=80, show_default=True)
@click.option('--rho',                      help='Time step exponent', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--guidance',                 help='Guidance strength  [default: 1; no guidance]', metavar='FLOAT',   type=float, default=None)
@click.option('--S_churn', 'S_churn',       help='Stochasticity strength', metavar='FLOAT',                         type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',           help='Stoch. min noise level', metavar='FLOAT',                         type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',           help='Stoch. max noise level', metavar='FLOAT',                         type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',       help='Stoch. noise inflation', metavar='FLOAT',                         type=float, default=1, show_default=True)

def cmdline(preset, **opts):
    """Generate random images using the given model.

    Examples:

    \b
    # Generate a couple of images and save them as out/*.png
    python generate_images.py --preset=edm2-img512-s-guid-dino --outdir=out

    \b
    # Generate 50000 images using 8 GPUs and save them as out/*/*.png
    torchrun --standalone --nproc_per_node=8 generate_images.py \\
        --preset=edm2-img64-s-fid --outdir=out --subdirs --seeds=0-49999
    """
    opts = dnnlib.EasyDict(opts)

    # Apply preset.
    if preset is not None:
        if preset not in config_presets:
            raise click.ClickException(f'Invalid configuration preset "{preset}"')
        for key, value in config_presets[preset].items():
            if opts[key] is None:
                opts[key] = value

    # Validate options.
    if opts.net is None:
        raise click.ClickException('Please specify either --preset or --net')
    if opts.guidance is None or opts.guidance == 1:
        opts.guidance = 1
        opts.gnet = None
    elif opts.gnet is None:
        raise click.ClickException('Please specify --gnet when using guidance')

    sampler_name = opts.pop('sampler')
    opts.sampler_fn = resolve_sampler(sampler_name, opts)

    # Generate.
    dist.init()
    image_iter = generate_images(**opts)
    for _r in tqdm.tqdm(image_iter, unit='batch', disable=(dist.get_rank() != 0)):
        pass

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
