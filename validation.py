# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modifications copyright (c) 2026, Vinay Saji Mathew.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Distributed in-training FID for the TrigFlow teacher and the sCM, MS-sCD,
MSCD and moment-matching students. Reference statistics can be an .npz with
mu/sigma or a .pkl from calculate_metrics.py."""

import json
import math
import os
import pickle
import time
from typing import Any, Dict, Optional

import numpy as np
import scipy.linalg
import torch
import tqdm

import dnnlib
from torch_utils import distributed as dist

from generate_images import (
    StackedRandomGenerator,
    trigflow_sampler,
    trigflow_edm_heun_sampler,
    scm_sampler,
    euler_sampler,
    ancestral_sampler,
    ms_scd_sampler,
)

#----------------------------------------------------------------------------

def _load_inception_detector(device: torch.device):
    """Load the StyleGAN3 Inception-v3 detector, from a local copy if available."""
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    detector_kwargs = dict(return_features=True)
    feature_dim = 2048

    local_path = os.environ.get('EDM_INCEPTION_PATH', None)
    if local_path is None:
        repo_local = os.path.join(os.path.dirname(__file__), 'metrics', 'inception-2015-12-05.pkl')
        if os.path.isfile(repo_local):
            local_path = repo_local

    if local_path is not None and os.path.isfile(local_path):
        dist.print0(f'[VAL] Loading Inception-v3 from local file "{local_path}"...')
        with open(local_path, 'rb') as f:
            detector_net = pickle.load(f).to(device)
    else:
        dist.print0('[VAL] Loading Inception-v3 from NGC...')
        detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
        with dnnlib.util.open_url(detector_url, verbose=(dist.get_rank() == 0)) as f:
            detector_net = pickle.load(f).to(device)

    if dist.get_rank() == 0:
        torch.distributed.barrier()

    return detector_net, detector_kwargs, feature_dim

#----------------------------------------------------------------------------

def _prepare_reference_stats(ref: Optional[str]):
    """Load reference Inception mu/sigma from .npz or .pkl."""
    if ref is None:
        raise RuntimeError('FID validation requires --val-ref (.npz or .pkl path/URL).')
    if dist.get_rank() != 0:
        return None, None
    with dnnlib.util.open_url(ref) as f:
        if ref.lower().endswith('.npz'):
            data = dict(np.load(f))
            mu_ref = data['mu']
            sigma_ref = data['sigma']
        else:
            data = pickle.load(f)
            if isinstance(data, dict) and 'fid' in data and isinstance(data['fid'], dict):
                mu_ref = data['fid']['mu']
                sigma_ref = data['fid']['sigma']
            else:
                mu_ref = data['mu']
                sigma_ref = data['sigma']
    return mu_ref, sigma_ref

#----------------------------------------------------------------------------

def _fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    """Frechet distance between two Gaussians given their mean/cov."""
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    return float(np.real(m + np.trace(sigma + sigma_ref - s * 2)))

#----------------------------------------------------------------------------

def run_fid_validation(
    net: torch.nn.Module,
    encoder,
    *,
    run_dir: str,
    dataset_kwargs: Dict[str, Any],
    num_images: int = 50000,
    batch: int = 32,
    seed: int = 0,
    num_steps: int = 32,
    order: int = 2,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    sampler: str = 'dpm2s',
    ref: Optional[str] = None,
    step_kimg: Optional[int] = None,
    wandb_run=None,
) -> Dict[str, Optional[float]]:
    """FID for a TrigFlow teacher. sampler='dpm2s' is DPM-Solver-2S (order=2,
    NFE = 2 * num_steps - 1) or DDIM (order=1); 'edm_heun' runs EDM Heun on the
    TrigFlow network as a cross-check."""
    device = torch.device('cuda')
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    net = net.eval().requires_grad_(False).to(device)
    sigma_data = float(getattr(net, 'sigma_data', 0.5))
    is_heun = (sampler == 'edm_heun')
    nfe = (2 * num_steps - 1) if (is_heun or order >= 2) else num_steps
    dist.print0(
        f'[VAL] kimg={step_kimg} starting FID: num_images={num_images}, '
        f'sampler={sampler}, steps={num_steps}, order={order if not is_heun else 2} ({nfe} NFEs), '
        f'sigma_data={sigma_data}, sigma=[{sigma_min},{sigma_max}], rho={rho}'
    )

    mu_ref, sigma_ref = _prepare_reference_stats(ref)
    detector, detector_kwargs, feature_dim = _load_inception_detector(device)

    all_indices = torch.arange(num_images, device=torch.device('cpu'))
    num_batches = math.ceil(num_images / (batch * world_size)) * world_size
    all_batches = all_indices.tensor_split(num_batches)
    rank_batches = list(all_batches[rank :: world_size])

    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)

    label_dim = int(getattr(net, 'label_dim', 0))
    use_labels = bool(label_dim and dataset_kwargs.get('use_labels', False))

    progress = tqdm.tqdm(rank_batches, unit='batch', disable=(rank != 0), ascii=True, mininterval=5.0)
    non_empty = sum(1 for b in rank_batches if len(b) > 0)
    local_idx = 0

    for b_idxs in progress:
        bsize = len(b_idxs)
        if bsize == 0:
            continue
        local_idx += 1
        if rank == 0 and (local_idx == 1 or local_idx % 10 == 0 or local_idx == non_empty):
            pct = 100.0 * local_idx / max(non_empty, 1)
            dist.print0(f'[VAL] Progress (rank0): {local_idx}/{non_empty} ({pct:.1f}%)')

        seeds = (seed + b_idxs).tolist()
        rnd = StackedRandomGenerator(device, seeds)
        noise = rnd.randn(
            [bsize, net.img_channels, net.img_resolution, net.img_resolution], device=device
        )

        class_labels = None
        if use_labels:
            class_labels = torch.eye(label_dim, device=device)[
                rnd.randint(label_dim, size=[bsize], device=device)
            ]

        if is_heun:
            latents = trigflow_edm_heun_sampler(
                net, noise, labels=class_labels,
                num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho,
                sigma_data=sigma_data, randn_like=rnd.randn_like,
            )
        else:
            latents = trigflow_sampler(
                net, noise, labels=class_labels,
                num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho,
                sigma_data=sigma_data, order=order,
                randn_like=rnd.randn_like,
            )

        images_u8 = encoder.decode(latents)
        if images_u8.shape[1] == 1:
            images_u8 = images_u8.repeat([1, 3, 1, 1])
        feats = detector(images_u8, **detector_kwargs).to(torch.float64)
        mu += feats.sum(0)
        sigma += feats.T @ feats

    torch.distributed.all_reduce(mu)
    torch.distributed.all_reduce(sigma)
    mu /= num_images
    sigma -= mu.ger(mu) * num_images
    sigma /= (num_images - 1)

    fid_value = None
    if rank == 0:
        fid_value = _fid_from_inception_stats(
            mu.cpu().numpy(), sigma.cpu().numpy(), mu_ref, sigma_ref
        )
        result = dict(
            kimg=int(step_kimg) if step_kimg is not None else None,
            fid=float(fid_value),
            num_images=int(num_images),
            num_steps=int(num_steps),
            sampler=str(sampler),
            order=int(2 if is_heun else order),
            nfe=int(nfe),
            timestamp=time.time(),
        )
        with open(os.path.join(run_dir, 'metrics-val.jsonl'), 'at') as f:
            f.write(json.dumps(result) + '\n')

        if wandb_run is not None:
            try:
                log_dict = {
                    'val/fid': float(fid_value),
                    'val/num_images': int(num_images),
                    'val/nfe': int(nfe),
                }
                if step_kimg is not None:
                    log_dict['val/progress_kimg'] = int(step_kimg)
                wandb_run.log(log_dict, step=int(step_kimg) * 1000 if step_kimg is not None else None)
            except Exception as _e:
                dist.print0(f'[VAL] W&B log failed: {_e}')

        dist.print0(f'[VAL] kimg={step_kimg} FID={fid_value:g} ({nfe} NFEs, {num_images} images)')

    torch.distributed.barrier()
    return {'fid': float(fid_value) if fid_value is not None else None}

#----------------------------------------------------------------------------

def run_scm_fid_validation(
    net: torch.nn.Module,
    encoder,
    *,
    run_dir: str,
    dataset_kwargs: Dict[str, Any],
    num_images: int = 50000,
    batch: int = 32,
    seed: int = 0,
    num_steps: int = 1,
    t_mid: float = 1.1,
    sigma_max: float = 80.0,
    ref: Optional[str] = None,
    step_kimg: Optional[int] = None,
    wandb_run=None,
) -> Dict[str, Optional[float]]:
    """Generate samples with ``scm_sampler`` and compute FID."""
    device = torch.device('cuda')
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    net = net.eval().requires_grad_(False).to(device)
    sigma_data = float(getattr(net, 'sigma_data', 0.5))
    nfe = int(num_steps)
    dist.print0(
        f'[VAL] kimg={step_kimg} starting sCM FID: num_images={num_images}, '
        f'num_steps={num_steps}, t_mid={t_mid}, nfe={nfe}, sigma_data={sigma_data}'
    )

    mu_ref, sigma_ref = _prepare_reference_stats(ref)
    detector, detector_kwargs, feature_dim = _load_inception_detector(device)

    all_indices = torch.arange(num_images, device=torch.device('cpu'))
    num_batches = math.ceil(num_images / (batch * world_size)) * world_size
    all_batches = all_indices.tensor_split(num_batches)
    rank_batches = list(all_batches[rank :: world_size])

    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)

    label_dim = int(getattr(net, 'label_dim', 0))
    use_labels = bool(label_dim and dataset_kwargs.get('use_labels', False))

    progress = tqdm.tqdm(rank_batches, unit='batch', disable=(rank != 0), ascii=True, mininterval=5.0)
    non_empty = sum(1 for b in rank_batches if len(b) > 0)
    local_idx = 0

    for b_idxs in progress:
        bsize = len(b_idxs)
        if bsize == 0:
            continue
        local_idx += 1
        if rank == 0 and (local_idx == 1 or local_idx % 10 == 0 or local_idx == non_empty):
            pct = 100.0 * local_idx / max(non_empty, 1)
            dist.print0(f'[VAL] Progress (rank0): {local_idx}/{non_empty} ({pct:.1f}%)')

        seeds = (seed + b_idxs).tolist()
        rnd = StackedRandomGenerator(device, seeds)
        noise = rnd.randn(
            [bsize, net.img_channels, net.img_resolution, net.img_resolution], device=device
        )

        class_labels = None
        if use_labels:
            class_labels = torch.eye(label_dim, device=device)[
                rnd.randint(label_dim, size=[bsize], device=device)
            ]

        latents = scm_sampler(
            net, noise, labels=class_labels,
            num_steps=num_steps, t_mid=t_mid, sigma_max=sigma_max, sigma_data=sigma_data,
            randn_like=rnd.randn_like,
        )

        images_u8 = encoder.decode(latents)
        if images_u8.shape[1] == 1:
            images_u8 = images_u8.repeat([1, 3, 1, 1])
        feats = detector(images_u8, **detector_kwargs).to(torch.float64)
        mu += feats.sum(0)
        sigma += feats.T @ feats

    torch.distributed.all_reduce(mu)
    torch.distributed.all_reduce(sigma)
    mu /= num_images
    sigma -= mu.ger(mu) * num_images
    sigma /= (num_images - 1)

    fid_value = None
    if rank == 0:
        fid_value = _fid_from_inception_stats(
            mu.cpu().numpy(), sigma.cpu().numpy(), mu_ref, sigma_ref
        )
        result = dict(
            kimg=int(step_kimg) if step_kimg is not None else None,
            fid=float(fid_value),
            num_images=int(num_images),
            num_steps=int(num_steps),
            nfe=int(nfe),
            t_mid=float(t_mid) if num_steps >= 2 else None,
            mode='scm',
            timestamp=time.time(),
        )
        with open(os.path.join(run_dir, 'metrics-val.jsonl'), 'at') as f:
            f.write(json.dumps(result) + '\n')

        if wandb_run is not None:
            try:
                log_dict = {
                    'val/fid': float(fid_value),
                    'val/num_images': int(num_images),
                    'val/nfe': int(nfe),
                    'val/scm_steps': int(num_steps),
                }
                if step_kimg is not None:
                    log_dict['val/progress_kimg'] = int(step_kimg)
                wandb_run.log(log_dict, step=int(step_kimg) * 1000 if step_kimg is not None else None)
            except Exception as _e:
                dist.print0(f'[VAL] W&B log failed: {_e}')

        dist.print0(f'[VAL] kimg={step_kimg} sCM FID={fid_value:g} ({nfe} NFEs, {num_images} images)')

    torch.distributed.barrier()
    return {'fid': float(fid_value) if fid_value is not None else None}

#----------------------------------------------------------------------------

def _run_sampler_fid_validation(
    net, encoder, *, run_dir, dataset_kwargs, sample_fn, mode_tag,
    num_images=50000, batch=32, seed=0, ref=None, step_kimg=None,
    wandb_run=None, extra_record=None,
):
    """Shared FID driver; sample_fn(net, noise, class_labels, rnd) returns latents."""
    device = torch.device('cuda')
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    net = net.eval().requires_grad_(False).to(device)
    dist.print0(f'[VAL] kimg={step_kimg} starting {mode_tag} FID: num_images={num_images}')

    mu_ref, sigma_ref = _prepare_reference_stats(ref)
    detector, detector_kwargs, feature_dim = _load_inception_detector(device)

    all_indices = torch.arange(num_images, device=torch.device('cpu'))
    num_batches = math.ceil(num_images / (batch * world_size)) * world_size
    all_batches = all_indices.tensor_split(num_batches)
    rank_batches = list(all_batches[rank :: world_size])

    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)

    label_dim = int(getattr(net, 'label_dim', 0))
    use_labels = bool(label_dim and dataset_kwargs.get('use_labels', False))

    progress = tqdm.tqdm(rank_batches, unit='batch', disable=(rank != 0), ascii=True, mininterval=5.0)
    for b_idxs in progress:
        bsize = len(b_idxs)
        if bsize == 0:
            continue
        seeds = (seed + b_idxs).tolist()
        rnd = StackedRandomGenerator(device, seeds)
        noise = rnd.randn([bsize, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        class_labels = None
        if use_labels:
            class_labels = torch.eye(label_dim, device=device)[rnd.randint(label_dim, size=[bsize], device=device)]
        latents = sample_fn(net, noise, class_labels, rnd)
        images_u8 = encoder.decode(latents)
        if images_u8.shape[1] == 1:
            images_u8 = images_u8.repeat([1, 3, 1, 1])
        feats = detector(images_u8, **detector_kwargs).to(torch.float64)
        mu += feats.sum(0)
        sigma += feats.T @ feats

    torch.distributed.all_reduce(mu)
    torch.distributed.all_reduce(sigma)
    mu /= num_images
    sigma -= mu.ger(mu) * num_images
    sigma /= (num_images - 1)

    fid_value = None
    if rank == 0:
        fid_value = _fid_from_inception_stats(mu.cpu().numpy(), sigma.cpu().numpy(), mu_ref, sigma_ref)
        result = dict(
            kimg=int(step_kimg) if step_kimg is not None else None,
            fid=float(fid_value),
            num_images=int(num_images),
            mode=mode_tag,
            timestamp=time.time(),
        )
        if extra_record:
            result.update(extra_record)
        with open(os.path.join(run_dir, 'metrics-val.jsonl'), 'at') as f:
            f.write(json.dumps(result) + '\n')
        if wandb_run is not None:
            try:
                log_dict = {'val/fid': float(fid_value), 'val/num_images': int(num_images)}
                if step_kimg is not None:
                    log_dict['val/progress_kimg'] = int(step_kimg)
                wandb_run.log(log_dict, step=int(step_kimg) * 1000 if step_kimg is not None else None)
            except Exception as _e:
                dist.print0(f'[VAL] W&B log failed: {_e}')
        dist.print0(f'[VAL] kimg={step_kimg} {mode_tag} FID={fid_value:g} ({num_images} images)')

    torch.distributed.barrier()
    return {'fid': float(fid_value) if fid_value is not None else None}

#----------------------------------------------------------------------------

def run_cd_fid_validation(
    net, encoder, *, run_dir, dataset_kwargs, num_images=50000, batch=32, seed=0,
    num_steps=8, sigma_min=0.002, sigma_max=80.0, rho=7.0, ref=None, step_kimg=None,
    wandb_run=None, step_sigmas=None,
):
    """FID for MSCD students with the Euler sampler."""
    nfe = (len(step_sigmas) - 1) if step_sigmas is not None else int(num_steps)
    def sample_fn(net, noise, class_labels, rnd):
        return euler_sampler(
            net, noise, labels=class_labels, num_steps=num_steps,
            sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, randn_like=rnd.randn_like,
            step_sigmas=step_sigmas,
        )
    extra = dict(num_steps=int(num_steps), nfe=nfe, sampler='euler')
    if step_sigmas is not None:
        extra['step_sigmas'] = [float(s) for s in step_sigmas]
    return _run_sampler_fid_validation(
        net, encoder, run_dir=run_dir, dataset_kwargs=dataset_kwargs, sample_fn=sample_fn,
        mode_tag='cd', num_images=num_images, batch=batch, seed=seed, ref=ref,
        step_kimg=step_kimg, wandb_run=wandb_run,
        extra_record=extra,
    )

#----------------------------------------------------------------------------

def run_mm_fid_validation(
    net, encoder, *, run_dir, dataset_kwargs, num_images=50000, batch=32, seed=0,
    num_steps=8, sigma_min=0.002, sigma_max=80.0, rho=7.0, ref=None, step_kimg=None,
    wandb_run=None,
):
    """FID for moment-matching students with the ancestral sampler."""
    def sample_fn(net, noise, class_labels, rnd):
        return ancestral_sampler(
            net, noise, labels=class_labels, randn_like=rnd.randn_like,
            num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho,
        )
    return _run_sampler_fid_validation(
        net, encoder, run_dir=run_dir, dataset_kwargs=dataset_kwargs, sample_fn=sample_fn,
        mode_tag='mm', num_images=num_images, batch=batch, seed=seed, ref=ref,
        step_kimg=step_kimg, wandb_run=wandb_run,
        extra_record=dict(num_steps=int(num_steps), nfe=int(num_steps), sampler='ancestral'),
    )

#----------------------------------------------------------------------------

def run_ms_scd_fid_validation(
    net, encoder, *, run_dir, dataset_kwargs, num_images=50000, batch=32, seed=0,
    num_segments=8, boundary_schedule='uniform_t', sigma_max=80.0, sigma_data=None,
    ref=None, step_kimg=None, wandb_run=None,
):
    """FID for MS-sCD students with the M-step segment sampler."""
    sd = float(sigma_data) if sigma_data is not None else float(getattr(net, 'sigma_data', 0.5))

    def sample_fn(net, noise, class_labels, rnd):
        return ms_scd_sampler(
            net, noise, labels=class_labels, num_segments=int(num_segments),
            boundary_schedule=str(boundary_schedule), sigma_max=float(sigma_max),
            sigma_data=sd, randn_like=rnd.randn_like,
        )
    return _run_sampler_fid_validation(
        net, encoder, run_dir=run_dir, dataset_kwargs=dataset_kwargs, sample_fn=sample_fn,
        mode_tag='ms_scm', num_images=num_images, batch=batch, seed=seed, ref=ref,
        step_kimg=step_kimg, wandb_run=wandb_run,
        extra_record=dict(
            num_segments=int(num_segments), nfe=int(num_segments),
            boundary_schedule=str(boundary_schedule), sampler='ms_scd',
        ),
    )

#----------------------------------------------------------------------------

def maybe_validate(
    *,
    cur_nimg: int,
    snapshot_nimg: Optional[int],
    net_ema: torch.nn.Module,
    encoder,
    run_dir: str,
    dataset_kwargs: Dict[str, Any],
    validation_kwargs: Optional[Dict[str, Any]],
    wandb_run=None,
):
    """Run FID at snapshot boundaries if validation is enabled."""
    if validation_kwargs is None or not validation_kwargs.get('enabled', False):
        return None
    if snapshot_nimg is None or snapshot_nimg == 0:
        return None

    every = int(validation_kwargs.get('every', 1))
    snapshot_idx = cur_nimg // snapshot_nimg
    at_start = bool(validation_kwargs.get('at_start', False))

    if snapshot_idx == 0:
        should_run = at_start
    elif every > 0 and (snapshot_idx % every == 0):
        should_run = True
    else:
        should_run = False

    if not should_run:
        return None

    val_mode = str(validation_kwargs.get('val_mode', 'teacher')).lower()
    step_kimg = int(cur_nimg // 1000)
    common = dict(
        net=net_ema,
        encoder=encoder,
        run_dir=run_dir,
        dataset_kwargs=dataset_kwargs,
        num_images=int(validation_kwargs.get('num_images', 50000)),
        batch=int(validation_kwargs.get('batch', 32)),
        seed=int(validation_kwargs.get('seed', 0)),
        ref=validation_kwargs.get('ref', None),
        step_kimg=step_kimg,
        wandb_run=wandb_run,
    )

    if val_mode == 'scm':
        results = []
        scm_steps = validation_kwargs.get('scm_steps', [int(validation_kwargs.get('steps', 1))])
        if isinstance(scm_steps, int):
            scm_steps = [scm_steps]
        t_mid = float(validation_kwargs.get('scm_t_mid', 1.1))
        sigma_max = float(validation_kwargs.get('sigma_max', 80.0))
        for num_steps in scm_steps:
            results.append(run_scm_fid_validation(
                num_steps=int(num_steps),
                t_mid=t_mid,
                sigma_max=sigma_max,
                **common,
            ))
        return results[-1] if results else None

    if val_mode == 'cd':
        return run_cd_fid_validation(
            num_steps=int(validation_kwargs.get('steps', 8)),
            sigma_min=float(validation_kwargs.get('sigma_min', 0.002)),
            sigma_max=float(validation_kwargs.get('sigma_max', 80.0)),
            rho=float(validation_kwargs.get('rho', 7.0)),
            step_sigmas=validation_kwargs.get('step_sigmas', None),
            **common,
        )

    if val_mode == 'mm':
        return run_mm_fid_validation(
            num_steps=int(validation_kwargs.get('steps', 8)),
            sigma_min=float(validation_kwargs.get('sigma_min', 0.002)),
            sigma_max=float(validation_kwargs.get('sigma_max', 80.0)),
            rho=float(validation_kwargs.get('rho', 7.0)),
            **common,
        )

    if val_mode == 'ms_scm':
        seg_list = validation_kwargs.get(
            'ms_segments', [int(validation_kwargs.get('num_segments', 8))]
        )
        if isinstance(seg_list, int):
            seg_list = [seg_list]
        boundary_schedule = str(validation_kwargs.get('boundary_schedule', 'uniform_t'))
        sigma_max = float(validation_kwargs.get('sigma_max', 80.0))
        results = []
        for num_segments in seg_list:
            results.append(run_ms_scd_fid_validation(
                num_segments=int(num_segments),
                boundary_schedule=boundary_schedule,
                sigma_max=sigma_max,
                **common,
            ))
        return results[-1] if results else None

    return run_fid_validation(
        num_steps=int(validation_kwargs.get('steps', 18)),
        order=int(validation_kwargs.get('order', 2)),
        sampler=str(validation_kwargs.get('sampler', 'dpm2s')),
        sigma_min=float(validation_kwargs.get('sigma_min', 0.002)),
        sigma_max=float(validation_kwargs.get('sigma_max', 80.0)),
        rho=float(validation_kwargs.get('rho', 7.0)),
        **common,
    )

#----------------------------------------------------------------------------
