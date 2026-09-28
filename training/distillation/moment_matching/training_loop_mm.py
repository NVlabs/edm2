# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modifications copyright (c) 2026, Vinay Saji Mathew.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Training loop for moment-matching distillation (Salimans et al., 2024)."""

import glob
import json
import os
import re
import time
import copy
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc

from validation import maybe_validate

#----------------------------------------------------------------------------
# Keep the `keep_recent` newest training-state-*.pt plus the best-FID one;
# optionally prune primary snapshots. phEMA snapshots are never deleted.

def _cleanup_checkpoint_artifacts(run_dir, *, keep_recent, cleanup_snapshots):
    def _state_kimg(path):
        m = re.match(r'^training-state-(\d+)\.pt$', os.path.basename(path))
        return int(m.group(1)) if m else -1

    metrics_path = os.path.join(run_dir, 'metrics-val.jsonl')
    best_kimg = None
    if os.path.isfile(metrics_path):
        try:
            best_fid = float('inf')
            best_entry = None
            with open(metrics_path, 'rt') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    fid = row.get('fid')
                    kimg = row.get('kimg')
                    if fid is None or kimg is None:
                        continue
                    if float(fid) < best_fid:
                        best_fid = float(fid)
                        best_entry = row
            if best_entry is not None:
                best_kimg = int(best_entry['kimg'])
        except Exception:
            pass

    all_states = sorted(glob.glob(os.path.join(run_dir, 'training-state-*.pt')), key=_state_kimg)
    if not all_states:
        return
    protected = set(all_states[-keep_recent:])
    if best_kimg is not None:
        best_path = os.path.join(run_dir, f'training-state-{best_kimg:07d}.pt')
        if os.path.isfile(best_path):
            protected.add(best_path)
    for p in all_states:
        if p not in protected:
            try:
                os.remove(p)
            except OSError:
                pass
    if not cleanup_snapshots:
        return
    protected_kmigs = {_state_kimg(p) for p in protected}
    for p in glob.glob(os.path.join(run_dir, 'network-snapshot-*.pkl')):
        m = re.match(r'^network-snapshot-(\d{7})\.pkl$', os.path.basename(p))
        if not m:
            continue
        if int(m.group(1)) not in protected_kmigs:
            try:
                os.remove(p)
            except OSError:
                pass

#----------------------------------------------------------------------------
# Linear warmup then linear anneal to zero, expressed in optimizer steps.

def lr_multiplier(cur_step, total_steps, warmup_steps, anneal):
    if cur_step < warmup_steps:
        return cur_step / max(warmup_steps, 1)
    if anneal and total_steps > warmup_steps:
        return max(0.0, 1.0 - (cur_step - warmup_steps) / (total_steps - warmup_steps))
    return 1.0

#----------------------------------------------------------------------------

def training_loop(
    dataset_kwargs      = dict(class_name='training.dataset.ImageFolderDataset', path=None),
    encoder_kwargs      = dict(class_name='training.encoders.StandardRGBEncoder'),
    data_loader_kwargs  = dict(class_name='torch.utils.data.DataLoader', pin_memory=True, num_workers=2, prefetch_factor=2),
    network_kwargs      = dict(class_name='training.networks_edm.EDMPrecond'),
    optimizer_kwargs    = dict(class_name='torch.optim.Adam', betas=(0.0, 0.99), eps=1e-12),
    ema_kwargs          = dict(class_name='training.phema.PowerFunctionEMA'),

    run_dir             = '.',
    seed                = 0,
    batch_size          = 512,
    batch_gpu           = None,
    total_nimg          = 200<<20,
    slice_nimg          = None,
    status_nimg         = 128<<10,
    snapshot_nimg       = 8<<20,
    checkpoint_nimg     = 128<<20,
    phema_snapshot_nimg = None,
    checkpoint_keep_recent = 3,
    checkpoint_cleanup_snapshots = True,

    resume_state_dump   = None,

    # Moment-matching specific.
    teacher_pkl         = None,         # Teacher pickle (required).
    mm_kwargs           = None,         # Dict of MM hyperparams (k, sigma_*, weight_mode, ...).
    lr                  = 2e-6,         # Peak learning rate (shared by student + aux).
    lr_warmup_kimg      = 512,          # Linear warmup horizon in kimg.
    lr_anneal           = True,         # Linear anneal to zero after warmup.
    grad_clip           = 1.0,          # Max grad norm (0 = off).

    # Validation EMA.
    ema_halflife_kimg   = 500.0,
    ema_rampup_ratio    = 0.05,

    validation_kwargs   = None,
    wandb_kwargs        = None,

    loss_scaling        = 1,
    force_finite        = True,
    cudnn_benchmark     = True,
    device              = torch.device('cuda'),
):
    assert teacher_pkl is not None, 'moment matching requires a teacher pkl'
    prev_status_time = time.time()
    misc.set_random_seed(seed, dist.get_rank())
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()
    assert total_nimg % batch_size == 0
    assert slice_nimg is None or slice_nimg % batch_size == 0
    assert status_nimg is None or status_nimg % batch_size == 0
    assert snapshot_nimg is None or (snapshot_nimg % batch_size == 0 and snapshot_nimg % 1024 == 0)
    assert checkpoint_nimg is None or (checkpoint_nimg % batch_size == 0 and checkpoint_nimg % 1024 == 0)

    # Dataset, encoder, network.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    ref_image, ref_label = dataset_obj[0]
    dist.print0('Setting up encoder...')
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    ref_image = encoder.encode_latents(torch.as_tensor(ref_image).to(device).unsqueeze(0))
    dist.print0('Constructing student network...')
    interface_kwargs = dict(img_resolution=ref_image.shape[-1], img_channels=ref_image.shape[1], label_dim=ref_label.shape[-1])
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    net.train().requires_grad_(True).to(device)

    # Load frozen teacher and build the trainable auxiliary ("fake score") net.
    dist.print0(f'Loading teacher from {teacher_pkl}...')
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    with dnnlib.util.open_url(teacher_pkl, verbose=(dist.get_rank() == 0)) as f:
        teacher_data = pickle.load(f)
    if dist.get_rank() == 0:
        torch.distributed.barrier()
    teacher_net = (teacher_data['ema'] if isinstance(teacher_data, dict) and 'ema' in teacher_data else teacher_data)
    teacher_net = teacher_net.eval().requires_grad_(False).to(device)
    aux_net = copy.deepcopy(teacher_net).train().requires_grad_(True).to(device)
    # The aux net takes the student's dropout, not the teacher's.
    net_dropout = float(network_kwargs.get('dropout', 0.0))
    for m in aux_net.modules():
        if hasattr(m, 'dropout') and isinstance(m.dropout, (int, float)):
            m.dropout = net_dropout
    # Seed student from teacher when shapes match.
    try:
        misc.copy_params_and_buffers(src_module=teacher_net, dst_module=net, require_all=True)
        dist.print0('[MM INIT] Seeded student from teacher.')
    except Exception as e:
        dist.print0(f'[MM INIT] Could not seed student from teacher: {e}')
    del teacher_data

    if dist.get_rank() == 0:
        misc.print_module_summary(net, [
            torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device),
            torch.ones([batch_gpu], device=device),
            torch.zeros([batch_gpu, net.label_dim], device=device),
        ], max_nesting=2)

    dist.print0('Setting up training state...')
    state = dnnlib.EasyDict(cur_nimg=0, total_elapsed_time=0)
    ddp = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[device], broadcast_buffers=False,
        gradient_as_bucket_view=True, bucket_cap_mb=100,
    )

    # Not checkpointed; the teacher is reloaded from teacher_pkl on every run.
    from .loss_mm import EDMMomentMatchLoss
    loss_fn = EDMMomentMatchLoss(teacher_net=teacher_net, aux_net=aux_net, **(mm_kwargs or {}))

    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs)
    optimizer_aux = dnnlib.util.construct_class_by_name(params=aux_net.parameters(), **optimizer_kwargs)

    ema = dnnlib.util.construct_class_by_name(net=net, **ema_kwargs) if ema_kwargs is not None else None
    ema_val = copy.deepcopy(net).eval().requires_grad_(False)
    dist.print0(f'[MM EMA] Validation EMA: halflife={ema_halflife_kimg} kimg, rampup={ema_rampup_ratio}')

    checkpoint = dist.CheckpointIO(
        state=state, net=net, optimizer=optimizer, ema=ema, ema_val=ema_val,
        aux_net=aux_net, optimizer_aux=optimizer_aux,
    )
    if resume_state_dump is not None:
        checkpoint.load(resume_state_dump)

    # W&B (rank 0).
    wandb_run = None
    if wandb_kwargs is not None and wandb_kwargs.get('enabled', False) and dist.get_rank() == 0:
        try:
            import wandb as _wandb
            import sys
            for stream in [sys.stdout, sys.stderr]:
                if not hasattr(stream, 'isatty'):
                    stream.isatty = lambda: False
            init_kwargs = dict(
                project=wandb_kwargs.get('project', 'mm'),
                entity=wandb_kwargs.get('entity', None),
                name=wandb_kwargs.get('name', None),
                tags=wandb_kwargs.get('tags', None),
            )
            mode = wandb_kwargs.get('mode', 'online')
            if mode in ('offline', 'disabled'):
                init_kwargs['mode'] = mode
            wandb_run = _wandb.init(**init_kwargs)
        except Exception as _e:
            dist.print0(f'[W&B] init failed: {_e}')
            wandb_run = None

    stop_at_nimg = total_nimg
    if slice_nimg is not None:
        granularity = checkpoint_nimg if checkpoint_nimg is not None else snapshot_nimg if snapshot_nimg is not None else batch_size
        slice_end_nimg = (state.cur_nimg + slice_nimg) // granularity * granularity
        stop_at_nimg = min(stop_at_nimg, slice_end_nimg)
    assert stop_at_nimg > state.cur_nimg
    total_steps = total_nimg // batch_size
    warmup_steps = max(int(lr_warmup_kimg * 1000 / batch_size), 0)
    dist.print0(f'Training from {state.cur_nimg // 1000} kimg to {stop_at_nimg // 1000} kimg:')
    dist.print0()

    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed, start_idx=state.cur_nimg)
    dataset_iterator = iter(dnnlib.util.construct_class_by_name(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))
    prev_status_nimg = state.cur_nimg
    cumulative_training_time = 0
    start_nimg = state.cur_nimg
    stats_jsonl = None
    while True:
        done = (state.cur_nimg >= stop_at_nimg)

        # Status report.
        if status_nimg is not None and (done or state.cur_nimg % status_nimg == 0) and (state.cur_nimg != start_nimg or start_nimg == 0):
            cur_time = time.time()
            state.total_elapsed_time += cur_time - prev_status_time
            cur_process = psutil.Process(os.getpid())
            cpu_memory_usage = sum(p.memory_info().rss for p in [cur_process] + cur_process.children(recursive=True))
            dist.print0(' '.join(['Status:',
                'kimg',     f"{training_stats.report0('Progress/kimg', state.cur_nimg / 1e3):<9.1f}",
                'time',     f"{dnnlib.util.format_time(training_stats.report0('Timing/total_sec', state.total_elapsed_time)):<12s}",
                'sec/kimg', f"{training_stats.report0('Timing/sec_per_kimg', cumulative_training_time / max(state.cur_nimg - prev_status_nimg, 1) * 1e3):<7.3f}",
                'gpumem',   f"{training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}",
            ]))
            cumulative_training_time = 0
            prev_status_nimg = state.cur_nimg
            prev_status_time = cur_time
            torch.cuda.reset_peak_memory_stats()
            training_stats.default_collector.update()
            if dist.get_rank() == 0:
                if stats_jsonl is None:
                    stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
                fmt = {'Progress/tick': '%.0f', 'Progress/kimg': '%.3f', 'timestamp': '%.3f'}
                items = [(name, value.mean) for name, value in training_stats.default_collector.as_dict().items()] + [('timestamp', time.time())]
                items = [f'"{name}": ' + (fmt.get(name, '%g') % value if np.isfinite(value) else 'NaN') for name, value in items]
                stats_jsonl.write('{' + ', '.join(items) + '}\n')
                stats_jsonl.flush()
                if wandb_run is not None:
                    try:
                        log_dict = {name: float(value.mean) for name, value in training_stats.default_collector.as_dict().items() if np.isfinite(value.mean)}
                        log_dict['progress_kimg'] = state.cur_nimg / 1e3
                        wandb_run.log(log_dict, commit=True)
                    except Exception as _e:
                        dist.print0(f'[W&B] log failed: {_e}')
            dist.update_progress(state.cur_nimg // 1000, stop_at_nimg // 1000)
            if state.cur_nimg == stop_at_nimg and state.cur_nimg < total_nimg:
                dist.request_suspend()
            if dist.should_stop() or dist.should_suspend():
                done = True

        is_snap_boundary = (snapshot_nimg is not None and state.cur_nimg % snapshot_nimg == 0 and (state.cur_nimg != start_nimg or start_nimg == 0))
        _phema_nimg = phema_snapshot_nimg if phema_snapshot_nimg is not None else snapshot_nimg
        is_phema_boundary = (_phema_nimg is not None and state.cur_nimg % _phema_nimg == 0 and (state.cur_nimg != start_nimg or start_nimg == 0))

        # Primary snapshot = validation EMA of the distilled generator.
        if is_snap_boundary and dist.get_rank() == 0:
            data = dnnlib.EasyDict(encoder=encoder, dataset_kwargs=dataset_kwargs)
            data.ema = copy.deepcopy(ema_val).cpu().eval().requires_grad_(False).to(torch.float16)
            fname = f'network-snapshot-{state.cur_nimg//1000:07d}.pkl'
            dist.print0(f'Saving {fname} ... ', end='', flush=True)
            with open(os.path.join(run_dir, fname), 'wb') as f:
                pickle.dump(data, f)
            dist.print0('done')
            del data

        # phEMA snapshots for post-hoc reconstruction.
        if is_phema_boundary and dist.get_rank() == 0 and ema is not None:
            ema_list = ema.get() if isinstance(ema.get(), list) else [(ema.get(), '')]
            for ema_net, ema_suffix in ema_list:
                data = dnnlib.EasyDict(encoder=encoder, dataset_kwargs=dataset_kwargs)
                data.ema = copy.deepcopy(ema_net).cpu().eval().requires_grad_(False).to(torch.float16)
                fname = f'network-snapshot-{state.cur_nimg//1000:07d}{ema_suffix}.pkl'
                dist.print0(f'Saving {fname} ... ', end='', flush=True)
                with open(os.path.join(run_dir, fname), 'wb') as f:
                    pickle.dump(data, f)
                dist.print0('done')
                del data

        # In-training FID validation (ancestral few-step sampler, val_mode='mm').
        if is_snap_boundary and validation_kwargs is not None and validation_kwargs.get('enabled', False):
            try:
                maybe_validate(
                    cur_nimg=state.cur_nimg, snapshot_nimg=snapshot_nimg,
                    net_ema=ema_val, encoder=encoder, run_dir=run_dir,
                    dataset_kwargs=dataset_kwargs, validation_kwargs=validation_kwargs,
                    wandb_run=wandb_run,
                )
            except Exception as _e:
                dist.print0(f'[VAL] validation failed: {_e}')
            net.train()

        # Checkpoint.
        if checkpoint_nimg is not None and (done or state.cur_nimg % checkpoint_nimg == 0) and state.cur_nimg != start_nimg:
            checkpoint.save(os.path.join(run_dir, f'training-state-{state.cur_nimg//1000:07d}.pt'))
            misc.check_ddp_consistency(net)
            if int(checkpoint_keep_recent) > 0 and dist.get_rank() == 0:
                _cleanup_checkpoint_artifacts(run_dir, keep_recent=max(1, int(checkpoint_keep_recent)), cleanup_snapshots=bool(checkpoint_cleanup_snapshots))
            torch.distributed.barrier()

        if done:
            break

        cur_step = state.cur_nimg // batch_size
        mm_is_even = (cur_step % 2 == 0)
        loss_fn.set_step_n(cur_step)
        lr_now = lr * lr_multiplier(cur_step, total_steps, warmup_steps, lr_anneal)
        training_stats.report('Loss/learning_rate', lr_now)

        batch_start_time = time.time()
        misc.set_random_seed(seed, dist.get_rank(), state.cur_nimg)

        if mm_is_even:
            # Even step: update the aux net.
            optimizer_aux.zero_grad(set_to_none=True)
            for _round in range(num_accumulation_rounds):
                images, labels = next(dataset_iterator)
                images = encoder.encode_latents(images.to(device))
                loss = loss_fn(net=net, images=images, labels=labels.to(device))
                training_stats.report('Loss/loss', loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()
            # The aux net is outside DDP; sanitize NaN/Inf and all-reduce its grads by hand.
            for p in aux_net.parameters():
                if p.grad is not None:
                    if force_finite:
                        torch.nan_to_num(p.grad, nan=0, posinf=0, neginf=0, out=p.grad)
                    torch.distributed.all_reduce(p.grad)
                    p.grad.div_(dist.get_world_size())
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(aux_net.parameters(), max_norm=grad_clip)
            for g in optimizer_aux.param_groups:
                g['lr'] = lr_now
            optimizer_aux.step()
        else:
            # Odd step: update the student.
            optimizer.zero_grad(set_to_none=True)
            for _round in range(num_accumulation_rounds):
                with misc.ddp_sync(ddp, (_round == num_accumulation_rounds - 1)):
                    images, labels = next(dataset_iterator)
                    images = encoder.encode_latents(images.to(device))
                    loss = loss_fn(net=ddp, images=images, labels=labels.to(device))
                    training_stats.report('Loss/loss', loss)
                    loss.sum().mul(loss_scaling / batch_gpu_total).backward()
            if force_finite:
                for param in net.parameters():
                    if param.grad is not None:
                        torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
            for g in optimizer.param_groups:
                g['lr'] = lr_now
            optimizer.step()

        # EMAs advance only on student steps.
        state.cur_nimg += batch_size
        if not mm_is_even:
            if ema is not None:
                ema.update(cur_nimg=state.cur_nimg, batch_size=batch_size)
            halflife_nimg = ema_halflife_kimg * 1000
            if ema_rampup_ratio is not None and ema_rampup_ratio > 0:
                halflife_nimg = min(halflife_nimg, state.cur_nimg * ema_rampup_ratio)
            ema_beta = 0.5 ** (batch_size / max(halflife_nimg, 1e-8))
            with torch.no_grad():
                for p_ema, p_net in zip(ema_val.parameters(), net.parameters()):
                    p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
                for b_ema, b_net in zip(ema_val.buffers(), net.buffers()):
                    b_ema.copy_(b_net)

        cumulative_training_time += time.time() - batch_start_time

    if wandb_run is not None and dist.get_rank() == 0:
        try:
            wandb_run.finish()
        except Exception:
            pass

#----------------------------------------------------------------------------
