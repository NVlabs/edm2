# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import glob
import json
import os
import re
import sys
import time
import copy
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import persistence
from torch_utils import misc


#----------------------------------------------------------------------------
# Stream wrapper for loggers without isatty(), which wandb probes.

class _WandbStreamProxy:
    """Wrap streams that do not implement .isatty()."""
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def _ensure_wandb_tty_streams():
    for stream_name in ('stdout', 'stderr'):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        has_working_isatty = False
        if hasattr(stream, 'isatty'):
            try:
                stream.isatty()
                has_working_isatty = True
            except Exception:
                has_working_isatty = False
        if not has_working_isatty:
            setattr(sys, stream_name, _WandbStreamProxy(stream))


#----------------------------------------------------------------------------
# Delete old training states, keeping the newest `keep_recent` and the best-FID
# one. Optionally prune primary snapshots too; phEMA and -val pickles are kept.

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

    all_states = sorted(
        glob.glob(os.path.join(run_dir, 'training-state-*.pt')),
        key=_state_kimg,
    )
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

    protected_kimgs = {_state_kimg(p) for p in protected}
    for p in glob.glob(os.path.join(run_dir, 'network-snapshot-*.pkl')):
        base = os.path.basename(p)
        m = re.match(r'^network-snapshot-(\d{7})\.pkl$', base)
        if not m:
            continue
        kimg = int(m.group(1))
        if kimg not in protected_kimgs:
            try:
                os.remove(p)
            except OSError:
                pass

#----------------------------------------------------------------------------
# Uncertainty-based loss function (Equations 14,15,16,21) proposed in the
# paper "Analyzing and Improving the Training Dynamics of Diffusion Models".

@persistence.persistent_class
class EDM2Loss:
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(images) * sigma
        denoised, logvar = net(images + noise, sigma, labels, return_logvar=True)
        loss = (weight / logvar.exp()) * ((denoised - images) ** 2) + logvar
        return loss

#----------------------------------------------------------------------------
# Learning rate decay schedule used in the paper "Analyzing and Improving
# the Training Dynamics of Diffusion Models".

def learning_rate_schedule(cur_nimg, batch_size, ref_lr=100e-4, ref_batches=70e3, rampup_Mimg=10):
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
    if rampup_Mimg > 0:
        lr *= min(cur_nimg / (rampup_Mimg * 1e6), 1)
    return lr

#----------------------------------------------------------------------------
# Main training loop.

def training_loop(
    dataset_kwargs      = dict(class_name='training.dataset.ImageFolderDataset', path=None),
    encoder_kwargs      = dict(class_name='training.encoders.StabilityVAEEncoder'),
    data_loader_kwargs  = dict(class_name='torch.utils.data.DataLoader', pin_memory=True, num_workers=4, prefetch_factor=2),
    network_kwargs      = dict(class_name='training.networks_edm2.Precond'),
    loss_kwargs         = dict(class_name='training.training_loop.EDM2Loss'),
    optimizer_kwargs    = dict(class_name='torch.optim.Adam', betas=(0.9, 0.99)),
    lr_kwargs           = dict(func_name='training.training_loop.learning_rate_schedule'),
    ema_kwargs          = dict(class_name='training.phema.PowerFunctionEMA'),

    run_dir             = '.',      # Output directory.
    seed                = 0,        # Global random seed.
    batch_size          = 2048,     # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU. None = no limit.
    total_nimg          = 8<<30,    # Train for a total of N training images.
    slice_nimg          = None,     # Train for a maximum of N training images in one invocation. None = no limit.
    status_nimg         = 128<<10,  # Report status every N training images. None = disable.
    snapshot_nimg       = 8<<20,    # Save network snapshot every N training images. None = disable.
    checkpoint_nimg     = 128<<20,  # Save state checkpoint every N training images. None = disable.
    checkpoint_keep_recent          = 3,    # Retain this many newest training-state-*.pt files (+ best-FID .pt). <=0 disables pruning.
    checkpoint_cleanup_snapshots    = True, # Also prune primary network-snapshot-{kimg}.pkl; phEMA / -val pkls are never touched.
    init_from           = None,     # Optional network pickle to initialize from.
    transfer_fn         = None,     # Optional transfer function name for initialization.
    teacher_pkl         = None,     # Frozen teacher for sCM/sCD training.
    validation_kwargs   = None,     # Optional dict of FID validation params. None = no in-training FID.
    wandb_kwargs        = None,     # Optional dict for Weights & Biases logging.

    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    force_finite        = True,     # Get rid of NaN/Inf gradients before feeding them to the optimizer.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
):
    # Initialize.
    prev_status_time = time.time()
    misc.set_random_seed(seed, dist.get_rank())
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    wandb_run = None

    # Validate batch size.
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

    is_scm_mode = teacher_pkl is not None

    # Setup dataset, encoder, and network.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    ref_image, ref_label = dataset_obj[0]
    dist.print0('Setting up encoder...')
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    ref_image = encoder.encode_latents(torch.as_tensor(ref_image).to(device).unsqueeze(0))
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=ref_image.shape[-1], img_channels=ref_image.shape[1], label_dim=ref_label.shape[-1])
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    net.train().requires_grad_(True).to(device)

    # Optional warm-start initialization.
    if init_from is not None:
        dist.print0(f'Initializing network from {init_from} ...')
        with dnnlib.util.open_url(init_from, verbose=(dist.get_rank() == 0)) as f:
            init_data = pickle.load(f)
        src_net = init_data['ema'] if isinstance(init_data, dict) and 'ema' in init_data else init_data
        if transfer_fn is None:
            missing, unexpected = net.load_state_dict(src_net.state_dict(), strict=False)
            dist.print0(f'Loaded with strict=False; missing={len(missing)}, unexpected={len(unexpected)}')
        else:
            copied, skipped = dnnlib.util.call_func_by_name(
                func_name=transfer_fn,
                trigflow_net=net,
                edm2_net=src_net,
            )
            dist.print0(f'Transfer complete; copied={len(copied)}, skipped={len(skipped)}')

    teacher_net = None
    if is_scm_mode:
        # Backbone-specific init helpers.
        if 'DDPMPP' in type(net).__name__:
            from training.networks_trigflow_ddpmpp import (
                reset_logvar_linear, apply_resolution_dropout, init_segment_embedding_zero,
            )
        else:
            from training.networks_trigflow import (
                reset_logvar_linear, apply_resolution_dropout, init_segment_embedding_zero,
            )

        dist.print0(f'Loading sCM teacher from {teacher_pkl}...')
        if dist.get_rank() != 0:
            torch.distributed.barrier()
        with dnnlib.util.open_url(teacher_pkl, verbose=(dist.get_rank() == 0)) as f:
            teacher_data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier()
        teacher_net = teacher_data['ema'].eval().requires_grad_(False).to(device)

        teacher_encoder = teacher_data.get('encoder', None)
        if teacher_encoder is not None:
            teacher_encoder_cls = type(teacher_encoder).__name__
            student_encoder_cls = encoder_kwargs['class_name'].split('.')[-1]
            if teacher_encoder_cls != student_encoder_cls:
                raise RuntimeError(
                    f'Encoder mismatch: teacher uses {teacher_encoder_cls!r} but '
                    f'student config specifies {student_encoder_cls!r}.'
                )

        try:
            misc.copy_params_and_buffers(src_module=teacher_net, dst_module=net, require_all=False)
            dist.print0('[sCM INIT] Seeded student from teacher EMA.')
        except Exception as err:
            dist.print0(f'[sCM INIT] Could not seed from teacher: {err}')

        reset_logvar_linear(net)
        dist.print0('[sCM INIT] Reset student logvar_linear for CM adaptive weighting.')

        init_segment_embedding_zero(net)
        dist.print0('[sCM INIT] Zero-initialized MS-sCD segment-bottom embedding.')

        # sCT uses sct_dropout, sCD the network's dropout. Dropout is applied only at
        # res <= 16 on ImageNet-64 and at all resolutions on CIFAR-10.
        scm_mode = str(loss_kwargs.get('mode', 'scd')).lower()
        if scm_mode == 'sct':
            drop_p = float(loss_kwargs.get('sct_dropout', 0.45))
        else:
            drop_p = float(network_kwargs.get('dropout', 0.0))
        if drop_p > 0:
            img_res = int(getattr(net, 'img_resolution', 64))
            max_res_dropout = 16 if img_res >= 64 else img_res
            apply_resolution_dropout(net, dropout=drop_p, max_resolution=max_res_dropout)
            unet = getattr(net, 'unet', None)
            configured = []
            if unet is not None:
                for part_name in ('enc', 'dec'):
                    part = getattr(unet, part_name, None)
                    if part is None:
                        continue
                    for block_key, block in part.items():
                        dout_res = tuple(getattr(block, 'dout_resolutions', ()) or ())
                        configured.append((part_name, block_key, float(getattr(block, 'dropout', 0.0)), dout_res))
            active = [(p, k, d, r) for (p, k, d, r) in configured if d > 0 and len(r) > 0]
            if not active:
                raise RuntimeError(
                    f'{scm_mode} dropout configuration failed: no active dropout blocks after '
                    f'apply_resolution_dropout(dropout={drop_p}).'
                )
            dist.print0(f'[sCM INIT] Applied {scm_mode} resolution-conditioned dropout '
                        f'({drop_p * 100:.0f}% at res<={max_res_dropout}).')
        else:
            dist.print0(f'[sCM INIT] Dropout disabled (rate=0) for {scm_mode}.')

        sample_block = next(iter(getattr(getattr(net, 'unet', net), 'dec', {}).values()), None)
        if sample_block is not None and hasattr(sample_block, 'emb_gain'):
            raise RuntimeError('sCM training requires TrigFlow blocks (no emb_gain).')

        # Identical start on all ranks; the manual grad all-reduce keeps them in sync.
        misc.sync_params_and_optimizer_from_rank0(net)
        dist.print0('[sCM INIT] Broadcast rank-0 params/buffers to all ranks (consistent start).')

        del teacher_data

    # Print network summary.
    if dist.get_rank() == 0:
        misc.print_module_summary(net, [
            torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device),
            torch.ones([batch_gpu], device=device),
            torch.zeros([batch_gpu, net.label_dim], device=device),
        ], max_nesting=2)

    # Setup training state.
    dist.print0('Setting up training state...')
    state = dnnlib.EasyDict(cur_nimg=0, total_elapsed_time=0)
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])

    if is_scm_mode:
        scm_class = str(loss_kwargs.get('class_name', '')).split('.')[-1]
        scm_loss_kwargs = {k: v for k, v in loss_kwargs.items() if k not in ('class_name', 'sct_dropout')}
        if scm_class == 'TrigFlowMSSCDLoss':
            from training.loss_ms_scm import TrigFlowMSSCDLoss

            loss_fn = TrigFlowMSSCDLoss(
                teacher_net=teacher_net,
                teacher_pkl_path=teacher_pkl,
                **scm_loss_kwargs,
            )
            dist.print0(
                f'[sCM INIT] MS-sCD loss: M={loss_fn.num_segments}, '
                f'schedule={loss_fn.boundary_schedule}, proposal={loss_fn.proposal_mode}, '
                f'mode={loss_fn.mode}.'
            )
        else:
            from training.loss_scm import TrigFlowSCMLoss

            loss_fn = TrigFlowSCMLoss(
                teacher_net=teacher_net,
                teacher_pkl_path=teacher_pkl,
                **scm_loss_kwargs,
            )
    else:
        loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs)
    ema = dnnlib.util.construct_class_by_name(net=net, **ema_kwargs) if ema_kwargs is not None else None

    # Traditional EMA for FID validation and -val snapshots, separate from phEMA.
    # rampup_ratio = ln(2) / (std_to_exp(sigma) + 1) matches the phEMA(sigma) half-life.
    ema_val = None
    ema_val_state = dnnlib.EasyDict(
        enabled=False,
        target_phema_std=None,
        rampup_ratio=None,
        halflife_nimg_cap=float('inf'),
    )
    if validation_kwargs is not None and validation_kwargs.get('enabled', False):
        from training.phema import std_to_exp
        ema_val_state.enabled = True
        ema_val_state.target_phema_std = float(validation_kwargs.get('val_phema_std', 0.075))
        _user_rampup = validation_kwargs.get('val_ema_rampup_ratio', None)
        if _user_rampup is None:
            _gamma = float(std_to_exp(ema_val_state.target_phema_std))
            ema_val_state.rampup_ratio = float(np.log(2.0) / (_gamma + 1.0))
        else:
            ema_val_state.rampup_ratio = float(_user_rampup)
        _hl_kimg = validation_kwargs.get('val_ema_halflife_kimg', None)
        ema_val_state.halflife_nimg_cap = (
            float('inf') if _hl_kimg is None else float(_hl_kimg) * 1000.0
        )
        ema_val = copy.deepcopy(net).eval().requires_grad_(False).to(device)
        dist.print0(
            f'[VAL EMA] Traditional EMA tracking phEMA std={ema_val_state.target_phema_std:.4f} '
            f'(rampup_ratio={ema_val_state.rampup_ratio:.6f}, halflife_cap_kimg='
            f'{"inf" if not np.isfinite(ema_val_state.halflife_nimg_cap) else int(ema_val_state.halflife_nimg_cap / 1000)})'
        )

    # Optional W&B setup (rank 0 only).
    if wandb_kwargs is not None and dist.get_rank() == 0:
        try:
            _ensure_wandb_tty_streams()
            os.environ.setdefault('WANDB_CONSOLE', 'off')
            os.environ.setdefault('WANDB_SILENT', 'true')
            import wandb
            _wandb_kwargs = dict(wandb_kwargs)
            _wandb_kwargs.setdefault('project', 'edm2-trigflow-teacher')
            _wandb_kwargs.setdefault('name', os.path.basename(run_dir.rstrip('/')) or 'trigflow-teacher')
            _wandb_kwargs.setdefault('dir', run_dir)
            _wandb_kwargs.setdefault('config', {})
            _wandb_kwargs['config'].update({
                'batch_size': batch_size,
                'total_nimg': total_nimg,
                'network': network_kwargs.get('class_name', str(network_kwargs)),
                'loss': loss_kwargs.get('class_name', str(loss_kwargs)),
                'lr_fn': lr_kwargs.get('func_name', str(lr_kwargs)),
                'optimizer': optimizer_kwargs.get('class_name', str(optimizer_kwargs)),
            })
            if 'settings' not in _wandb_kwargs or _wandb_kwargs['settings'] is None:
                try:
                    _wandb_kwargs['settings'] = wandb.Settings(console='off')
                except Exception:
                    pass
            wandb_run = wandb.init(**_wandb_kwargs)
            dist.print0(f'W&B enabled: project={_wandb_kwargs.get("project")} run={_wandb_kwargs.get("name")}')
        except Exception as err:
            dist.print0(f'Warning: failed to initialize W&B ({err}); continuing without W&B.')
            wandb_run = None

    # Load previous checkpoint and decide how long to train.
    _ckpt_objs = dict(state=state, net=net, loss_fn=loss_fn, optimizer=optimizer, ema=ema)
    if ema_val is not None:
        _ckpt_objs['ema_val'] = ema_val
    checkpoint = dist.CheckpointIO(**_ckpt_objs)
    checkpoint.load_latest(run_dir)
    if is_scm_mode and hasattr(loss_fn, 'reload_teacher') and loss_fn.teacher_net is None:
        loss_fn.reload_teacher(device)
        dist.print0('[sCM RESUME] Re-attached teacher after checkpoint load.')
    stop_at_nimg = total_nimg
    if slice_nimg is not None:
        granularity = checkpoint_nimg if checkpoint_nimg is not None else snapshot_nimg if snapshot_nimg is not None else batch_size
        slice_end_nimg = (state.cur_nimg + slice_nimg) // granularity * granularity # round down
        stop_at_nimg = min(stop_at_nimg, slice_end_nimg)
    assert stop_at_nimg > state.cur_nimg
    dist.print0(f'Training from {state.cur_nimg // 1000} kimg to {stop_at_nimg // 1000} kimg:')
    dist.print0()

    # Main training loop.
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed, start_idx=state.cur_nimg)
    dataset_iterator = iter(dnnlib.util.construct_class_by_name(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))
    prev_status_nimg = state.cur_nimg
    cumulative_training_time = 0
    start_nimg = state.cur_nimg
    stats_jsonl = None
    while True:
        done = (state.cur_nimg >= stop_at_nimg)

        # Report status.
        if status_nimg is not None and (done or state.cur_nimg % status_nimg == 0) and (state.cur_nimg != start_nimg or start_nimg == 0):
            cur_time = time.time()
            state.total_elapsed_time += cur_time - prev_status_time
            cur_process = psutil.Process(os.getpid())
            cpu_memory_usage = sum(p.memory_info().rss for p in [cur_process] + cur_process.children(recursive=True))
            dist.print0(' '.join(['Status:',
                'kimg',         f"{training_stats.report0('Progress/kimg',                              state.cur_nimg / 1e3):<9.1f}",
                'time',         f"{dnnlib.util.format_time(training_stats.report0('Timing/total_sec',   state.total_elapsed_time)):<12s}",
                'sec/tick',     f"{training_stats.report0('Timing/sec_per_tick',                        cur_time - prev_status_time):<8.2f}",
                'sec/kimg',     f"{training_stats.report0('Timing/sec_per_kimg',                        cumulative_training_time / max(state.cur_nimg - prev_status_nimg, 1) * 1e3):<7.3f}",
                'maintenance',  f"{training_stats.report0('Timing/maintenance_sec',                     cur_time - prev_status_time - cumulative_training_time):<7.2f}",
                'cpumem',       f"{training_stats.report0('Resources/cpu_mem_gb',                       cpu_memory_usage / 2**30):<6.2f}",
                'gpumem',       f"{training_stats.report0('Resources/peak_gpu_mem_gb',                  torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}",
                'reserved',     f"{training_stats.report0('Resources/peak_gpu_mem_reserved_gb',         torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}",
            ]))
            cumulative_training_time = 0
            prev_status_nimg = state.cur_nimg
            prev_status_time = cur_time
            torch.cuda.reset_peak_memory_stats()

            # Flush training stats.
            training_stats.default_collector.update()
            if dist.get_rank() == 0:
                if stats_jsonl is None:
                    stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
                fmt = {'Progress/tick': '%.0f', 'Progress/kimg': '%.3f', 'timestamp': '%.3f'}
                collector = training_stats.default_collector.as_dict()
                items = [(name, value.mean) for name, value in collector.items()] + [('timestamp', time.time())]
                items = [f'"{name}": ' + (fmt.get(name, '%g') % value if np.isfinite(value) else 'NaN') for name, value in items]
                stats_jsonl.write('{' + ', '.join(items) + '}\n')
                stats_jsonl.flush()

                # Optional W&B logging.
                if wandb_run is not None:
                    try:
                        wandb_log = {name: float(value.mean) for name, value in collector.items() if np.isfinite(value.mean)}
                        wandb_log['Progress/kimg'] = state.cur_nimg / 1e3
                        wandb_log['progress_kimg'] = state.cur_nimg / 1e3
                        wandb_run.log(wandb_log, step=int(state.cur_nimg), commit=True)
                    except Exception as _e:
                        dist.print0(f'[W&B] log failed: {_e}')

            # Update progress and check for abort.
            dist.update_progress(state.cur_nimg // 1000, stop_at_nimg // 1000)
            if state.cur_nimg == stop_at_nimg and state.cur_nimg < total_nimg:
                dist.request_suspend()
            if dist.should_stop() or dist.should_suspend():
                done = True

        # Determine snapshot boundary (used for both PKL save and FID validation).
        is_snap_boundary = (
            snapshot_nimg is not None
            and state.cur_nimg % snapshot_nimg == 0
            and (state.cur_nimg != start_nimg or start_nimg == 0)
        )

        # Save network snapshot (rank 0 only).
        if is_snap_boundary and dist.get_rank() == 0:
            ema_list = ema.get() if ema is not None else optimizer.get_ema(net) if hasattr(optimizer, 'get_ema') else net
            ema_list = ema_list if isinstance(ema_list, list) else [(ema_list, '')]
            # Also save the validation EMA.
            if ema_val is not None:
                ema_list = list(ema_list) + [(ema_val, '-val')]
            for ema_net, ema_suffix in ema_list:
                data = dnnlib.EasyDict(encoder=encoder, dataset_kwargs=dataset_kwargs, loss_fn=loss_fn)
                data.ema = copy.deepcopy(ema_net).cpu().eval().requires_grad_(False).to(torch.float16)
                fname = f'network-snapshot-{state.cur_nimg//1000:07d}{ema_suffix}.pkl'
                dist.print0(f'Saving {fname} ... ', end='', flush=True)
                with open(os.path.join(run_dir, fname), 'wb') as f:
                    pickle.dump(data, f)
                dist.print0('done')
                del data # conserve memory

        # FID validation at snapshot boundaries (all ranks).
        if is_snap_boundary and validation_kwargs is not None and validation_kwargs.get('enabled', False):
            try:
                from validation import maybe_validate
                # Prefer the validation EMA, else the first phEMA.
                if ema_val is not None:
                    val_net = ema_val
                elif ema is not None:
                    ema_list = ema.get()
                    val_net = ema_list[0][0] if isinstance(ema_list, list) and len(ema_list) > 0 else net
                else:
                    val_net = net
                maybe_validate(
                    cur_nimg=state.cur_nimg,
                    snapshot_nimg=snapshot_nimg,
                    net_ema=val_net,
                    encoder=encoder,
                    run_dir=run_dir,
                    dataset_kwargs=dataset_kwargs,
                    validation_kwargs=validation_kwargs,
                    wandb_run=wandb_run,
                )
            except Exception as _e:
                # No barrier: other ranks may still be inside validation collectives.
                dist.print0(f'[VAL] FID validation failed: {_e}')
            net.train()

        # Save state checkpoint.
        if checkpoint_nimg is not None and (done or state.cur_nimg % checkpoint_nimg == 0) and state.cur_nimg != start_nimg:
            checkpoint.save(os.path.join(run_dir, f'training-state-{state.cur_nimg//1000:07d}.pt'))
            misc.check_ddp_consistency(net)

            # Prune old checkpoints.
            if int(checkpoint_keep_recent) > 0 and dist.get_rank() == 0:
                _cleanup_checkpoint_artifacts(
                    run_dir,
                    keep_recent=max(1, int(checkpoint_keep_recent)),
                    cleanup_snapshots=bool(checkpoint_cleanup_snapshots),
                )
            torch.distributed.barrier()

        # Done?
        if done:
            break

        # Evaluate loss and accumulate gradients.
        batch_start_time = time.time()
        misc.set_random_seed(seed, dist.get_rank(), state.cur_nimg)
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = encoder.encode_latents(images.to(device))
                if hasattr(loss_fn, 'cur_iter'):
                    loss_fn.cur_iter = int(state.cur_nimg // batch_size)
                loss = loss_fn(net=ddp, images=images, labels=labels.to(device))
                training_stats.report('Loss/loss', loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        # Run optimizer and update weights.
        lr = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nimg, batch_size=batch_size, **lr_kwargs)
        training_stats.report('Loss/learning_rate', lr)
        for g in optimizer.param_groups:
            g['lr'] = lr

        # Gradient norm and NaN/Inf count before sanitization, accumulated on-device.
        grad_sq_sum = torch.zeros([], device=device)
        num_nan_inf_grads = torch.zeros([], device=device)
        for param in net.parameters():
            if param.grad is not None:
                g = param.grad.float()
                is_bad = (~torch.isfinite(g)).any()
                num_nan_inf_grads += is_bad
                grad_sq_sum += torch.where(is_bad, torch.zeros_like(grad_sq_sum), g.square().sum())
        grad_norm_pre = grad_sq_sum.sqrt()
        training_stats.report('Loss/grad_norm', grad_norm_pre)
        training_stats.report('Loss/nan_inf_grad_count', num_nan_inf_grads)

        # Sum over ranks so NaN/Inf grads on any rank get reported.
        nan_inf_count = num_nan_inf_grads.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            torch.distributed.all_reduce(nan_inf_count, op=torch.distributed.ReduceOp.SUM)
        nan_inf_count = int(nan_inf_count.item())
        if nan_inf_count > 0:
            dist.print0(f'[WARN] kimg {state.cur_nimg / 1e3:.1f}: {nan_inf_count} param(s) had NaN/Inf gradients '
                        f'(zeroed by force_finite={force_finite}).')

        if force_finite:
            for param in net.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
        # The JVP loss bypasses DDP hooks, so average grads manually (after force_finite).
        if is_scm_mode:
            misc.allreduce_grads_mean(net)

        optimizer.step()

        # Report |out_gain|.
        try:
            with torch.no_grad():
                _out_gain = getattr(getattr(net, 'unet', None), 'out_gain', None)
                if isinstance(_out_gain, torch.nn.Parameter):
                    training_stats.report('Params/out_gain_abs', _out_gain.detach().abs().to(torch.float32))
        except Exception:
            pass

        # Update EMA and training state.
        state.cur_nimg += batch_size
        if ema is not None:
            ema.update(cur_nimg=state.cur_nimg, batch_size=batch_size)

        # Update validation EMA: beta = 0.5^(batch_size / halflife_nimg).
        if ema_val is not None:
            halflife_nimg = ema_val_state.halflife_nimg_cap
            if ema_val_state.rampup_ratio is not None and ema_val_state.rampup_ratio > 0:
                halflife_nimg = min(halflife_nimg, state.cur_nimg * ema_val_state.rampup_ratio)
            ema_beta = 0.5 ** (batch_size / max(halflife_nimg, 1e-8))
            training_stats.report('EMA/val_halflife_kimg', halflife_nimg / 1000.0)
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
