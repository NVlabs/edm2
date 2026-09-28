# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modifications copyright (c) 2026, Vinay Saji Mathew.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Distill an EDM ImageNet-64 teacher into a few-step generator via moment
matching (Salimans et al., 2024)."""

import os
import re
import json
import warnings
import click
import torch
import dnnlib
from torch_utils import distributed as dist
import training.distillation.moment_matching.training_loop_mm as training_loop_mm

warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`')

#----------------------------------------------------------------------------

config_presets = {
    # Settings of the reported ImageNet-64 run (teacher: edm-imagenet-64x64-cond-adm.pkl).
    'mm-img64-edm': dnnlib.EasyDict(
        backbone='edm', duration=410000*1000, batch=2048, channels=192,
        lr=2e-6, dropout=0.0, fp16=True, lr_warmup_kimg=2048,
        ema_halflife_kimg=0.0, ema_rampup=0.05,
    ),
}

#----------------------------------------------------------------------------

def parse_nimg(s):
    if isinstance(s, int):
        return s
    if s.endswith('Ki'):
        return int(s[:-2]) << 10
    if s.endswith('Mi'):
        return int(s[:-2]) << 20
    if s.endswith('Gi'):
        return int(s[:-2]) << 30
    return int(s)

def parse_int_list(s):
    if s is None:
        return None
    if isinstance(s, list):
        return s
    return [int(x.strip()) for x in s.split(',') if x.strip()]

#----------------------------------------------------------------------------

def setup_training_config(preset='mm-img64-edm', **opts):
    opts = dnnlib.EasyDict(opts)
    c = dnnlib.EasyDict()

    if preset not in config_presets:
        raise click.ClickException(f'Invalid configuration preset "{preset}"')
    pre = config_presets[preset]
    for key, value in pre.items():
        if key == 'backbone':
            continue
        if opts.get(key, None) is None:
            opts[key] = value

    if not opts.get('teacher'):
        raise click.ClickException('--teacher is required for moment matching.')

    # Dataset.
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data, use_labels=opts.get('cond', True))
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_channels = dataset_obj.num_channels
        if c.dataset_kwargs.use_labels and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True, but no labels found in the dataset')
        del dataset_obj
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    if dataset_channels == 3:
        c.encoder_kwargs = dnnlib.EasyDict(class_name='training.encoders.StandardRGBEncoder')
    else:
        raise click.ClickException(f'--data: moment matching expects 3-channel RGB, got {dataset_channels}')

    c.update(total_nimg=opts.duration, batch_size=opts.batch)

    # Network (EDM ADM backbone).
    c.network_kwargs = dnnlib.EasyDict(
        class_name='training.networks_edm.EDMPrecond',
        model_type='DhariwalUNet',
        model_channels=opts.channels,
        channel_mult=[1, 2, 3, 4],
        dropout=opts.dropout,
    )
    dout_res = opts.get('dout_resolutions')
    if dout_res is not None:
        c.network_kwargs.dout_resolutions = dout_res
    c.network_kwargs.use_fp16 = bool(opts.get('fp16'))

    # Adam beta1=0, beta2=0.99, eps=1e-12 (Salimans et al., 2024).
    c.optimizer_kwargs = dnnlib.EasyDict(
        class_name='torch.optim.Adam',
        betas=(opts.get('beta1', 0.0), opts.get('beta2', 0.99)),
        eps=opts.get('adam_eps', 1e-12),
    )

    # Performance / I/O.
    c.batch_gpu = opts.get('batch_gpu', 0) or None
    c.loss_scaling = opts.get('ls', 1)
    c.cudnn_benchmark = opts.get('bench', True)
    workers = opts.get('workers', 2)
    c.data_loader_kwargs = dnnlib.EasyDict(
        class_name='torch.utils.data.DataLoader', pin_memory=True,
        num_workers=workers, prefetch_factor=2 if workers > 0 else None,
    )
    c.status_nimg = opts.get('status', 0) or None
    c.snapshot_nimg = opts.get('snapshot', 0) or None
    c.checkpoint_nimg = opts.get('checkpoint', 0) or None
    c.phema_snapshot_nimg = opts.get('phema_snap', 0) or None
    c.checkpoint_keep_recent = int(opts.get('checkpoint_keep_recent', 3))
    c.checkpoint_cleanup_snapshots = not bool(opts.get('no_checkpoint_snapshot_prune', False))
    c.seed = opts.get('seed', 0)

    resume_pt = opts.get('resume')
    if resume_pt is not None:
        if not re.fullmatch(r'training-state-(\d+)\.pt', os.path.basename(resume_pt)):
            raise click.ClickException('--resume must point to a training-state-*.pt file from a previous run')
        if not os.path.isfile(resume_pt):
            raise click.ClickException(f'--resume: file not found: {resume_pt}')
        c.resume_state_dump = resume_pt

    # Moment-matching core + LR schedule.
    c.teacher_pkl = opts['teacher']
    c.mm_kwargs = dict(
        k=opts.get('s', 8),
        sigma_min=opts.get('sigma_min', 0.002),
        sigma_max=opts.get('sigma_max', 80.0),
        rho=opts.get('rho', 7.0),
        sigma_data=0.5,
        weight_mode=opts.get('mm_weight_mode', 'edm'),
        sync_dropout=opts.get('sync_dropout', True),
    )
    c.lr = opts.lr
    c.lr_warmup_kimg = opts.lr_warmup_kimg
    c.lr_anneal = opts.get('lr_anneal', True)
    c.grad_clip = opts.get('grad_clip', 1.0)
    c.ema_halflife_kimg = opts.ema_halflife_kimg
    c.ema_rampup_ratio = opts.ema_rampup or None

    # In-training FID with the ancestral sampler.
    val_ref = opts.get('val_ref')
    if val_ref is not None:
        c.validation_kwargs = dnnlib.EasyDict(
            enabled=True,
            val_mode='mm',
            ref=val_ref,
            every=opts.get('val_every', 1),
            num_images=opts.get('val_num', 50000),
            steps=opts.get('val_steps') or opts.get('s', 8),
            seed=opts.get('val_seed', 0),
            batch=opts.get('val_batch', 64),
            sigma_min=opts.get('sigma_min', 0.002),
            sigma_max=opts.get('sigma_max', 80.0),
            rho=opts.get('rho', 7.0),
            at_start=opts.get('val_at_start', False),
        )

    if opts.get('wandb', False):
        tags = opts.get('wandb_tags')
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',') if t.strip()]
        c.wandb_kwargs = dnnlib.EasyDict(
            enabled=True,
            project=opts.get('wandb_project', 'mm'),
            entity=opts.get('wandb_entity', None),
            name=opts.get('wandb_run', None),
            tags=tags,
            mode=opts.get('wandb_mode', 'online'),
        )

    return c

#----------------------------------------------------------------------------

def print_training_config(run_dir, c):
    dist.print0()
    dist.print0('Moment-matching training config:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Teacher:                 {c.teacher_pkl}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0(f'MM params:               k={c.mm_kwargs["k"]}, weight_mode={c.mm_kwargs["weight_mode"]}')
    dist.print0()

#----------------------------------------------------------------------------

def launch_training(run_dir, c):
    if dist.get_rank() == 0 and not os.path.isdir(run_dir):
        dist.print0('Creating output directory...')
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
    torch.distributed.barrier()
    dnnlib.util.Logger(file_name=os.path.join(run_dir, 'log.txt'), file_mode='a', should_flush=True)
    training_loop_mm.training_loop(run_dir=run_dir, **c)

#----------------------------------------------------------------------------

@click.command()
@click.option('--outdir',           help='Where to save the results', metavar='DIR',            type=str, required=True)
@click.option('--data',             help='Path to the dataset', metavar='ZIP|DIR',              type=str, required=True)
@click.option('--cond',             help='Train class-conditional model', metavar='BOOL',       type=bool, default=True, show_default=True)
@click.option('--preset',           help='Configuration preset', metavar='STR',                 type=click.Choice(list(config_presets.keys())), default='mm-img64-edm', show_default=True)

@click.option('--duration',         help='Training duration', metavar='NIMG',                   type=parse_nimg, default=None)
@click.option('--batch',            help='Total batch size', metavar='NIMG',                    type=parse_nimg, default=None)
@click.option('--channels',         help='Channel multiplier', metavar='INT',                   type=click.IntRange(min=64), default=None)
@click.option('--dropout',          help='Dropout probability', metavar='FLOAT',                type=click.FloatRange(min=0, max=1), default=None)
@click.option('--lr',               help='Peak learning rate', metavar='FLOAT',                 type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--beta1',            help='Adam beta1', metavar='FLOAT',                         type=click.FloatRange(min=0, max=1), default=0.0, show_default=True)
@click.option('--beta2',            help='Adam beta2', metavar='FLOAT',                         type=click.FloatRange(min=0, max=1), default=0.99, show_default=True)
@click.option('--adam_eps',         help='Adam epsilon', metavar='FLOAT',                       type=float, default=1e-12, show_default=True)
@click.option('--lr_warmup_kimg',   help='LR linear warmup horizon (kimg)', type=click.IntRange(min=0), default=None)
@click.option('--lr_anneal',        help='Anneal LR to zero after warmup', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--grad_clip',        help='Gradient clip max norm (0=off)', type=click.FloatRange(min=0), default=1.0, show_default=True)

@click.option('--batch-gpu',        help='Limit batch size per GPU', metavar='NIMG',            type=parse_nimg, default=0, show_default=True)
@click.option('--fp16',             help='Enable mixed-precision training', metavar='BOOL',     type=bool, default=None)
@click.option('--ls',               help='Loss scaling', metavar='FLOAT',                       type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',            help='Enable cuDNN benchmarking', metavar='BOOL',           type=bool, default=True, show_default=True)
@click.option('--workers',          help='DataLoader worker processes', metavar='INT',          type=click.IntRange(min=0), default=2, show_default=True)

@click.option('--status',           help='Interval of status prints', metavar='NIMG',           type=parse_nimg, default='128Ki', show_default=True)
@click.option('--snapshot',         help='Interval of network snapshots', metavar='NIMG',       type=parse_nimg, default='8Mi', show_default=True)
@click.option('--phema_snap',       help='Interval of phEMA snapshots (default: same as --snapshot)', metavar='NIMG', type=parse_nimg, default=None)
@click.option('--checkpoint',       help='Interval of training checkpoints', metavar='NIMG',    type=parse_nimg, default='128Mi', show_default=True)
@click.option('--checkpoint_keep_recent', help='Retain N newest training-state .pt plus best-FID .pt', type=click.IntRange(min=1), default=3, show_default=True)
@click.option('--no_checkpoint_snapshot_prune', help='Keep all primary network-snapshot-{kimg}.pkl', is_flag=True, default=False)
@click.option('--seed',             help='Random seed', metavar='INT',                          type=int, default=0, show_default=True)
@click.option('--resume',           help='Resume from training-state-*.pt', metavar='PT',       type=str, default=None)
@click.option('--nosubdir',         help='Do not create a numbered subdirectory inside --outdir', is_flag=True)
@click.option('--desc',             help='String to include in the output directory name',      metavar='STR', type=str, default=None)
@click.option('-n', '--dry-run',    help='Print training options and exit',                     is_flag=True)

# Moment-matching core.
@click.option('--teacher',          help='Teacher pickle (required)', metavar='PKL|URL',        type=str, required=True)
@click.option('--S',                help='Student step count (k)', type=click.IntRange(min=1), default=8, show_default=True)
@click.option('--rho',              help='Karras rho exponent', type=click.FloatRange(min=0, min_open=True), default=7.0, show_default=True)
@click.option('--sigma_min',        help='Min sigma', type=click.FloatRange(min=0, min_open=True), default=0.002, show_default=True)
@click.option('--sigma_max',        help='Max sigma', type=click.FloatRange(min=0, min_open=True), default=80.0, show_default=True)
@click.option('--mm_weight_mode',   help='MM weight mode', type=click.Choice(['edm','vlike','flat','snr','snr+1','karras','sqrt_karras','truncated-snr','uniform']), default='edm', show_default=True)
@click.option('--sync_dropout/--no_sync_dropout', help='Sync CUDA RNG for dropout between g_phi and g_theta', default=True, show_default=True)
@click.option('--dout_resolutions', help='Apply dropout only at these resolutions (e.g. 16,8). None = all.', type=parse_int_list, default=None)

# Validation EMA.
@click.option('--ema_halflife_kimg', help='Halflife of exponential validation EMA (kimg), 0 = live weights', type=float, default=None)
@click.option('--ema_rampup',       help='EMA rampup ratio (0 = no rampup)', type=float, default=None)

# FID validation.
@click.option('--val_ref',          help='FID reference stats (.npz or .pkl/URL)', metavar='NPZ|PKL', type=str, default=None)
@click.option('--val_every',        help='Validate every N snapshots', type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--val_num',          help='Images for FID evaluation', type=int, default=50000, show_default=True)
@click.option('--val_steps',        help='Ancestral sampler steps for validation (None = k)', type=int, default=None)
@click.option('--val_seed',         help='Validation base seed', type=int, default=0, show_default=True)
@click.option('--val_batch',        help='Validation batch size per GPU', type=int, default=64, show_default=True)
@click.option('--val_at_start',     help='Run validation at first snapshot', metavar='BOOL', type=bool, default=False, show_default=True)

# Weights & Biases.
@click.option('--wandb',            help='Enable W&B logging', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--wandb_project',    help='W&B project name', type=str, default='mm', show_default=True)
@click.option('--wandb_entity',     help='W&B entity (user/team)', type=str, default=None)
@click.option('--wandb_run',        help='W&B run name', type=str, default=None)
@click.option('--wandb_tags',       help='W&B tags (comma-separated)', type=str, default=None)
@click.option('--wandb_mode',       help='W&B mode', type=click.Choice(['online','offline','disabled']), default='online', show_default=True)

def cmdline(outdir, dry_run, nosubdir, desc, **opts):
    """Moment-matching distillation of an EDM diffusion teacher.

    Example (EDM ADM ImageNet-64 teacher, 8 GPUs):

    \b
    torchrun --standalone --nproc_per_node=8 train_momentmatch.py \\
        --outdir=training-runs --data=datasets/img64.zip \\
        --preset=mm-img64-edm --teacher=path/to/teacher.pkl \\
        --S=8 --batch-gpu=32 --val_ref=stats.npz
    """
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    dist.print0('Setting up moment-matching training config...')
    c = setup_training_config(**opts)

    if nosubdir:
        run_dir = outdir
    else:
        data_name = os.path.splitext(os.path.basename(opts.get('data', 'data')))[0]
        cond_str  = 'cond' if c.dataset_kwargs.get('use_labels', False) else 'uncond'
        dtype_str = 'fp16' if c.network_kwargs.get('use_fp16', False) else 'fp32'
        preset    = opts.get('preset', 'mm')
        gpus      = dist.get_world_size()
        auto_desc = f'{data_name}-{cond_str}-{preset}-gpus{gpus}-batch{c.batch_size}-{dtype_str}-mmK{c.mm_kwargs.get("k", 8)}'
        if desc is not None:
            auto_desc += f'-{desc}'
        if dist.get_rank() == 0:
            prev_run_dirs = []
            if os.path.isdir(outdir):
                prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
            prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
            prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
            cur_run_id   = max(prev_run_ids, default=-1) + 1
            run_dir      = os.path.join(outdir, f'{cur_run_id:05d}-{auto_desc}')
            assert not os.path.exists(run_dir), f'Run directory already exists: {run_dir}'
        else:
            run_dir = None
        run_dir_list = [run_dir]
        torch.distributed.broadcast_object_list(run_dir_list, src=0)
        run_dir = run_dir_list[0]

    print_training_config(run_dir=run_dir, c=c)
    if dry_run:
        dist.print0('Dry run; exiting.')
    else:
        launch_training(run_dir=run_dir, c=c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
