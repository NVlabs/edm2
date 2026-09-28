# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modifications copyright (c) 2026, Vinay Saji Mathew.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Distill an EDM or EDM2 teacher into a few-step student with Multistep
Consistency Distillation (Heek et al., 2024). The student is initialized from
the teacher, so both must use the same backbone."""

import os
import re
import json
import warnings
import click
import torch
import dnnlib
from torch_utils import distributed as dist
import training.distillation.consistency.training_loop_cd as training_loop_cd

warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`')

#----------------------------------------------------------------------------
# Configuration presets. `backbone` selects the network family; remaining keys
# are MSCD/optimization defaults that the CLI can override.

config_presets = {
    'mscd-img64-edm': dnnlib.EasyDict(
        backbone='edm', duration=200<<20, batch=512, channels=192,
        lr=2e-4, decay=35000, dropout=0.0, fp16=True,
        t_start=256, t_end=1024, t_anneal_kimg=750, sampling_mode='vp',
        terminal_teacher_hop=False, dout_resolutions=None,
        ema_halflife_kimg=500.0, ema_rampup=0.05,
    ),
    # Settings of the released EDM2-S MSCD students (teacher: edm2-img64-s-1073741-0.075.pkl).
    'mscd-img64-edm2-s': dnnlib.EasyDict(
        backbone='edm2', duration=200<<20, batch=2048, channels=192,
        lr=5e-4, decay=0, dropout=0.1, fp16=True,
        t_start=64, t_end=1280, t_anneal_kimg=104800, sampling_mode='edm',
        terminal_teacher_hop=True, dout_resolutions=[16, 8],
        ema_halflife_kimg=0.0, ema_rampup=0,
    ),
}

LR_FUNC = 'training.distillation.consistency.training_loop_cd.learning_rate_schedule'

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

def parse_float_list(s):
    """Parse a comma-separated list of floats, e.g. '1.1' or '2.5,1.1'."""
    if s is None:
        return None
    if isinstance(s, (list, tuple)):
        return [float(x) for x in s]
    return [float(x.strip()) for x in s.split(',') if x.strip()]

#----------------------------------------------------------------------------

def setup_training_config(preset='mscd-img64-edm', **opts):
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
    backbone = pre['backbone']

    if not opts.get('teacher'):
        raise click.ClickException('--teacher is required (MSCD always distills from a teacher).')

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

    # Encoder.
    if dataset_channels == 3:
        c.encoder_kwargs = dnnlib.EasyDict(class_name='training.encoders.StandardRGBEncoder')
    elif dataset_channels == 8:
        c.encoder_kwargs = dnnlib.EasyDict(class_name='training.encoders.StabilityVAEEncoder')
    else:
        raise click.ClickException(f'--data: Unsupported channel count {dataset_channels}')

    c.update(total_nimg=opts.duration, batch_size=opts.batch)

    # Network: backbone-specific.
    if backbone == 'edm':
        c.network_kwargs = dnnlib.EasyDict(
            class_name='training.networks_edm.EDMPrecond',
            model_type='DhariwalUNet',
            model_channels=opts.channels,
            channel_mult=[1, 2, 3, 4],
            dropout=opts.dropout,
        )
    else:
        assert backbone == 'edm2'
        c.network_kwargs = dnnlib.EasyDict(
            class_name='training.networks_edm2.Precond',
            model_channels=opts.channels,
            dropout=opts.dropout,
        )
    dout_res = opts.get('dout_resolutions')
    if dout_res is not None:
        c.network_kwargs.dout_resolutions = dout_res

    c.lr_kwargs = dnnlib.EasyDict(func_name=LR_FUNC, ref_lr=opts.lr, ref_batches=opts.decay)

    # Performance-related.
    c.batch_gpu = opts.get('batch_gpu', 0) or None
    c.network_kwargs.use_fp16 = bool(opts.get('fp16'))
    c.loss_scaling = opts.get('ls', 1)
    c.cudnn_benchmark = opts.get('bench', True)

    workers = opts.get('workers', 2)
    c.data_loader_kwargs = dnnlib.EasyDict(
        class_name='torch.utils.data.DataLoader', pin_memory=True,
        num_workers=workers, prefetch_factor=2 if workers > 0 else None,
    )

    # I/O.
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

    # MSCD core. Click lowercases option names (--S -> s, --T_start -> t_start).
    c.teacher_pkl = opts['teacher']
    cd_S = opts.get('s', 8)
    student_sigma_mids = opts.get('student_sigma_mids', None)
    if student_sigma_mids is not None:
        assert len(student_sigma_mids) == cd_S - 1, (
            f"--student_sigma_mids has {len(student_sigma_mids)} values "
            f"but --S={cd_S} requires exactly {cd_S - 1} interior knots"
        )
    c.cd_kwargs = dict(
        S=cd_S,
        T_start=opts.t_start,
        T_end=opts.t_end,
        T_anneal_kimg=opts.t_anneal_kimg,
        rho=opts.get('rho', 7.0),
        sigma_min=opts.get('sigma_min', 0.002),
        sigma_max=opts.get('sigma_max', 80.0),
        loss_type=opts.get('cd_loss', 'pseudo_huber'),
        weight_mode=opts.get('cd_weight_mode', 'sqrt_karras'),
        sigma_data=0.5,
        sampling_mode=opts.sampling_mode,
        terminal_anchor=opts.get('terminal_anchor', True),
        terminal_teacher_hop=opts.terminal_teacher_hop,
        sync_dropout=opts.get('sync_dropout', True),
        student_sigma_mids=student_sigma_mids,
    )
    c.ema_halflife_kimg = opts.ema_halflife_kimg
    c.ema_rampup_ratio = opts.ema_rampup or None
    if opts.get('cd_lr') is not None:
        c.lr_kwargs['ref_lr'] = opts['cd_lr']
    if opts.get('cd_decay') is not None:
        c.lr_kwargs['ref_batches'] = opts['cd_decay']

    # In-training FID with the Euler sampler.
    val_ref = opts.get('val_ref')
    if val_ref is not None:
        val_step_sigmas = None
        if student_sigma_mids is not None:
            sigma_max_val = opts.get('sigma_max', 80.0)
            val_step_sigmas = [sigma_max_val] + list(student_sigma_mids) + [0.0]
        c.validation_kwargs = dnnlib.EasyDict(
            enabled=True,
            val_mode='cd',
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
            step_sigmas=val_step_sigmas,
        )

    # Weights & Biases.
    if opts.get('wandb', False):
        tags = opts.get('wandb_tags')
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',') if t.strip()]
        c.wandb_kwargs = dnnlib.EasyDict(
            enabled=True,
            project=opts.get('wandb_project', 'mscd'),
            entity=opts.get('wandb_entity', None),
            name=opts.get('wandb_run', None),
            tags=tags,
            mode=opts.get('wandb_mode', 'online'),
        )

    return c

#----------------------------------------------------------------------------

def print_training_config(run_dir, c):
    dist.print0()
    dist.print0('MSCD training config:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Backbone:                {c.network_kwargs.class_name}')
    dist.print0(f'Teacher:                 {c.teacher_pkl}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0(f'CD params:               S={c.cd_kwargs["S"]}, T={c.cd_kwargs["T_start"]}->{c.cd_kwargs["T_end"]}, '
                f'loss={c.cd_kwargs["loss_type"]}/{c.cd_kwargs["weight_mode"]}')
    if c.get('validation_kwargs') and c.validation_kwargs.get('enabled'):
        vk = c.validation_kwargs
        dist.print0(f'Validation:              FID every {vk.get("every",1)} snapshot(s), {vk.get("num_images",50000)} images, '
                    f'{vk.get("steps",8)} Euler steps')
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
    training_loop_cd.training_loop(run_dir=run_dir, **c)

#----------------------------------------------------------------------------

@click.command()
# Main options.
@click.option('--outdir',           help='Where to save the results', metavar='DIR',            type=str, required=True)
@click.option('--data',             help='Path to the dataset', metavar='ZIP|DIR',              type=str, required=True)
@click.option('--cond',             help='Train class-conditional model', metavar='BOOL',       type=bool, default=True, show_default=True)
@click.option('--preset',           help='Configuration preset', metavar='STR',                 type=click.Choice(list(config_presets.keys())), default='mscd-img64-edm', show_default=True)

# Hyperparameters.
@click.option('--duration',         help='Training duration', metavar='NIMG',                   type=parse_nimg, default=None)
@click.option('--batch',            help='Total batch size', metavar='NIMG',                    type=parse_nimg, default=None)
@click.option('--channels',         help='Channel multiplier', metavar='INT',                   type=click.IntRange(min=64), default=None)
@click.option('--dropout',          help='Dropout probability', metavar='FLOAT',                type=click.FloatRange(min=0, max=1), default=None)
@click.option('--lr',               help='Learning rate max. (alpha_ref)', metavar='FLOAT',     type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--decay',            help='Learning rate decay (t_ref)', metavar='BATCHES',      type=click.FloatRange(min=0), default=None)

# Performance.
@click.option('--batch-gpu',        help='Limit batch size per GPU', metavar='NIMG',            type=parse_nimg, default=0, show_default=True)
@click.option('--fp16',             help='Enable mixed-precision training', metavar='BOOL',     type=bool, default=None)
@click.option('--ls',               help='Loss scaling', metavar='FLOAT',                       type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',            help='Enable cuDNN benchmarking', metavar='BOOL',           type=bool, default=True, show_default=True)
@click.option('--workers',          help='DataLoader worker processes', metavar='INT',          type=click.IntRange(min=0), default=2, show_default=True)

# I/O.
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

# Teacher / MSCD core.
@click.option('--teacher',          help='Teacher pickle (required)', metavar='PKL|URL',        type=str, required=True)
@click.option('--S',                help='Student step count', type=click.IntRange(min=2), default=8, show_default=True)
@click.option('--T_start',          help='Initial teacher edges', type=click.IntRange(min=2), default=None)
@click.option('--T_end',            help='Final teacher edges', type=click.IntRange(min=2), default=None)
@click.option('--T_anneal_kimg',    help='Teacher edge annealing horizon (kimg)', type=click.IntRange(min=0), default=None)
@click.option('--rho',              help='Karras rho exponent', type=click.FloatRange(min=0, min_open=True), default=7.0, show_default=True)
@click.option('--cd_loss',          help='CD loss type', type=click.Choice(['huber','l2','l2_root','pseudo_huber']), default='pseudo_huber', show_default=True)
@click.option('--cd_weight_mode',   help='CD loss weight mode', type=click.Choice(['edm','sqrt_karras','flat','snr','karras','uniform']), default='sqrt_karras', show_default=True)
@click.option('--sampling_mode',    help='Edge sampling distribution', type=click.Choice(['uniform','vp','edm']), default=None)
@click.option('--terminal_anchor/--no_terminal_anchor', help='Anchor terminal edge to 1/T probability', default=True, show_default=True)
@click.option('--terminal_teacher_hop/--no_terminal_teacher_hop', help='Use teacher hop for terminal edge', default=None)
@click.option('--sigma_min',        help='Min sigma for CD Karras grids', type=float, default=0.002, show_default=True)
@click.option('--sigma_max',        help='Max sigma for CD Karras grids', type=float, default=80.0, show_default=True)
@click.option('--student_sigma_mids', help='Interior student sigmas, descending (e.g. 1.1); replaces the Karras student grid', type=parse_float_list, default=None)
@click.option('--sync_dropout/--no_sync_dropout', help='Sync CUDA RNG for dropout', default=True, show_default=True)
@click.option('--dout_resolutions', help='Apply dropout only at these resolutions (e.g. 16,8). None = all.', type=parse_int_list, default=None)
@click.option('--cd_lr',            help='CD-mode ref_lr override', type=float, default=None)
@click.option('--cd_decay',         help='CD-mode ref_batches override', type=float, default=None)

# Validation EMA.
@click.option('--ema_halflife_kimg', help='Halflife of exponential validation EMA (kimg), 0 = live weights', type=float, default=None)
@click.option('--ema_rampup',       help='EMA rampup ratio (0 = no rampup)', type=float, default=None)

# FID validation.
@click.option('--val_ref',          help='FID reference stats (.npz or .pkl/URL)', metavar='NPZ|PKL', type=str, default=None)
@click.option('--val_every',        help='Validate every N snapshots', type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--val_num',          help='Images for FID evaluation', type=int, default=50000, show_default=True)
@click.option('--val_steps',        help='Euler sampler steps for validation (None = S)', type=int, default=None)
@click.option('--val_seed',         help='Validation base seed', type=int, default=0, show_default=True)
@click.option('--val_batch',        help='Validation batch size per GPU', type=int, default=64, show_default=True)
@click.option('--val_at_start',     help='Run validation at first snapshot', metavar='BOOL', type=bool, default=False, show_default=True)

# Weights & Biases.
@click.option('--wandb',            help='Enable W&B logging', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--wandb_project',    help='W&B project name', type=str, default='mscd', show_default=True)
@click.option('--wandb_entity',     help='W&B entity (user/team)', type=str, default=None)
@click.option('--wandb_run',        help='W&B run name', type=str, default=None)
@click.option('--wandb_tags',       help='W&B tags (comma-separated)', type=str, default=None)
@click.option('--wandb_mode',       help='W&B mode', type=click.Choice(['online','offline','disabled']), default='online', show_default=True)

def cmdline(outdir, dry_run, nosubdir, desc, **opts):
    """Multistep Consistency Distillation (MSCD) of an EDM/EDM2 diffusion teacher.

    Example (EDM ADM ImageNet-64 teacher, 8 GPUs):

    \b
    torchrun --standalone --nproc_per_node=8 train_mscd.py \\
        --outdir=training-runs --data=datasets/img64.zip \\
        --preset=mscd-img64-edm --teacher=path/to/teacher.pkl \\
        --S=8 --batch-gpu=32 --val_ref=stats.npz
    """
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    dist.print0('Setting up MSCD training config...')
    c = setup_training_config(**opts)

    if nosubdir:
        run_dir = outdir
    else:
        data_name = os.path.splitext(os.path.basename(opts.get('data', 'data')))[0]
        cond_str  = 'cond' if c.dataset_kwargs.get('use_labels', False) else 'uncond'
        dtype_str = 'fp16' if c.network_kwargs.get('use_fp16', False) else 'fp32'
        preset    = opts.get('preset', 'mscd')
        gpus      = dist.get_world_size()
        auto_desc = f'{data_name}-{cond_str}-{preset}-gpus{gpus}-batch{c.batch_size}-{dtype_str}'
        auto_desc += f'-cdS{c.cd_kwargs.get("S", 8)}-T{c.cd_kwargs.get("T_start")}-{c.cd_kwargs.get("T_end")}'
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
