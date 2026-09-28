# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modifications copyright (c) 2026, Vinay Saji Mathew.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train multistep TrigFlow consistency students (MS-sCD) with M segments.
M = 1 is single-step sCM."""

import json
import os
import warnings
import click
import torch
import dnnlib
from torch_utils import distributed as dist
import training.training_loop

warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`")


def _duration_from_kiters(kiters, batch=2048):
    return int(kiters * 1000) * batch


config_presets = {
    "ms-scd-img64-s-scd": dnnlib.EasyDict(
        duration=_duration_from_kiters(400),
        batch=2048,
        channels=192,
        lr=5.0e-4, # sCM uses 1e-4
        decay=35000,
        dropout=0.10, # applied at resolutions <= 16; sCM uses 0 for sCD
        P_mean=-1.0,
        P_std=1.6,
        mode="scd",
        M=8,
        boundary_schedule="uniform_t",
        proposal_mode="global",
    ),
    "ms-scd-img64-s-sct": dnnlib.EasyDict(
        duration=_duration_from_kiters(400),
        batch=2048,
        channels=192,
        lr=5.0e-4, # sCM uses 1e-4
        decay=35000,
        dropout=0.00,
        P_mean=-1.0,
        P_std=1.6,
        mode="sct",
        sct_dropout=0.45,
        M=8,
        boundary_schedule="uniform_t",
        proposal_mode="global",
    ),
    # CIFAR-10, using the batch size and proposal of the DDPM++ teacher.
    "ms-scd-cifar10-scd": dnnlib.EasyDict(
        duration=_duration_from_kiters(400, batch=512),
        batch=512,
        channels=128,
        lr=1.0e-4,
        decay=0,
        dropout=0.00,
        P_mean=-1.2,
        P_std=1.2,
        mode="scd",
        M=8,
        boundary_schedule="uniform_t",
        proposal_mode="global",
    ),
    "ms-scd-cifar10-sct": dnnlib.EasyDict(
        duration=_duration_from_kiters(400, batch=512),
        batch=512,
        channels=128,
        lr=1.0e-4,
        decay=0,
        dropout=0.00,
        P_mean=-1.2,
        P_std=1.2,
        mode="sct",
        sct_dropout=0.45,
        M=8,
        boundary_schedule="uniform_t",
        proposal_mode="global",
    ),
}


def setup_training_config(preset="ms-scd-img64-s-scd", **opts):
    opts = dnnlib.EasyDict(opts)
    c = dnnlib.EasyDict()
    is_cifar = "cifar" in preset

    if preset not in config_presets:
        raise click.ClickException(f'Invalid configuration preset "{preset}"')
    for key, value in config_presets[preset].items():
        if opts.get(key, None) is None:
            opts[key] = value

    if opts.teacher is None:
        raise click.ClickException("--teacher is required for MS-sCD training (seeds the student).")

    c.dataset_kwargs = dnnlib.EasyDict(
        class_name="training.dataset.ImageFolderDataset", path=opts.data, use_labels=(not is_cifar) if opts.get("cond") is None else opts.cond
    )
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_channels = dataset_obj.num_channels
        dataset_resolution = dataset_obj.resolution
        if c.dataset_kwargs.use_labels and not dataset_obj.has_labels:
            raise click.ClickException("--cond=True, but no labels found in the dataset")
        if dataset_channels != 3:
            raise click.ClickException(f"--data: expected RGB image dataset; got {dataset_channels} channels")
        expected_res = 32 if is_cifar else 64
        if dataset_resolution != expected_res:
            raise click.ClickException(
                f"--data: expected {expected_res}x{expected_res} dataset for preset {preset}, "
                f"got {dataset_resolution}x{dataset_resolution}"
            )
        del dataset_obj
    except IOError as err:
        raise click.ClickException(f"--data: {err}")

    c.encoder_kwargs = dnnlib.EasyDict(class_name="training.encoders.StandardRGBEncoder")
    c.update(total_nimg=opts.duration, batch_size=opts.batch)
    if is_cifar:
        # Same backbone as the DDPM++ teacher, which initializes the student.
        # augment_dim is kept for the weight copy; distillation does not augment.
        c.network_kwargs = dnnlib.EasyDict(
            class_name="training.networks_trigflow_ddpmpp.TrigFlowDDPMPPPrecond",
            model_channels=opts.channels,
            channel_mult=[2, 2, 2],
            num_blocks=4,
            attn_resolutions=[16],
            dropout=opts.dropout,
            augment_dim=9,
            segment_conditioning=True,
        )
    else:
        c.network_kwargs = dnnlib.EasyDict(
            class_name="training.networks_trigflow.TrigFlowPrecond",
            model_channels=opts.channels,
            dropout=opts.dropout,
            segment_conditioning=True,
        )
    sigma_max = float(opts.get("sigma_max", 80.0))
    c.loss_kwargs = dnnlib.EasyDict(
        class_name="training.loss_ms_scm.TrigFlowMSSCDLoss",
        mode=opts.mode,
        num_segments=int(opts.M),
        boundary_schedule=str(opts.boundary_schedule),
        proposal_mode=str(opts.proposal_mode),
        sigma_max=sigma_max,
        P_mean=opts.P_mean,
        P_std=opts.P_std,
        sigma_data=0.5,
        tangent_c=float(opts.tangent_c),
        tangent_warmup_iters=int(opts.tangent_warmup),
        sct_dropout=float(opts.get("sct_dropout", 0.45)),
    )
    c.lr_kwargs = dnnlib.EasyDict(
        func_name="training.training_loop_trigflow.trigflow_learning_rate_schedule",
        ref_lr=opts.lr,
        ref_batches=opts.decay,
    )
    c.optimizer_kwargs = dnnlib.EasyDict(class_name="torch.optim.Adam", betas=(0.9, 0.99), eps=1e-11)
    c.teacher_pkl = opts.teacher

    c.batch_gpu = opts.get("batch_gpu", 0) or None
    # CIFAR-10 DDPM++ is unstable in FP16, so it defaults to FP32.
    # img64 runs in BF16 rather than FP16 because the JVP tangents overflow FP16.
    _fp16 = opts.get("fp16", None)
    if _fp16 is None:
        _fp16 = not is_cifar
    c.network_kwargs.use_fp16 = bool(_fp16)
    c.loss_scaling = opts.get("ls", 1)
    c.cudnn_benchmark = opts.get("bench", True)

    workers = int(opts.get("workers", 4))
    c.data_loader_kwargs = dnnlib.EasyDict(
        class_name="torch.utils.data.DataLoader",
        pin_memory=True,
        num_workers=workers,
        prefetch_factor=2 if workers > 0 else None,
    )

    c.status_nimg = opts.get("status", 0) or None
    c.snapshot_nimg = opts.get("snapshot", 0) or None
    c.checkpoint_nimg = opts.get("checkpoint", 0) or None
    c.checkpoint_keep_recent = int(opts.get("checkpoint_keep_recent", 3))
    c.checkpoint_cleanup_snapshots = not bool(opts.get("no_checkpoint_snapshot_prune", False))
    c.seed = opts.get("seed", 0)

    val_ref = opts.get("val_ref", None)
    c.validation_kwargs = None
    if val_ref is not None:
        ms_segments = [int(opts.M)]
        extra = opts.get("val_segments", None)
        if extra:
            for s in str(extra).split(","):
                s = s.strip()
                if s and int(s) not in ms_segments:
                    ms_segments.append(int(s))
        c.validation_kwargs = dnnlib.EasyDict(
            enabled=True,
            ref=val_ref,
            every=int(opts.get("val_every", 1)),
            num_images=int(opts.get("val_num", 50000)),
            batch=int(opts.get("val_batch", 32)),
            seed=int(opts.get("val_seed", 0)),
            at_start=bool(opts.get("val_at_start", False)),
            val_mode="ms_scm",
            num_segments=int(opts.M),
            ms_segments=ms_segments,
            boundary_schedule=str(opts.boundary_schedule),
            sigma_max=sigma_max,
            val_phema_std=float(opts.get("val_phema_std", 0.05)),
            val_ema_halflife_kimg=opts.get("val_ema_halflife_kimg", None),
            val_ema_rampup_ratio=opts.get("val_ema_rampup_ratio", None),
        )

    c.wandb_kwargs = None
    if opts.get("wandb", False):
        c.wandb_kwargs = dnnlib.EasyDict(
            project=opts.get("wandb_project", "trigflow-ms-scd-img64"),
            name=opts.get("wandb_run_name", None),
            entity=opts.get("wandb_entity", None),
        )
    return c


def print_training_config(run_dir, c):
    dist.print0()
    dist.print0("Training config:")
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f"Output directory:        {run_dir}")
    dist.print0(f"Teacher checkpoint:      {c.teacher_pkl}")
    dist.print0(f"Dataset path:            {c.dataset_kwargs.path}")
    dist.print0(f"MS-sCD mode:             {c.loss_kwargs.mode}")
    dist.print0(f"Segments (M):            {c.loss_kwargs.num_segments}")
    dist.print0(f"Boundary schedule:       {c.loss_kwargs.boundary_schedule}")
    dist.print0(f"Proposal mode:           {c.loss_kwargs.proposal_mode}")
    dist.print0(f"Number of GPUs:          {dist.get_world_size()}")
    dist.print0(f"Batch size:              {c.batch_size}")
    if c.get("validation_kwargs") and c.validation_kwargs.get("enabled"):
        vk = c.validation_kwargs
        dist.print0(
            f"MS-sCD FID validation:   M={vk.get('ms_segments')}, "
            f"schedule={vk.get('boundary_schedule')}, ref={vk.get('ref')}"
        )


def launch_training(run_dir, c):
    if dist.get_rank() == 0 and not os.path.isdir(run_dir):
        dist.print0("Creating output directory...")
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "training_options.json"), "wt") as f:
            json.dump(c, f, indent=2)

    torch.distributed.barrier()
    dnnlib.util.Logger(file_name=os.path.join(run_dir, "log.txt"), file_mode="a", should_flush=True)
    training.training_loop.training_loop(run_dir=run_dir, **c)


def parse_nimg(s):
    if isinstance(s, int):
        return s
    if s.endswith("Ki"):
        return int(s[:-2]) << 10
    if s.endswith("Mi"):
        return int(s[:-2]) << 20
    if s.endswith("Gi"):
        return int(s[:-2]) << 30
    return int(s)


@click.command()
@click.option("--outdir", help="Where to save the results", metavar="DIR", type=str, required=True)
@click.option("--data", help="Path to the dataset", metavar="ZIP|DIR", type=str, required=True)
@click.option("--teacher", help="Frozen TrigFlow teacher EMA snapshot", metavar="PKL", type=str, required=True)
@click.option("--cond", help="Train class-conditional model  [default: False for CIFAR-10, else True]", metavar="BOOL", type=bool, default=None)
@click.option("--preset", help="Configuration preset", metavar="STR", type=str, default="ms-scd-img64-s-scd", show_default=True)
@click.option("--mode", help="MS-sCD mode", metavar="STR", type=click.Choice(["scd", "sct"]), default=None)
@click.option("--M", "M", help="Number of segments (M=1 -> sCM)", metavar="INT", type=click.IntRange(min=1), default=None)
@click.option("--boundary-schedule", "boundary_schedule", help="Segment boundary schedule", metavar="STR", type=click.Choice(["uniform_t", "uniform_log_snr"]), default=None)
@click.option("--proposal-mode", "proposal_mode", help="Time proposal", metavar="STR", type=click.Choice(["global", "segment_balanced"]), default=None)
@click.option("--duration", help="Training duration", metavar="NIMG", type=parse_nimg, default=None)
@click.option("--batch", help="Total batch size", metavar="NIMG", type=parse_nimg, default=None)
@click.option("--channels", help="Channel multiplier", metavar="INT", type=click.IntRange(min=64), default=None)
@click.option("--dropout", help="Dropout probability (sCD uses 0)", metavar="FLOAT", type=click.FloatRange(min=0, max=1), default=None)
@click.option("--P_mean", "P_mean", help="Proposal mean", metavar="FLOAT", type=float, default=None)
@click.option("--P_std", "P_std", help="Proposal std", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=None)
@click.option("--sigma-max", "sigma_max", help="Max noise (defines t_max)", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=80.0, show_default=True)
@click.option("--tangent-c", "tangent_c", help="Tangent normalization constant", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=0.1, show_default=True)
@click.option("--tangent-warmup", "tangent_warmup", help="Tangent warmup iterations", metavar="INT", type=click.IntRange(min=1), default=10000, show_default=True)
@click.option("--lr", help="Learning rate max. (alpha_ref)", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=None)
@click.option("--decay", help="Learning rate decay (t_ref)", metavar="BATCHES", type=click.FloatRange(min=0), default=None)
@click.option("--batch-gpu", help="Limit batch size per GPU", metavar="NIMG", type=parse_nimg, default=0, show_default=True)
@click.option("--fp16", help="Enable half precision  [default: BF16 for img64, FP32 for CIFAR-10]", metavar="BOOL", type=bool, default=None)
@click.option("--ls", help="Loss scaling", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option("--bench", help="Enable cuDNN benchmarking", metavar="BOOL", type=bool, default=True, show_default=True)
@click.option("--status", help="Interval of status prints", metavar="NIMG", type=parse_nimg, default="128Ki", show_default=True)
@click.option("--snapshot", help="Interval of network snapshots", metavar="NIMG", type=parse_nimg, default="8Mi", show_default=True)
@click.option("--checkpoint", help="Interval of training checkpoints", metavar="NIMG", type=parse_nimg, default="128Mi", show_default=True)
@click.option("--checkpoint-keep-recent", "checkpoint_keep_recent", help="Retain N newest training-state .pt files", metavar="INT", type=click.IntRange(min=0), default=3, show_default=True)
@click.option("--no-checkpoint-snapshot-prune", "no_checkpoint_snapshot_prune", help="Disable pruning of primary network-snapshot-{kimg}.pkl files", is_flag=True)
@click.option("--workers", help="DataLoader workers per rank", metavar="INT", type=click.IntRange(min=0), default=4, show_default=True)
@click.option("--seed", help="Random seed", metavar="INT", type=int, default=0, show_default=True)
@click.option("--val-ref", "val_ref", help="FID reference stats (.npz or .pkl)", metavar="PKL|NPZ|URL", type=str, default=None)
@click.option("--val-num", "val_num", help="Images per FID evaluation", metavar="INT", type=click.IntRange(min=2), default=50000, show_default=True)
@click.option("--val-segments", "val_segments", help="Extra M values for FID (comma-separated)", metavar="STR", type=str, default=None)
@click.option("--val-seed", "val_seed", help="Base seed for FID validation", metavar="INT", type=int, default=0, show_default=True)
@click.option("--val-batch", "val_batch", help="Per-GPU batch size for FID validation", metavar="INT", type=click.IntRange(min=1), default=32, show_default=True)
@click.option("--val-every", "val_every", help="Run FID every N snapshots", metavar="INT", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--val-at-start", "val_at_start", help="Run FID validation before any training step", metavar="BOOL", type=bool, default=False, show_default=True)
@click.option("--val-phema-std", "val_phema_std", help="Target phEMA std for validation tradEMA", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=0.05, show_default=True)
@click.option("--val-ema-halflife-kimg", "val_ema_halflife_kimg", help="Validation tradEMA half-life cap (kimg)", metavar="FLOAT", type=float, default=None)
@click.option("--val-ema-rampup-ratio", "val_ema_rampup_ratio", help="Override validation tradEMA rampup ratio", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=None)
@click.option("--wandb", help="Enable Weights & Biases logging", is_flag=True)
@click.option("--wandb-project", help="W&B project name", metavar="STR", type=str, default="trigflow-ms-scd-img64", show_default=True)
@click.option("--wandb-run-name", help="W&B run name", metavar="STR", type=str, default=None)
@click.option("--wandb-entity", help="W&B entity/team", metavar="STR", type=str, default=None)
@click.option("-n", "--dry-run", help="Print training options and exit", is_flag=True)
def cmdline(outdir, dry_run, **opts):
    torch.multiprocessing.set_start_method("spawn")
    dist.init()
    dist.print0("Setting up MS-sCD training config...")
    c = setup_training_config(**opts)
    print_training_config(run_dir=outdir, c=c)
    if dry_run:
        dist.print0("Dry run; exiting.")
    else:
        launch_training(run_dir=outdir, c=c)


if __name__ == "__main__":
    cmdline()
