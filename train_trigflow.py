# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modifications copyright (c) 2026, Vinay Saji Mathew.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train TrigFlow diffusion teachers on ImageNet-64 or CIFAR-10."""

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
    # ImageNet-64 teacher settings (sCM appendix).
    "trigflow-img64-s": dnnlib.EasyDict(
        duration=_duration_from_kiters(1048), batch=2048, channels=192, lr=0.0100, decay=35000, dropout=0.00, P_mean=-0.8, P_std=1.6
    ),
    "trigflow-img64-m": dnnlib.EasyDict(
        duration=_duration_from_kiters(1486), batch=2048, channels=256, lr=0.0090, decay=35000, dropout=0.10, P_mean=-0.8, P_std=1.6
    ),
    "trigflow-img64-l": dnnlib.EasyDict(
        duration=_duration_from_kiters(761), batch=2048, channels=320, lr=0.0080, decay=35000, dropout=0.10, P_mean=-0.8, P_std=1.6
    ),
    "trigflow-img64-xl": dnnlib.EasyDict(
        duration=_duration_from_kiters(540), batch=2048, channels=384, lr=0.0070, decay=35000, dropout=0.10, P_mean=-0.8, P_std=1.6
    ),
    # Unconditional CIFAR-10 teacher settings (sCM Appendix G.1).
    "trigflow-cifar10": dnnlib.EasyDict(
        duration=_duration_from_kiters(400, batch=512),
        batch=512,
        channels=128,
        lr=0.0010,
        decay=0,
        dropout=0.13,
        P_mean=-1.2,
        P_std=1.2,
    ),
}


def setup_training_config(preset="trigflow-img64-s", **opts):
    opts = dnnlib.EasyDict(opts)
    c = dnnlib.EasyDict()
    is_cifar = preset == "trigflow-cifar10"

    if preset not in config_presets:
        raise click.ClickException(f'Invalid configuration preset "{preset}"')
    for key, value in config_presets[preset].items():
        if opts.get(key, None) is None:
            opts[key] = value

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
            raise click.ClickException(f"--data: expected {expected_res}x{expected_res} dataset for preset {preset}, got {dataset_resolution}x{dataset_resolution}")
        del dataset_obj
    except IOError as err:
        raise click.ClickException(f"--data: {err}")

    c.encoder_kwargs = dnnlib.EasyDict(class_name="training.encoders.StandardRGBEncoder")
    c.update(total_nimg=opts.duration, batch_size=opts.batch)
    if is_cifar:
        c.network_kwargs = dnnlib.EasyDict(
            class_name="training.networks_trigflow_ddpmpp.TrigFlowDDPMPPPrecond",
            model_channels=opts.channels,
            channel_mult=[2, 2, 2],
            num_blocks=4,
            attn_resolutions=[16],
            dropout=opts.dropout,
            augment_dim=9,
        )
        c.loss_kwargs = dnnlib.EasyDict(
            class_name="training.training_loop_trigflow.TrigFlowLoss",
            P_mean=opts.P_mean,
            P_std=opts.P_std,
            sigma_data=0.5,
            augment_kwargs=dnnlib.EasyDict(
                class_name="training.augment.AugmentPipe",
                p=0.12,
                xflip=1e8,
                yflip=1,
                scale=1,
                rotate_frac=1,
                aniso=1,
                translate_frac=1,
            ),
        )
    else:
        c.network_kwargs = dnnlib.EasyDict(
            class_name="training.networks_trigflow.TrigFlowPrecond", model_channels=opts.channels, dropout=opts.dropout
        )
        c.loss_kwargs = dnnlib.EasyDict(
            class_name="training.training_loop_trigflow.TrigFlowLoss", P_mean=opts.P_mean, P_std=opts.P_std, sigma_data=0.5
        )
    c.lr_kwargs = dnnlib.EasyDict(
        func_name="training.training_loop_trigflow.trigflow_learning_rate_schedule",
        ref_lr=opts.lr,
        ref_batches=opts.decay,
    )
    if is_cifar:
        c.optimizer_kwargs = dnnlib.EasyDict(class_name="torch.optim.Adam", betas=(0.9, 0.999), eps=1e-8)
    else:
        c.optimizer_kwargs = dnnlib.EasyDict(class_name="torch.optim.Adam", betas=(0.9, 0.99), eps=1e-11)

    c.batch_gpu = opts.get("batch_gpu", 0) or None
    # DDPM++ has no forced weight norm and is unstable in FP16, so CIFAR-10 defaults to FP32.
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
    c.init_from = opts.get("init_from")
    c.transfer_fn = None
    if c.init_from is not None and not is_cifar:
        c.transfer_fn = "training.networks_trigflow.transfer_edm2_weights"

    # In-training FID validation (off unless --val-ref is provided).
    val_ref = opts.get("val_ref", None)
    c.validation_kwargs = None
    if val_ref is not None:
        # DPM-Solver-2S with 18 steps (CIFAR-10) or 32 steps (img64). CIFAR-10 uses
        # EDM's EMA (half-life 500 kimg, rampup 0.05) for validation.
        _val_steps_default = 18 if is_cifar else 32
        _val_sampler_default = "dpm2s"
        _val_hl_default = 500.0 if is_cifar else None
        _val_ramp_default = 0.05 if is_cifar else None

        def _resolve(key, default):
            v = opts.get(key, None)
            return v if v is not None else default

        c.validation_kwargs = dnnlib.EasyDict(
            enabled=True,
            ref=val_ref,
            every=int(opts.get("val_every", 1)),
            num_images=int(opts.get("val_num", 50000)),
            steps=int(_resolve("val_steps", _val_steps_default)),
            order=int(opts.get("val_order", 2)),
            sampler=str(_resolve("val_sampler", _val_sampler_default)),
            seed=int(opts.get("val_seed", 0)),
            batch=int(opts.get("val_batch", 32)),
            sigma_min=float(opts.get("val_sigma_min", 0.002)),
            sigma_max=float(opts.get("val_sigma_max", 80.0)),
            rho=float(opts.get("val_rho", 7.0)),
            at_start=bool(opts.get("val_at_start", False)),
            val_phema_std=float(opts.get("val_phema_std", 0.075)),
            val_ema_halflife_kimg=_resolve("val_ema_halflife_kimg", _val_hl_default),
            val_ema_rampup_ratio=_resolve("val_ema_rampup_ratio", _val_ramp_default),
        )

    c.wandb_kwargs = None
    if opts.get("wandb", False):
        c.wandb_kwargs = dnnlib.EasyDict(
            project=opts.get("wandb_project", "edm2-trigflow-teacher"),
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
    dist.print0(f"Dataset path:            {c.dataset_kwargs.path}")
    dist.print0(f"Class-conditional:       {c.dataset_kwargs.use_labels}")
    dist.print0(f"Number of GPUs:          {dist.get_world_size()}")
    dist.print0(f"Batch size:              {c.batch_size}")
    dist.print0(f"Mixed-precision:         {c.network_kwargs.use_fp16}")
    if c.get("validation_kwargs") and c.validation_kwargs.get("enabled"):
        vk = c.validation_kwargs
        steps = int(vk.get("steps", 18))
        order = int(vk.get("order", 2))
        sampler = str(vk.get("sampler", "edm_heun"))
        nfe = (2 * steps - 1) if (sampler == "edm_heun" or order >= 2) else steps
        dist.print0(
            f"FID validation:          every {vk.get('every', 1)} snapshot(s), "
            f"{vk.get('num_images', 50000)} images, sampler={sampler}, {steps} steps ({nfe} NFEs)"
        )
        dist.print0(f"FID reference:           {vk.get('ref')}")
        _val_std = vk.get("val_phema_std", None)
        _val_hl = vk.get("val_ema_halflife_kimg", None)
        _val_ramp = vk.get("val_ema_rampup_ratio", None)
        _hl_str = "uncapped" if _val_hl is None else f"{int(_val_hl)} kimg"
        _ramp_str = (
            "auto (derived from val-phema-std)"
            if _val_ramp is None
            else f"{float(_val_ramp):.6f}"
        )
        dist.print0(
            f"Val tradEMA:             target phEMA std={_val_std}, "
            f"halflife_cap={_hl_str}, rampup_ratio={_ramp_str}"
        )
    dist.print0()


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
@click.option("--cond", help="Train class-conditional model  [default: False for CIFAR-10, else True]", metavar="BOOL", type=bool, default=None)
@click.option("--preset", help="Configuration preset", metavar="STR", type=str, default="trigflow-img64-s", show_default=True)
@click.option("--init-from", help="Warm-start checkpoint", metavar="PKL", type=str, default=None)
@click.option("--duration", help="Training duration", metavar="NIMG", type=parse_nimg, default=None)
@click.option("--batch", help="Total batch size", metavar="NIMG", type=parse_nimg, default=None)
@click.option("--channels", help="Channel multiplier", metavar="INT", type=click.IntRange(min=64), default=None)
@click.option("--dropout", help="Dropout probability", metavar="FLOAT", type=click.FloatRange(min=0, max=1), default=None)
@click.option("--P_mean", "P_mean", help="Noise level mean", metavar="FLOAT", type=float, default=None)
@click.option(
    "--P_std", "P_std", help="Noise level standard deviation", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=None
)
@click.option("--lr", help="Learning rate max. (alpha_ref)", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=None)
@click.option("--decay", help="Learning rate decay (t_ref)", metavar="BATCHES", type=click.FloatRange(min=0), default=None)
@click.option("--batch-gpu", help="Limit batch size per GPU", metavar="NIMG", type=parse_nimg, default=0, show_default=True)
@click.option("--fp16", help="Enable mixed-precision training  [default: BF16 for img64, FP32 for CIFAR-10]", metavar="BOOL", type=bool, default=None)
@click.option("--ls", help="Loss scaling", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option("--bench", help="Enable cuDNN benchmarking", metavar="BOOL", type=bool, default=True, show_default=True)
@click.option("--status", help="Interval of status prints", metavar="NIMG", type=parse_nimg, default="128Ki", show_default=True)
@click.option("--snapshot", help="Interval of network snapshots", metavar="NIMG", type=parse_nimg, default="8Mi", show_default=True)
@click.option("--checkpoint", help="Interval of training checkpoints", metavar="NIMG", type=parse_nimg, default="128Mi", show_default=True)
@click.option("--checkpoint-keep-recent", "checkpoint_keep_recent", help="Retain N newest training-state .pt files (plus best-FID .pt); <=0 disables pruning", metavar="INT", type=click.IntRange(min=0), default=3, show_default=True)
@click.option("--no-checkpoint-snapshot-prune", "no_checkpoint_snapshot_prune", help="Disable pruning of primary network-snapshot-{kimg}.pkl files", is_flag=True)
@click.option("--workers", help="DataLoader workers per rank", metavar="INT", type=click.IntRange(min=0), default=4, show_default=True)
@click.option("--seed", help="Random seed", metavar="INT", type=int, default=0, show_default=True)
@click.option("--val-ref", "val_ref", help="FID reference stats (.npz or .pkl); enables in-training FID", metavar="PKL|NPZ|URL", type=str, default=None)
@click.option("--val-num", "val_num", help="Images per FID evaluation", metavar="INT", type=click.IntRange(min=2), default=50000, show_default=True)
@click.option("--val-steps", "val_steps", help="Sampler steps for FID validation  [default: 18 for CIFAR-10, 32 for img64]", metavar="INT", type=click.IntRange(min=1), default=None)
@click.option("--val-order", "val_order", help="Order of the dpm2s sampler (1 = DDIM, 2 = DPM-Solver-2S)", metavar="INT", type=click.IntRange(min=1, max=2), default=2, show_default=True)
@click.option("--val-sampler", "val_sampler", help="FID sampler: dpm2s (DPM-Solver-2S) or edm_heun (cross-check)  [default: dpm2s]", metavar="STR", type=click.Choice(["dpm2s", "edm_heun"]), default=None)
@click.option("--val-seed", "val_seed", help="Base seed for FID validation", metavar="INT", type=int, default=0, show_default=True)
@click.option("--val-batch", "val_batch", help="Per-GPU batch size for FID validation", metavar="INT", type=click.IntRange(min=1), default=32, show_default=True)
@click.option("--val-every", "val_every", help="Run FID every N snapshots", metavar="INT", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--val-at-start", "val_at_start", help="Run FID validation before any training step", metavar="BOOL", type=bool, default=False, show_default=True)
@click.option("--val-sigma-min", "val_sigma_min", help="FID sampler sigma_min", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=0.002, show_default=True)
@click.option("--val-sigma-max", "val_sigma_max", help="FID sampler sigma_max", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=80.0, show_default=True)
@click.option("--val-rho", "val_rho", help="FID sampler rho", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=7.0, show_default=True)
@click.option("--val-phema-std", "val_phema_std", help="Target phEMA std for validation tradEMA", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=0.075, show_default=True)
@click.option("--val-ema-halflife-kimg", "val_ema_halflife_kimg", help="Validation tradEMA half-life cap (kimg)", metavar="FLOAT", type=float, default=None)
@click.option("--val-ema-rampup-ratio", "val_ema_rampup_ratio", help="Override validation tradEMA rampup ratio", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=None)
@click.option("--wandb", help="Enable Weights & Biases logging", is_flag=True)
@click.option("--wandb-project", help="W&B project name", metavar="STR", type=str, default="edm2-trigflow-teacher", show_default=True)
@click.option("--wandb-run-name", help="W&B run name", metavar="STR", type=str, default=None)
@click.option("--wandb-entity", help="W&B entity/team", metavar="STR", type=str, default=None)
@click.option("-n", "--dry-run", help="Print training options and exit", is_flag=True)
def cmdline(outdir, dry_run, **opts):
    torch.multiprocessing.set_start_method("spawn")
    dist.init()
    dist.print0("Setting up training config...")
    c = setup_training_config(**opts)
    print_training_config(run_dir=outdir, c=c)
    if dry_run:
        dist.print0("Dry run; exiting.")
    else:
        launch_training(run_dir=outdir, c=c)


if __name__ == "__main__":
    cmdline()
