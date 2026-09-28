# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modifications copyright (c) 2026, Vinay Saji Mathew.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Compute FID for TrigFlow teacher snapshots using the in-training
validation sampler."""

import argparse
import json
import os
import pickle
import re
import sys
import time

import torch

# Make the repository root importable when run as scripts/eval_trigflow_fid.py.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import dnnlib
from torch_utils import distributed as dist
from validation import run_fid_validation


def _parse_snapshot_kimg(path):
    match = re.search(r'network-snapshot-(\d+)', os.path.basename(path))
    return int(match.group(1)) if match else None


def _load_snapshot(path, device):
    dist.print0(f'[EVAL] Loading snapshot: {path}')
    with dnnlib.util.open_url(path, verbose=(dist.get_rank() == 0)) as f:
        data = pickle.load(f)
    net = data['ema'].to(device).eval().requires_grad_(False)
    encoder = data.get('encoder', None)
    if encoder is None:
        encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder')
    encoder.init(device)
    dataset_kwargs = data.get('dataset_kwargs', {})
    return net, encoder, dataset_kwargs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshots', nargs='+', required=True, help='Network snapshot PKLs to evaluate.')
    parser.add_argument('--outdir', required=True, help='Directory for metrics-val.jsonl and summary.jsonl.')
    parser.add_argument('--ref', required=True, help='FID reference stats (.pkl or .npz).')
    parser.add_argument('--num-images', type=int, default=50000)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--order', type=int, default=2)
    parser.add_argument('--sigma-min', type=float, default=0.002)
    parser.add_argument('--sigma-max', type=float, default=80.0)
    parser.add_argument('--rho', type=float, default=7.0)
    args = parser.parse_args()

    dist.init()
    device = torch.device('cuda')
    if dist.get_rank() == 0:
        os.makedirs(args.outdir, exist_ok=True)
        with open(os.path.join(args.outdir, 'config.json'), 'wt') as f:
            json.dump(vars(args), f, indent=2)

    torch.distributed.barrier()

    for snapshot in args.snapshots:
        step_kimg = _parse_snapshot_kimg(snapshot)
        net, encoder, dataset_kwargs = _load_snapshot(snapshot, device)
        result = run_fid_validation(
            net,
            encoder,
            run_dir=args.outdir,
            dataset_kwargs=dataset_kwargs,
            num_images=args.num_images,
            batch=args.batch,
            seed=args.seed,
            num_steps=args.steps,
            order=args.order,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            rho=args.rho,
            ref=args.ref,
            step_kimg=step_kimg,
            wandb_run=None,
        )
        if dist.get_rank() == 0:
            row = dict(
                snapshot=snapshot,
                snapshot_kimg=step_kimg,
                fid=result.get('fid'),
                num_images=args.num_images,
                steps=args.steps,
                order=args.order,
                nfe=(2 * args.steps - 1) if args.order >= 2 else args.steps,
                timestamp=time.time(),
            )
            with open(os.path.join(args.outdir, 'summary.jsonl'), 'at') as f:
                f.write(json.dumps(row) + '\n')
            dist.print0(f'[EVAL] RESULT {os.path.basename(snapshot)}: FID={row["fid"]}')
        del net, encoder
        torch.cuda.empty_cache()

    if dist.get_rank() == 0:
        dist.print0('[EVAL] Done.')


if __name__ == '__main__':
    main()

