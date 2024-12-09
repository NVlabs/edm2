# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Perform post-hoc EMA reconstruction."""

import os
import re
import copy
import warnings
import click
import tqdm
import pickle
import numpy as np
import torch
import dnnlib
import training.phema

warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`')

#----------------------------------------------------------------------------
# Construct the full path of a network pickle.

def pkl_path(dir, prefix, nimg, std):
    name = prefix + f'-{nimg//1000:07d}-{std:.3f}.pkl'
    if dir is None:
        return None
    if dnnlib.util.is_url(dir):
        return f'{dir}/{name}'
    return os.path.join(dir, name)

#----------------------------------------------------------------------------
# Deduce nimg based on kimg (= nimg//1000).

def kimg_to_nimg(kimg):
    nimg = (kimg * 1000 + 999) // 1024 * 1024
    assert nimg // 1000 == kimg
    return nimg

#----------------------------------------------------------------------------
# List input pickles for post-hoc EMA reconstruction.
# Returns a list of dnnlib.EasyDict(path, nimg, std).

def list_input_pickles(
    in_dir,             # Directory containing the input pickles.
    in_prefix   = None, # Input filename prefix. None = anything goes.
    in_std      = None, # Relative standard deviations of the input pickles. None = anything goes.
):
    if not os.path.isdir(in_dir):
        raise click.ClickException('Input directory does not exist')
    in_std = set(in_std) if in_std is not None else None

    pkls = []
    with os.scandir(in_dir) as it:
        for e in it:
            m = re.fullmatch(r'(.*)-(\d+)-(\d+\.\d+)\.pkl', e.name)
            if not m or not e.is_file():
                continue
            prefix = m.group(1)
            nimg = kimg_to_nimg(int(m.group(2)))
            std = float(m.group(3))
            if in_prefix is not None and prefix != in_prefix:
                continue
            if in_std is not None and std not in in_std:
                continue
            pkls.append(dnnlib.EasyDict(path=e.path, nimg=nimg, std=std))
    pkls = sorted(pkls, key=lambda pkl: (pkl.nimg, pkl.std))
    return pkls

#----------------------------------------------------------------------------
# Perform post-hoc EMA reconstruction.
# Returns an iterable that yields dnnlib.EasyDict(out, step_idx, num_steps),
# where 'out' is a list of dnnlib.EasyDict(net, nimg, std, pkl_data, pkl_path)

def reconstruct_phema(
    in_pkls,                    # List of input pickles, expressed as dnnlib.EasyDict(path, nimg, std).
    out_std,                    # List of relative standard deviations to reconstruct.
    out_nimg        = None,     # Training time of the snapshot to reconstruct. None = highest input time.
    out_dir         = None,     # Where to save the reconstructed network pickles. None = do not save.
    out_prefix      = 'phema',  # Output filename prefix.
    skip_existing   = False,    # Skip output files that already exist?
    max_batch_size  = 8,        # Maximum simultaneous reconstructions
    verbose         = True,     # Enable status prints?
):
    # Validate input pickles.
    if out_nimg is None:
        out_nimg = max((pkl.nimg for pkl in in_pkls), default=0)
    elif not any(out_nimg == pkl.nimg for pkl in in_pkls):
        raise click.ClickException('Reconstruction time must match one of the input pickles')
    in_pkls = [pkl for pkl in in_pkls if 0 < pkl.nimg <= out_nimg]
    if len(in_pkls) == 0:
        raise click.ClickException('No valid input pickles found')
    in_nimg = [pkl.nimg for pkl in in_pkls]
    in_std = [pkl.std for pkl in in_pkls]
    if verbose:
        print(f'Loading {len(in_pkls)} input pickles...')
        for pkl in in_pkls:
            print('    ' + pkl.path)

    # Determine output pickles.
    out_std = [out_std] if isinstance(out_std, float) else sorted(set(out_std))
    if skip_existing and out_dir is not None:
        out_std = [std for std in out_std if not os.path.isfile(pkl_path(out_dir, out_prefix, out_nimg, std))]
    num_batches = (len(out_std) - 1) // max_batch_size + 1
    out_std_batches = np.array_split(out_std, num_batches)
    if verbose:
        print(f'Reconstructing {len(out_std)} output pickles in {num_batches} batches...')
        for i, batch in enumerate(out_std_batches):
            for std in batch:
                print(f'    batch {i}: ', end='')
                print(pkl_path(out_dir, out_prefix, out_nimg, std) if out_dir is not None else pkl_path('', '<yield>', out_nimg, std))

    # Return an iterable over the reconstruction steps.
    class ReconstructionIterable:
        def __len__(self):
            return num_batches * len(in_pkls)

        def __iter__(self):
            # Loop over batches.
            r = dnnlib.EasyDict(step_idx=0, num_steps=len(self))
            for out_std_batch in out_std_batches:
                coefs = training.phema.solve_posthoc_coefficients(in_nimg, in_std, out_nimg, out_std_batch)
                out = [dnnlib.EasyDict(net=None, nimg=out_nimg, std=std) for std in out_std_batch]
                r.out = []

                # Loop over input pickles.
                for i in range(len(in_pkls)):
                    with dnnlib.util.open_url(in_pkls[i].path, verbose=False) as f:
                        in_pkl_data = pickle.load(f)
                        in_net = in_pkl_data['ema'].to(torch.float32)

                    # Accumulate weights for each output pickle.
                    for j in range(len(out)):
                        if out[j].net is None:
                            out[j].pkl_data = copy.deepcopy(in_pkl_data)
                            out[j].net = out[j].pkl_data['ema']
                            for pj in out[j].net.parameters():
                                pj.zero_()
                        for pi, pj in zip(in_net.parameters(), out[j].net.parameters()):
                            pj += pi * coefs[i, j]
                        for pi, pj in zip(in_net.buffers(), out[j].net.buffers()):
                            pj.copy_(pi)

                    # Finalize outputs.
                    if i == len(in_pkls) - 1:
                        for j in range(len(out)):
                            out[j].net.to(torch.float16)
                            out[j].pkl_path = pkl_path(out_dir, out_prefix, out_nimg, out[j].std)
                            if out[j].pkl_path is not None:
                                os.makedirs(out_dir, exist_ok=True)
                                with open(out[j].pkl_path, 'wb') as f:
                                    pickle.dump(out[j].pkl_data, f)
                        r.out = out

                    # Yield results.
                    del in_pkl_data, in_net # conserve memory
                    yield r
                    r.step_idx += 1

    return ReconstructionIterable()

#----------------------------------------------------------------------------
# Parse a comma separated list of relative standard deviations.
# The special token '...' interpreted as an evenly spaced interval.
# Example: '0.01,0.02,...,0.05' returns [0.01, 0.02, 0.03, 0.04, 0.05]

def parse_std_list(s):
    if isinstance(s, list):
        return s

    # Parse raw values.
    raw = [None if v == '...' else float(v) for v in s.split(',')]

    # Fill in '...' tokens.
    out = []
    for i, v in enumerate(raw):
        if v is not None:
            out.append(v)
            continue
        if i - 2 < 0 or raw[i - 2] is None or raw[i - 1] is None:
            raise click.ClickException("'...' must be preceded by at least two floats")
        if i + 1 >= len(raw) or raw[i + 1] is None:
            raise click.ClickException("'...' must be followed by at least one float")
        if raw[i - 2] == raw[i - 1]:
            raise click.ClickException("The floats preceding '...' must not be equal")
        approx_num = (raw[i + 1] - raw[i - 1]) / (raw[i - 1] - raw[i - 2]) - 1
        num = round(approx_num)
        if num <= 0:
            raise click.ClickException("'...' must correspond to a non-empty interval")
        if abs(num - approx_num) > 1e-4:
            raise click.ClickException("'...' must correspond to an evenly spaced interval")
        for j in range(num):
            out.append(raw[i - 1] + (raw[i - 1] - raw[i - 2]) * (j + 1))

    # Validate.
    out = sorted(set(out))
    if not all(0.000 < v < 0.289 for v in out):
        raise click.ClickException('Relative standard deviation must be positive and less than 0.289')
    return out

#----------------------------------------------------------------------------
# Command line interface.

@click.command()
@click.option('--indir', 'in_dir',          help='Directory containing the input pickles', metavar='DIR',           type=str, required=True)
@click.option('--inprefix', 'in_prefix',    help='Filter inputs based on filename prefix', metavar='STR',           type=str, default=None)
@click.option('--instd', 'in_std',          help='Filter inputs based on standard deviations', metavar='LIST',      type=parse_std_list, default=None)

@click.option('--outdir', 'out_dir',        help='Where to save the reconstructed network pickles', metavar='DIR',  type=str, required=True)
@click.option('--outprefix', 'out_prefix',  help='Output filename prefix', metavar='STR',                           type=str, default='phema', show_default=True)
@click.option('--outstd', 'out_std',        help='List of desired relative standard deviations', metavar='LIST',    type=parse_std_list, required=True)
@click.option('--outkimg', 'out_kimg',      help='Training time of the snapshot to reconstruct', metavar='KIMG',    type=click.IntRange(min=1), default=None)

@click.option('--skip', 'skip_existing',    help='Skip output files that already exist',                            is_flag=True)
@click.option('--batch', 'max_batch_size',  help='Maximum simultaneous reconstructions', metavar='INT',             type=click.IntRange(min=1), default=8, show_default=True)

def cmdline(in_dir, in_prefix, in_std, out_kimg, **opts):
    """Perform post-hoc EMA reconstruction.

    Examples:

    \b
    # Download raw snapshots for the pre-trained edm2-img512-xs model
    rclone copy --progress --http-url https://nvlabs-fi-cdn.nvidia.com/edm2 \\
        :http:raw-snapshots/edm2-img512-xs/ raw-snapshots/edm2-img512-xs/

    \b
    # Reconstruct a new EMA profile with std=0.150
    python reconstruct_phema.py --indir=raw-snapshots/edm2-img512-xs \\
        --outdir=out --outstd=0.150

    \b
    # Reconstruct a set of 31 EMA profiles, streaming over the input data 4 times
    python reconstruct_phema.py --indir=raw-snapshots/edm2-img512-xs \\
        --outdir=out --outstd=0.010,0.015,...,0.250 --batch=8

    \b
    # Perform reconstruction for the latest snapshot of a given training run
    python reconstruct_phema.py --indir=training-runs/00000-edm2-img512-xs \\
        --outdir=out --outstd=0.150
    """
    if os.environ.get('WORLD_SIZE', '1') != '1':
        raise click.ClickException('Distributed execution is not supported')
    out_nimg = kimg_to_nimg(out_kimg) if out_kimg is not None else None
    in_pkls = list_input_pickles(in_dir=in_dir, in_prefix=in_prefix, in_std=in_std)
    rec_iter = reconstruct_phema(in_pkls=in_pkls, out_nimg=out_nimg, **opts)
    for _r in tqdm.tqdm(rec_iter, unit='step'):
        pass

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
