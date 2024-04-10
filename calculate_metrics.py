# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Calculate evaluation metrics (FID and FD_DINOv2)."""

import os
import click
import tqdm
import pickle
import numpy as np
import scipy.linalg
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from torch_utils import misc
from training import dataset
import generate_images

#----------------------------------------------------------------------------
# Abstract base class for feature detectors.

class Detector:
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim

    def __call__(self, x): # NCHW, uint8, 3 channels => NC, float32
        raise NotImplementedError # to be overridden by subclass

#----------------------------------------------------------------------------
# InceptionV3 feature detector.
# This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz

class InceptionV3Detector(Detector):
    def __init__(self):
        super().__init__(feature_dim=2048)
        url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
        with dnnlib.util.open_url(url, verbose=False) as f:
            self.model = pickle.load(f)

    def __call__(self, x):
        return self.model.to(x.device)(x, return_features=True)

#----------------------------------------------------------------------------
# DINOv2 feature detector.
# Modeled after https://github.com/layer6ai-labs/dgm-eval

class DINOv2Detector(Detector):
    def __init__(self, resize_mode='torch'):
        super().__init__(feature_dim=1024)
        self.resize_mode = resize_mode
        import warnings
        warnings.filterwarnings('ignore', 'xFormers is not available')
        torch.hub.set_dir(dnnlib.make_cache_dir_path('torch_hub'))
        self.model = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vitl14', trust_repo=True, verbose=False, skip_validation=True)
        self.model.eval().requires_grad_(False)

    def __call__(self, x):
        # Resize images.
        if self.resize_mode == 'pil': # Slow reference implementation that matches the original dgm-eval codebase exactly.
            device = x.device
            x = x.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            x = np.stack([np.uint8(PIL.Image.fromarray(xx, 'RGB').resize((224, 224), PIL.Image.Resampling.BICUBIC)) for xx in x])
            x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
        elif self.resize_mode == 'torch': # Fast practical implementation that yields almost the same results.
            x = torch.nn.functional.interpolate(x.to(torch.float32), size=(224, 224), mode='bicubic', antialias=True)
        else:
            raise ValueError(f'Invalid resize mode "{self.resize_mode}"')

        # Adjust dynamic range.
        x = x.to(torch.float32) / 255
        x = x - misc.const_like(x, [0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        x = x / misc.const_like(x, [0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

        # Run DINOv2 model.
        return self.model.to(x.device)(x)

#----------------------------------------------------------------------------
# Metric specifications.

metric_specs = {
    'fid':          dnnlib.EasyDict(detector_kwargs=dnnlib.EasyDict(class_name=InceptionV3Detector)),
    'fd_dinov2':    dnnlib.EasyDict(detector_kwargs=dnnlib.EasyDict(class_name=DINOv2Detector)),
}

#----------------------------------------------------------------------------
# Get feature detector for the given metric.

_detector_cache = dict()

def get_detector(metric, verbose=True):
    # Lookup from cache.
    if metric in _detector_cache:
        return _detector_cache[metric]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Construct detector.
    kwargs = metric_specs[metric].detector_kwargs
    if verbose:
        name = kwargs.class_name.split('.')[-1] if isinstance(kwargs.class_name, str) else kwargs.class_name.__name__
        dist.print0(f'Setting up {name}...')
    detector = dnnlib.util.construct_class_by_name(**kwargs)
    _detector_cache[metric] = detector

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()
    return detector

#----------------------------------------------------------------------------
# Load feature statistics from the given .pkl or .npz file.

def load_stats(path, verbose=True):
    if verbose:
        print(f'Loading feature statistics from {path} ...')
    with dnnlib.util.open_url(path, verbose=verbose) as f:
        if path.lower().endswith('.npz'): # backwards compatibility with https://github.com/NVlabs/edm
            return {'fid': dict(np.load(f))}
        return pickle.load(f)

#----------------------------------------------------------------------------
# Save feature statistics to the given .pkl file.

def save_stats(stats, path, verbose=True):
    if verbose:
        print(f'Saving feature statistics to {path} ...')
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(stats, f)

#----------------------------------------------------------------------------
# Calculate feature statistics for the given image batches
# in a distributed fashion. Returns an iterable that yields
# dnnlib.EasyDict(stats, images, batch_idx, num_batches)

def calculate_stats_for_iterable(
    image_iter,                         # Iterable of image batches: NCHW, uint8, 3 channels.
    metrics     = ['fid', 'fd_dinov2'], # Metrics to compute the statistics for.
    verbose     = True,                 # Enable status prints?
    dest_path   = None,                 # Where to save the statistics. None = do not save.
    device      = torch.device('cuda'), # Which compute device to use.
):
    # Initialize.
    num_batches = len(image_iter)
    detectors = [get_detector(metric, verbose=verbose) for metric in metrics]
    if verbose:
        dist.print0('Calculating feature statistics...')

    # Convenience wrapper for torch.distributed.all_reduce().
    def all_reduce(x):
        x = x.clone()
        torch.distributed.all_reduce(x)
        return x

    # Return an iterable over the batches.
    class StatsIterable:
        def __len__(self):
            return num_batches

        def __iter__(self):
            state = [dnnlib.EasyDict(metric=metric, detector=detector) for metric, detector in zip(metrics, detectors)]
            for s in state:
                s.cum_mu = torch.zeros([s.detector.feature_dim], dtype=torch.float64, device=device)
                s.cum_sigma = torch.zeros([s.detector.feature_dim, s.detector.feature_dim], dtype=torch.float64, device=device)
            cum_images = torch.zeros([], dtype=torch.int64, device=device)

            # Loop over batches.
            for batch_idx, images in enumerate(image_iter):
                if isinstance(images, dict) and 'images' in images: # dict(images)
                    images = images['images']
                elif isinstance(images, (tuple, list)) and len(images) == 2: # (images, labels)
                    images = images[0]
                images = torch.as_tensor(images).to(device)

                # Accumulate statistics.
                if images is not None:
                    for s in state:
                        features = s.detector(images).to(torch.float64)
                        s.cum_mu += features.sum(0)
                        s.cum_sigma += features.T @ features
                    cum_images += images.shape[0]

                # Output results.
                r = dnnlib.EasyDict(stats=None, images=images, batch_idx=batch_idx, num_batches=num_batches)
                r.num_images = int(all_reduce(cum_images).cpu())
                if batch_idx == num_batches - 1:
                    assert r.num_images >= 2
                    r.stats = dict(num_images=r.num_images)
                    for s in state:
                        mu = all_reduce(s.cum_mu) / r.num_images
                        sigma = (all_reduce(s.cum_sigma) - mu.ger(mu) * r.num_images) / (r.num_images - 1)
                        r.stats[s.metric] = dict(mu=mu.cpu().numpy(), sigma=sigma.cpu().numpy())
                    if dest_path is not None and dist.get_rank() == 0:
                        save_stats(stats=r.stats, path=dest_path, verbose=False)
                yield r

    return StatsIterable()

#----------------------------------------------------------------------------
# Calculate feature statistics for the given directory or ZIP of images
# in a distributed fashion. Returns an iterable that yields
# dnnlib.EasyDict(stats, images, batch_idx, num_batches)

def calculate_stats_for_files(
    image_path,             # Path to a directory or ZIP file containing the images.
    num_images      = None, # Number of images to use. None = all available images.
    seed            = 0,    # Random seed for selecting the images.
    max_batch_size  = 64,   # Maximum batch size.
    num_workers     = 2,    # How many subprocesses to use for data loading.
    prefetch_factor = 2,    # Number of images loaded in advance by each worker.
    verbose         = True, # Enable status prints?
    **stats_kwargs,         # Arguments for calculate_stats_for_iterable().
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # List images.
    if verbose:
        dist.print0(f'Loading images from {image_path} ...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_images, random_seed=seed)
    if num_images is not None and len(dataset_obj) < num_images:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_images}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = max((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(dataset_obj)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches,
        num_workers=num_workers, prefetch_factor=(prefetch_factor if num_workers > 0 else None))

    # Return an interable for calculating the statistics.
    return calculate_stats_for_iterable(image_iter=data_loader, verbose=verbose, **stats_kwargs)

#----------------------------------------------------------------------------
# Calculate metrics based on the given feature statistics.

def calculate_metrics_from_stats(
    stats,                          # Feature statistics of the generated images.
    ref,                            # Reference statistics of the dataset. Can be a path or URL.
    metrics = ['fid', 'fd_dinov2'], # List of metrics to compute.
    verbose = True,                 # Enable status prints?
):
    if isinstance(ref, str):
        ref = load_stats(ref, verbose=verbose)
    results = dict()
    for metric in metrics:
        if metric not in stats or metric not in ref:
            if verbose:
                print(f'No statistics computed for {metric} -- skipping.')
            continue
        if verbose:
            print(f'Calculating {metric}...')
        m = np.square(stats[metric]['mu'] - ref[metric]['mu']).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(stats[metric]['sigma'], ref[metric]['sigma']), disp=False)
        value = float(np.real(m + np.trace(stats[metric]['sigma'] + ref[metric]['sigma'] - s * 2)))
        results[metric] = value
        if verbose:
            print(f'{metric} = {value:g}')
    return results

#----------------------------------------------------------------------------
# Parse a comma separated list of strings.

def parse_metric_list(s):
    metrics = s if isinstance(s, list) else s.split(',')
    for metric in metrics:
        if metric not in metric_specs:
            raise click.ClickException(f'Invalid metric "{metric}"')
    return metrics

#----------------------------------------------------------------------------
# Main command line.

@click.group()
def cmdline():
    """Calculate evaluation metrics (FID and FD_DINOv2).

    Examples:

    \b
    # Generate 50000 images using 8 GPUs and save them as out/*/*.png
    torchrun --standalone --nproc_per_node=8 generate_images.py \\
        --preset=edm2-img512-xxl-guid-fid --outdir=out --subdirs --seeds=0-49999

    \b
    # Calculate metrics for a random subset of 50000 images in out/
    python calculate_metrics.py calc --images=out \\
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl

    \b
    # Calculate metrics directly for a given model without saving any images
    torchrun --standalone --nproc_per_node=8 calculate_metrics.py gen \\
        --net=https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-s-2147483-0.130.pkl \\
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl \\
        --seed=123456789

    \b
    # Compute dataset reference statistics
    python calculate_metrics.py ref --data=datasets/my-dataset.zip \\
        --dest=fid-refs/my-dataset.pkl
    """

#----------------------------------------------------------------------------
# 'calc' subcommand.

@cmdline.command()
@click.option('--images', 'image_path',     help='Path to the images', metavar='PATH|ZIP',                  type=str, required=True)
@click.option('--ref', 'ref_path',          help='Dataset reference statistics ', metavar='PKL|NPZ|URL',    type=str, required=True)
@click.option('--metrics',                  help='List of metrics to compute', metavar='LIST',              type=parse_metric_list, default='fid,fd_dinov2', show_default=True)
@click.option('--num', 'num_images',        help='Number of images to use', metavar='INT',                  type=click.IntRange(min=2), default=50000, show_default=True)
@click.option('--seed',                     help='Random seed for selecting the images', metavar='INT',     type=int, default=0, show_default=True)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                       type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--workers', 'num_workers',   help='Subprocesses to use for data loading', metavar='INT',     type=click.IntRange(min=0), default=2, show_default=True)

def calc(ref_path, metrics, **opts):
    """Calculate metrics for a given set of images."""
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    if dist.get_rank() == 0:
        ref = load_stats(path=ref_path) # do this first, just in case it fails
    stats_iter = calculate_stats_for_files(metrics=metrics, **opts)
    for r in tqdm.tqdm(stats_iter, unit='batch', disable=(dist.get_rank() != 0)):
        pass
    if dist.get_rank() == 0:
        calculate_metrics_from_stats(stats=r.stats, ref=ref, metrics=metrics)
    torch.distributed.barrier()

#----------------------------------------------------------------------------
# 'gen' subcommand.

@cmdline.command()
@click.option('--net',                      help='Network pickle filename', metavar='PATH|URL',             type=str, required=True)
@click.option('--ref', 'ref_path',          help='Dataset reference statistics ', metavar='PKL|NPZ|URL',    type=str, required=True)
@click.option('--metrics',                  help='List of metrics to compute', metavar='LIST',              type=parse_metric_list, default='fid,fd_dinov2', show_default=True)
@click.option('--num', 'num_images',        help='Number of images to generate', metavar='INT',             type=click.IntRange(min=2), default=50000, show_default=True)
@click.option('--seed',                     help='Random seed for the first image', metavar='INT',          type=int, default=0, show_default=True)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                       type=click.IntRange(min=1), default=32, show_default=True)

def gen(net, ref_path, metrics, num_images, seed, **opts):
    """Calculate metrics for a given model using default sampler settings."""
    dist.init()
    if dist.get_rank() == 0:
        ref = load_stats(path=ref_path) # do this first, just in case it fails
    image_iter = generate_images.generate_images(net=net, seeds=range(seed, seed + num_images), **opts)
    stats_iter = calculate_stats_for_iterable(image_iter, metrics=metrics)
    for r in tqdm.tqdm(stats_iter, unit='batch', disable=(dist.get_rank() != 0)):
        pass
    if dist.get_rank() == 0:
        calculate_metrics_from_stats(stats=r.stats, ref=ref, metrics=metrics)
    torch.distributed.barrier()

#----------------------------------------------------------------------------
# 'ref' subcommand.

@cmdline.command()
@click.option('--data', 'image_path',       help='Path to the dataset', metavar='PATH|ZIP',             type=str, required=True)
@click.option('--dest', 'dest_path',        help='Destination file', metavar='PKL',                     type=str, required=True)
@click.option('--metrics',                  help='List of metrics to compute', metavar='LIST',          type=parse_metric_list, default='fid,fd_dinov2', show_default=True)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                   type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--workers', 'num_workers',   help='Subprocesses to use for data loading', metavar='INT', type=click.IntRange(min=0), default=2, show_default=True)

def ref(**opts):
    """Calculate dataset reference statistics for 'calc' and 'gen'."""
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    stats_iter = calculate_stats_for_files(**opts)
    for _r in tqdm.tqdm(stats_iter, unit='batch', disable=(dist.get_rank() != 0)):
        pass

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
