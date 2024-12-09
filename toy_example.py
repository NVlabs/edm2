# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""2D toy example from the paper "Guiding a Diffusion Model with a Bad Version of Itself"."""

import os
import copy
import pickle
import warnings
import functools
import numpy as np
import torch
import matplotlib.pyplot as plt
import click
import tqdm
import dnnlib
from torch_utils import persistence

warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`')

#----------------------------------------------------------------------------
# Multivariate mixture of Gaussians. Allows efficient evaluation of the
# probability density function (PDF) and score vector, as well as sampling,
# using the GPU. The distribution can be optionally smoothed by applying heat
# diffusion (sigma >= 0) on a per-sample basis.

class GaussianMixture(torch.nn.Module):
    def __init__(self,
        phi,                        # Per-component weight: [comp]
        mu,                         # Per-component mean: [comp, dim]
        Sigma,                      # Per-component covariance matrix: [comp, dim, dim]
        sample_lut_size = 64<<10,   # Lookup table size for efficient sampling.
    ):
        super().__init__()
        self.register_buffer('phi', torch.tensor(np.asarray(phi) / np.sum(phi), dtype=torch.float32))
        self.register_buffer('mu', torch.tensor(np.asarray(mu), dtype=torch.float32))
        self.register_buffer('Sigma', torch.tensor(np.asarray(Sigma), dtype=torch.float32))

        # Precompute eigendecompositions of Sigma for efficient heat diffusion.
        L, Q = torch.linalg.eigh(self.Sigma) # Sigma = Q @ L @ Q
        self.register_buffer('_L', L) # L: [comp, dim, dim]
        self.register_buffer('_Q', Q) # Q: [comp, dim, dim]

        # Precompute lookup table for efficient sampling.
        self.register_buffer('_sample_lut', torch.zeros(sample_lut_size, dtype=torch.int64))
        phi_ranges = (torch.cat([torch.zeros_like(self.phi[:1]), self.phi.cumsum(0)]) * sample_lut_size + 0.5).to(torch.int32)
        for idx, (begin, end) in enumerate(zip(phi_ranges[:-1], phi_ranges[1:])):
            self._sample_lut[begin : end] = idx

    # Evaluate the terms needed for calculating PDF and score.
    def _eval(self, x, sigma=0):                                                    # x: [..., dim], sigma: [...]
        L = self._L + sigma[..., None, None] ** 2                                   # L' = L + sigma * I: [..., dim]
        d = L.prod(-1)                                                              # d = det(Sigma') = det(Q @ L' @ Q) = det(L'): [...]
        y = self.mu - x[..., None, :]                                               # y = mu - x: [..., comp, dim]
        z = torch.einsum('...ij,...j,...kj,...k->...i', self._Q, 1 / L, self._Q, y) # z = inv(Sigma') @ (mu - x): [..., comp, dim]
        c = self.phi / (((2 * np.pi) ** x.shape[-1]) * d).sqrt()                    # normalization factor of N(x; mu, Sigma')
        w = c * (-1/2 * torch.einsum('...i,...i->...', y, z)).exp()                 # w = N(x; mu, Sigma'): [..., comp]
        return z, w

    # Calculate p(x; sigma) for the given sample points, processing at most the given number of samples at a time.
    def pdf(self, x, sigma=0, max_batch_size=1<<14):
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=x.device).broadcast_to(x.shape[:-1])
        x_batches = x.flatten(0, -2).split(max_batch_size)
        sigma_batches = sigma.flatten().split(max_batch_size)
        pdf_batches = [self._eval(xx, ss)[1].sum(-1) for xx, ss in zip(x_batches, sigma_batches)]
        return torch.cat(pdf_batches).reshape(x.shape[:-1]) # x.shape[:-1]

    # Calculate log(p(x; sigma)) for the given sample points, processing at most the given number of samples at a time.
    def logp(self, x, sigma=0, max_batch_size=1<<14):
        return self.pdf(x, sigma, max_batch_size).log()

    # Calculate \nabla_x log(p(x; sigma)) for the given sample points.
    def score(self, x, sigma=0):
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=x.device).broadcast_to(x.shape[:-1])
        z, w = self._eval(x, sigma)
        w = w[..., None]
        return (w * z).sum(-2) / w.sum(-2) # x.shape

    # Draw the given number of random samples from p(x; sigma).
    def sample(self, shape, sigma=0, generator=None):
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=self.mu.device).broadcast_to(shape)
        i = self._sample_lut[torch.randint(len(self._sample_lut), size=sigma.shape, device=sigma.device, generator=generator)]
        L = self._L[i] + sigma[..., None] ** 2                                                  # L' = L + sigma * I: [..., dim]
        x = torch.randn(L.shape, device=sigma.device, generator=generator)                      # x ~ N(0, I): [..., dim]
        y = torch.einsum('...ij,...j,...kj,...k->...i', self._Q[i], L.sqrt(), self._Q[i], x)    # y = sqrt(Sigma') @ x: [..., dim]
        return y + self.mu[i] # [..., dim]

#----------------------------------------------------------------------------
# Construct a ground truth 2D distribution for the given set of classes
# ('A', 'B', or 'AB').

@functools.lru_cache(None)
def gt(classes='A', device=torch.device('cpu'), seed=2, origin=np.array([0.0030, 0.0325]), scale=np.array([1.3136, 1.3844])):
    rnd = np.random.RandomState(seed)
    comps = []

    # Recursive function to generate a given branch of the distribution.
    def recurse(cls, depth, pos, angle):
        if depth >= 7:
            return

        # Choose parameters for the current branch.
        dir = np.array([np.cos(angle), np.sin(angle)])
        dist = 0.292 * (0.8 ** depth) * (rnd.randn() * 0.2 + 1)
        thick = 0.2 * (0.8 ** depth) / dist
        size = scale * dist * 0.06

        # Represent the current branch as a sequence of Gaussian components.
        for t in np.linspace(0.07, 0.93, num=8):
            c = dnnlib.EasyDict()
            c.cls = cls
            c.phi = dist * (0.5 ** depth)
            c.mu = (pos + dir * dist * t) * scale
            c.Sigma = (np.outer(dir, dir) + (np.eye(2) - np.outer(dir, dir)) * (thick ** 2)) * np.outer(size, size)
            comps.append(c)

        # Generate each child branch.
        for sign in [1, -1]:
            recurse(cls=cls, depth=(depth + 1), pos=(pos + dir * dist), angle=(angle + sign * (0.7 ** depth) * (rnd.randn() * 0.2 + 1)))

    # Generate each class.
    recurse(cls='A', depth=0, pos=origin, angle=(np.pi * 0.25))
    recurse(cls='B', depth=0, pos=origin, angle=(np.pi * 1.25))

    # Construct a GaussianMixture object for the selected classes.
    sel = [c for c in comps if c.cls in classes]
    distrib = GaussianMixture([c.phi for c in sel], [c.mu for c in sel], [c.Sigma for c in sel])
    return distrib.to(device)

#----------------------------------------------------------------------------
# Low-level primitives used by ToyModel.
# Adapted from training/networks/networks_edm2.py.

def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

@persistence.persistent_class
class MPSiLU(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.silu(x) / 0.596

@persistence.persistent_class
class MPLinear(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_dim, in_dim))

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(self.weight))
        w = normalize(self.weight) / np.sqrt(self.weight[0].numel())
        return x @ w.t()

#----------------------------------------------------------------------------
# Denoiser model for learning 2D toy distributions. Inputs a set of sample
# positions and a single scalar for each, representing the logarithm of the
# corresponding unnormalized probability density. The score vector can then
# be obtained through automatic differentiation.

@persistence.persistent_class
class ToyModel(torch.nn.Module):
    def __init__(self,
        in_dim      = 2,    # Input dimensionality.
        num_layers  = 4,    # Number of hidden layers.
        hidden_dim  = 64,   # Number of hidden features.
        sigma_data  = 0.5,  # Expected standard deviation of the training data.
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.layers = torch.nn.Sequential()
        self.layers.append(MPLinear(in_dim + 2, hidden_dim))
        for _layer_idx in range(num_layers):
            self.layers.append(MPSiLU())
            self.layers.append(MPLinear(hidden_dim, hidden_dim))
        self.gain = torch.nn.Parameter(torch.zeros([]))

    def forward(self, x, sigma=0):
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=x.device).broadcast_to(x.shape[:-1]).unsqueeze(-1)
        x = x / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        y = self.layers(torch.cat([x, sigma.log() / 4, torch.ones_like(sigma)], dim=-1))
        z = (y ** 2).mean(-1) * self.gain / sigma.squeeze(-1) - 0.5 * (x ** 2).sum(-1) # preconditioning
        return z

    def logp(self, x, sigma=0):
        return self(x, sigma)

    def pdf(self, x, sigma=0):
        logp = self.logp(x, sigma=sigma)
        pdf = (logp - logp.max()).exp()
        return pdf

    def score(self, x, sigma=0, graph=False):
        x = x.detach().requires_grad_(True)
        logp = self.logp(x, sigma=sigma)
        score = torch.autograd.grad(outputs=[logp.sum()], inputs=[x], create_graph=graph)[0]
        return score

#----------------------------------------------------------------------------
# Train a 2D toy model with the given parameters.

def do_train(
    classes='A', num_layers=4, hidden_dim=64, batch_size=4<<10, total_iter=4<<10, seed=0,
    P_mean=-2.3, P_std=1.5, sigma_data=0.5, lr_ref=1e-2, lr_iter=512, ema_std=0.010,
    pkl_pattern=None, pkl_iter=256, viz_iter=32,
    device=torch.device('cuda'),
):
    import training.phema
    torch.manual_seed(seed)

    # Initialize model.
    net = ToyModel(num_layers=num_layers, hidden_dim=hidden_dim, sigma_data=sigma_data).to(device).train().requires_grad_(True)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), betas=(0.9, 0.99))

    # Initialize plot.
    if viz_iter is not None:
        plt.ion()
        plt.rcParams['toolbar'] = 'None'
        plt.rcParams['figure.subplot.left'] = plt.rcParams['figure.subplot.bottom'] = 0
        plt.rcParams['figure.subplot.right'] = plt.rcParams['figure.subplot.top'] = 1
        plt.figure(figsize=[12, 12], dpi=75)
        do_plot(ema, elems={'gt_uncond', 'gt_outline'}, device=device)
        plt.gcf().canvas.flush_events()

    # Training loop.
    for iter_idx in tqdm.tqdm(range(total_iter)):

        # Visualize current sample distribution.
        if viz_iter is not None and iter_idx % viz_iter == 0:
            for x in plt.gca().lines: x.remove()
            do_plot(ema, elems={'samples'}, device=device)
            plt.gcf().canvas.flush_events()

        # Execute one training iteration.
        opt.param_groups[0]['lr'] = lr_ref / np.sqrt(max(iter_idx / lr_iter, 1))
        opt.zero_grad()
        sigma = (torch.randn(batch_size, device=device) * P_std + P_mean).exp()
        samples = gt(classes, device).sample(batch_size, sigma)
        gt_scores = gt(classes, device).score(samples, sigma)
        net_scores = net.score(samples, sigma, graph=True)
        loss = (sigma ** 2) * ((gt_scores - net_scores) ** 2).mean(-1)
        loss.mean().backward()
        opt.step()

        # Update EMA.
        beta = training.phema.power_function_beta(std=ema_std, t_next=iter_idx+1, t_delta=1)
        for p_net, p_ema in zip(net.parameters(), ema.parameters()):
            p_ema.lerp_(p_net.detach(), 1 - beta)

        # Save model snapshot.
        if pkl_pattern is not None and (iter_idx + 1) % pkl_iter == 0:
            pkl_path = pkl_pattern % (iter_idx + 1)
            if os.path.dirname(pkl_path):
                os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
            with open(pkl_path, 'wb') as f:
                pickle.dump(copy.deepcopy(ema).cpu(), f)

#----------------------------------------------------------------------------
# Simulate the EDM sampling ODE for the given set of initial sample points.
# Adapted from generate_images.py.

def do_sample(net, x_init, guidance=1, gnet=None, num_steps=32, sigma_min=0.002, sigma_max=5, rho=7):
    # Guided denoiser.
    def denoise(x, sigma):
        score = net.score(x, sigma)
        if gnet is not None:
            score = gnet.score(x, sigma).lerp(score, guidance)
        return x + score * (sigma ** 2)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=x_init.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_cur = x_init
    trajectory = [x_cur]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1

        # Euler step.
        d_cur = (x_cur - denoise(x_cur, t_cur)) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            d_prime = (x_next - denoise(x_next, t_next)) / t_next
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

        # Record trajectory.
        x_cur = x_next
        trajectory.append(x_cur)
    return torch.stack(trajectory)

#----------------------------------------------------------------------------
# Draw the given set of plot elements using matplotlib.

def do_plot(
    net=None, guidance=1, gnet=None, elems={'gt_uncond', 'gt_outline', 'samples'},
    view_x=0, view_y=0, view_size=1.6, grid_resolution=400, arrow_len=0.002,
    num_samples=1<<13, seed=1, sample_distance=0, sigma_max=5,
    device=torch.device('cuda'),
):
    # Generate initial samples.
    if any(x.startswith(y) for x in elems for y in ['samples', 'trajectories', 'scores']):
        samples = gt('A', device).sample(num_samples, sigma_max, generator=torch.Generator(device).manual_seed(seed))
        if sample_distance > 0:
            ok = torch.ones(len(samples), dtype=torch.bool, device=device)
            for i in range(1, len(samples)):
                ok[i] = (samples[i] - samples[:i][ok[:i]]).square().sum(-1).sqrt().min() >= sample_distance
            samples = samples[ok]

    # Run sampler.
    if any(x.startswith(y) for x in elems for y in ['samples', 'trajectories']):
        trajectories = do_sample(net=(net or gt('A', device)), x_init=samples, guidance=guidance, gnet=gnet, sigma_max=sigma_max)

    # Initialize plot.
    gridx = torch.linspace(view_x - view_size, view_x + view_size, steps=grid_resolution, device=device)
    gridy = torch.linspace(view_y - view_size, view_y + view_size, steps=grid_resolution, device=device)
    gridxy = torch.stack(torch.meshgrid(gridx, gridy, indexing='xy'), axis=-1)
    plt.xlim(float(gridx[0]), float(gridx[-1]))
    plt.ylim(float(gridy[0]), float(gridy[-1]))
    plt.gca().set_aspect('equal')
    plt.gca().set_axis_off()

    # Plot helper functions.
    def contours(values, levels, colors=None, cmap=None, alpha=1, linecolors='black', linealpha=1, linewidth=2.5):
        values = -(values.max() - values).sqrt().cpu().numpy()
        plt.contourf(gridx.cpu().numpy(), gridy.cpu().numpy(), values, levels=levels, antialiased=True, extend='max', colors=colors, cmap=cmap, alpha=alpha)
        plt.contour(gridx.cpu().numpy(), gridy.cpu().numpy(), values, levels=levels, antialiased=True, colors=linecolors, alpha=linealpha, linestyles='solid', linewidths=linewidth)
    def lines(pos, color='black', alpha=1):
        plt.plot(*pos.cpu().numpy().T, '-', linewidth=5, solid_capstyle='butt', color=color, alpha=alpha)
    def arrows(pos, dir, color='black', alpha=1):
        plt.quiver(*pos.cpu().numpy().T, *dir.cpu().numpy().T * arrow_len, scale=0.6, width=5e-3, headwidth=4, headlength=3, headaxislength=2.5, capstyle='round', color=color, alpha=alpha)
    def points(pos, color='black', alpha=1, size=30):
        plt.plot(*pos.cpu().numpy().T, '.', markerfacecolor='black', markeredgecolor='none', color=color, alpha=alpha, markersize=size)

    # Draw requested plot elements.
    if 'p_net' in elems:            contours(net.logp(gridxy, sigma_max), levels=np.linspace(-2.5, 2.5, num=20)[1:-1], cmap='Greens', linealpha=0.2)
    if 'p_gnet' in elems:           contours(gnet.logp(gridxy, sigma_max), levels=np.linspace(-2.5, 3.5, num=20)[1:-1], cmap='Reds', linealpha=0.2)
    if 'p_ratio' in elems:          contours(net.logp(gridxy, sigma_max) - gnet.logp(gridxy, sigma_max), levels=np.linspace(-2.2, 1.0, num=20)[1:-1], cmap='Blues', linealpha=0.2)
    if 'gt_uncond' in elems:        contours(gt('AB', device).logp(gridxy), levels=[-2.12, 0], colors=[[0.9,0.9,0.9]], linecolors=[[0.7,0.7,0.7]], linewidth=1.5)
    if 'gt_outline' in elems:       contours(gt('A', device).logp(gridxy), levels=[-2.12, 0], colors=[[1.0,0.8,0.6]], linecolors=[[0.8,0.6,0.5]], linewidth=1.5)
    if 'gt_smax' in elems:          contours(gt('A', device).logp(gridxy, sigma_max), levels=[-1.41, 0], colors=['C1'], alpha=0.2, linealpha=0.2)
    if 'gt_shaded' in elems:        contours(gt('A', device).logp(gridxy), levels=np.linspace(-2.5, 3.07, num=15)[1:-1], cmap='Oranges', linealpha=0.2)
    if 'trajectories' in elems:     lines(trajectories.transpose(0, 1), alpha=0.3)
    if 'scores_net' in elems:       arrows(samples, net.score(samples, sigma_max), color='C2')
    if 'scores_gnet' in elems:      arrows(samples, gnet.score(samples, sigma_max), color='C3')
    if 'scores_ratio' in elems:     arrows(samples, net.score(samples, sigma_max) - gnet.score(samples, sigma_max), color='C0')
    if 'samples' in elems:          points(trajectories[-1], size=15, alpha=0.25)
    if 'samples_before' in elems:   points(samples)
    if 'samples_after' in elems:    points(trajectories[-1])

#----------------------------------------------------------------------------
# Main command line.

@click.group()
def cmdline():
    """2D toy example from the paper "Guiding a Diffusion Model with a Bad Version of Itself".

    Examples:

    \b
    # Visualize sampling distributions using autoguidance.
    python toy_example.py plot

    \b
    # Same, but save the plot as PNG instead of displaying it.
    python toy_example.py plot --save=out.png

    \b
    # Same, but specify the models explicitly.
    python toy_example.py plot \\
        --net=https://nvlabs-fi-cdn.nvidia.com/edm2/toy-example/clsA-layers04-dim64/iter4096.pkl \\
        --gnet=https://nvlabs-fi-cdn.nvidia.com/edm2/toy-example/clsA-layers04-dim32/iter0512.pkl \\
        --guidance=3

    \b
    # Same, but using classifier-free guidance.
    python toy_example.py plot \\
        --net=https://nvlabs-fi-cdn.nvidia.com/edm2/toy-example/clsA-layers04-dim64/iter4096.pkl \\
        --gnet=https://nvlabs-fi-cdn.nvidia.com/edm2/toy-example/clsAB-layers04-dim32/iter0512.pkl \\
        --guidance=4

    \b
    # Retrain the main model and visualize progress.
    python toy_example.py train

    \b
    # Retrain the main model and save snapshots.
    python toy_example.py train \\
        --outdir=toy-example/clsA-layers04-dim64 \\
        --cls=A --layers=4 --dim=64 --viz=false
    """
    if os.environ.get('WORLD_SIZE', '1') != '1':
        raise click.ClickException('Distributed execution is not supported.')

#----------------------------------------------------------------------------
# 'train' subcommand.

@cmdline.command()
@click.option('--outdir',   help='Output directory', metavar='DIR',     type=str, default=None)
@click.option('--cls',      help='Target classes', metavar='A|B|AB',    type=str, default='A', show_default=True)
@click.option('--layers',   help='Number of layers', metavar='INT',     type=int, default=4, show_default=True)
@click.option('--dim',      help='Hidden dimension', metavar='INT',     type=int, default=64, show_default=True)
@click.option('--viz',      help='Visualize progress?', metavar='BOOL', type=bool, default=True, show_default=True)
def train(outdir, cls, layers, dim, viz):
    """Train a 2D toy model with the given parameters."""
    if outdir is not None:
        print(f'Will save snapshots to {outdir}')
    pkl_pattern = f'{outdir}/iter%04d.pkl' if outdir is not None else None
    viz_iter = 32 if viz else None
    print('Training...')
    do_train(pkl_pattern=pkl_pattern, classes=cls, num_layers=layers, hidden_dim=dim, viz_iter=viz_iter)
    print('Done.')

#----------------------------------------------------------------------------
# 'plot' subcommand.

@cmdline.command()
@click.option('--net',      help='Main model  [default: download]', metavar='PKL|URL',          type=str, default='https://nvlabs-fi-cdn.nvidia.com/edm2/toy-example/clsA-layers04-dim64/iter4096.pkl')
@click.option('--gnet',     help='Guiding model  [default: autoguidance]', metavar='PKL|URL',   type=str, default='https://nvlabs-fi-cdn.nvidia.com/edm2/toy-example/clsA-layers04-dim32/iter0512.pkl')
@click.option('--guidance', help='Guidance weight', metavar='FLOAT',                            type=float, default=3, show_default=True)
@click.option('--save',     help='Save figure, do not display', metavar='PNG|PDF',              type=str, default=None)
def plot(net, gnet, guidance, save, device=torch.device('cuda')):
    """Visualize sampling distributions with and without guidance."""
    print('Loading models...')
    if isinstance(net, str):
        with dnnlib.util.open_url(net) as f:
            net = pickle.load(f).to(device)
    if isinstance(gnet, str):
        with dnnlib.util.open_url(gnet) as f:
            gnet = pickle.load(f).to(device)

    # Initialize plot.
    print('Drawing plots...')
    plt.rcParams['font.size'] = 28
    plt.figure(figsize=[48, 25], dpi=40, tight_layout=True)
    fig1_kwargs = dict(view_x=0.30, view_y=0.30, view_size=1.2, num_samples=1<<14, device=device)
    fig2_kwargs = dict(view_x=0.45, view_y=1.22, view_size=0.3, num_samples=1<<12, device=device, sample_distance=0.045, sigma_max=0.03)

    # Draw first row.
    plt.subplot(2, 4, 1)
    plt.title('Ground truth distribution')
    do_plot(elems={'gt_uncond', 'gt_outline', 'samples'}, **fig1_kwargs)
    plt.subplot(2, 4, 2)
    plt.title('Sample distribution without guidance')
    do_plot(net=net, elems={'gt_uncond', 'gt_outline', 'samples'}, **fig1_kwargs)
    plt.subplot(2, 4, 3)
    plt.title('Sample distribution with guidance')
    do_plot(net=net, gnet=gnet, guidance=guidance, elems={'gt_uncond', 'gt_outline', 'samples'}, **fig1_kwargs)
    plt.subplot(2, 4, 4)
    plt.title('Trajectories without guidance')
    do_plot(net=net, elems={'gt_shaded', 'trajectories', 'samples_after'}, **fig2_kwargs)

    # Draw second row.
    plt.subplot(2, 4, 5)
    plt.title('PDF of main model')
    do_plot(net=net, elems={'p_net', 'gt_smax', 'scores_net', 'samples_before'}, **fig2_kwargs)
    plt.subplot(2, 4, 6)
    plt.title('PDF of guiding model')
    do_plot(net=net, gnet=gnet, elems={'p_gnet', 'gt_smax', 'scores_gnet', 'samples_before'}, **fig2_kwargs)
    plt.subplot(2, 4, 7)
    plt.title('PDF ratio (main / guiding)')
    do_plot(net=net, gnet=gnet, elems={'p_ratio', 'gt_smax', 'scores_ratio', 'samples_before'}, **fig2_kwargs)
    plt.subplot(2, 4, 8)
    plt.title('Trajectories with guidance')
    do_plot(net=net, gnet=gnet, guidance=guidance, elems={'gt_shaded', 'trajectories', 'samples_after'}, **fig2_kwargs)

    # Save or display.
    if save is not None:
        print(f'Saving to {save}')
        if os.path.dirname(save):
            os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save, dpi=80)
    else:
        print('Displaying...')
        plt.show()
    print('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
