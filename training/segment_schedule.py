# Copyright (c) 2026, Vinay Saji Mathew.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Segment boundaries 0 = t_0 < ... < t_M = arctan(sigma_max / sigma_d) for
multistep TrigFlow consistency models; segment k covers (t_k, t_{k+1}]."""

from __future__ import annotations

import math
from typing import Tuple

import torch


def t_max_from_sigma(sigma_max: float, sigma_data: float) -> float:
    return float(math.atan(float(sigma_max) / float(sigma_data)))


def build_boundaries(
    num_segments: int,
    t_max: float,
    schedule: str = 'uniform_t',
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Return segment boundaries ``[t_0, ..., t_M]`` with ``t_0=0``, ``t_M=t_max``."""
    M = int(num_segments)
    if M < 1:
        raise ValueError(f'num_segments must be >= 1, got {M}')
    t_max_f = float(t_max)
    if schedule == 'uniform_t':
        bounds = torch.linspace(0.0, t_max_f, M + 1, device=device, dtype=dtype)
    elif schedule == 'uniform_log_snr':
        # Uniform in log(tan(t)), then convert back to t.
        eps = 1e-6
        lo = math.log(max(math.tan(eps), eps))
        hi = math.log(max(math.tan(t_max_f), eps))
        u = torch.linspace(lo, hi, M + 1, device=device, dtype=dtype)
        bounds = torch.atan(u.exp())
        bounds[0] = 0.0
        bounds[-1] = t_max_f
    else:
        raise ValueError(f'Unknown boundary schedule "{schedule}"')
    return bounds


def lookup_segment(
    t: torch.Tensor,
    boundaries: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map each ``t`` to segment index ``k`` with ``t in (t_k, t_{k+1}]`` (``t=0`` -> ``k=0``).

    Returns:
        k:        ``[B]`` int64 segment indices in ``[0, M-1]``
        t_k:      ``[B]`` segment bottoms
        t_k1:     ``[B]`` segment tops
        delta_t:  ``[B, 1, 1, 1]`` with ``delta_t = t - t_k``
    """
    t_flat = t.reshape(-1).to(torch.float32)
    bounds = boundaries.to(device=t_flat.device, dtype=torch.float32)
    M = bounds.numel() - 1
    # k s.t. bounds[k] < t <= bounds[k+1]; t=0 maps to k=0.
    k = (t_flat.unsqueeze(1) > bounds[:-1].unsqueeze(0)).sum(dim=1) - 1
    k = k.clamp(min=0, max=M - 1)
    t_k = bounds[k]
    t_k1 = bounds[k + 1]
    delta_t = (t_flat - t_k).reshape(-1, 1, 1, 1)
    return k, t_k, t_k1, delta_t


def sample_time_option_a(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    P_mean: float,
    P_std: float,
    sigma_data: float,
) -> torch.Tensor:
    """Global log-normal proposal on tau = log(sigma_d * tan(t)), as in single-segment sCM."""
    tau = torch.randn([batch_size, 1, 1, 1], device=device, dtype=dtype) * P_std + P_mean
    return torch.atan(tau.exp() / sigma_data)


def sample_time_option_b(
    batch_size: int,
    boundaries: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Segment-balanced proposal (Heek et al., 2024): sample a segment
    ``k ~ U{0..M-1}`` uniformly, then ``t ~ U(t_k, t_{k+1}]`` within it."""
    bounds = boundaries.to(device=device, dtype=dtype)
    M = bounds.numel() - 1
    k = torch.randint(0, M, (batch_size,), device=device)
    t_lo = bounds[k]
    t_hi = bounds[k + 1]
    u = torch.rand(batch_size, device=device, dtype=dtype)
    t_flat = t_lo + u * (t_hi - t_lo)
    return t_flat.reshape(batch_size, 1, 1, 1)
