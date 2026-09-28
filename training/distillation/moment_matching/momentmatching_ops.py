# Copyright (c) 2026, Vinay Saji Mathew.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Time/sigma mapping, timestep sampling and the conditional posterior sampler for
moment matching (Salimans et al., 2024). All arithmetic is done in float64."""

import torch


def time_to_sigma(
    t: torch.Tensor,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
) -> torch.Tensor:
    """Karras schedule: t=0 -> sigma_min, t=1 -> sigma_max."""
    t = t.to(torch.float64)
    inv_rho = 1.0 / rho
    sigma = ((1.0 - t) * sigma_min ** inv_rho + t * sigma_max ** inv_rho) ** rho
    return sigma


def sample_timesteps_mm(
    batch_size: int,
    k: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
) -> dict:
    """Algorithm 2 of Salimans et al. (2024), step 1:
    s ~ U(0, 1), t = min(s + U(0, 1/k), 1).
    """
    s = torch.rand(batch_size, device=device, dtype=torch.float64)
    delta_t = torch.rand(batch_size, device=device, dtype=torch.float64) / k
    t = torch.clamp(s + delta_t, max=1.0)

    sigma_s = time_to_sigma(s, sigma_min, sigma_max, rho)
    sigma_t = time_to_sigma(t, sigma_min, sigma_max, rho)

    return dict(
        sigma_t=sigma_t,
        sigma_s=sigma_s,
        time_s=s,
        time_t=t,
    )


def sample_conditional_posterior(
    z_t: torch.Tensor,
    x_pred: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_s: torch.Tensor,
    randn_like=torch.randn_like,
) -> torch.Tensor:
    """Sample z_s ~ q(z_s | z_t, x_pred) = N(mu, lambda^2 I) under z = x + sigma * eps, with
    mu = (1 - sigma_s^2/sigma_t^2) * x + (sigma_s^2/sigma_t^2) * z_t and
    lambda^2 = sigma_s^2 * (sigma_t^2 - sigma_s^2) / sigma_t^2.
    """
    out_dtype = z_t.dtype
    z_t64 = z_t.to(torch.float64)
    x64 = x_pred.to(torch.float64)

    st = sigma_t.to(torch.float64).reshape(-1, 1, 1, 1)
    ss = sigma_s.to(torch.float64).reshape(-1, 1, 1, 1)

    st_sq = st * st
    ss_sq = ss * ss
    diff_sq = st_sq - ss_sq

    mu = (diff_sq / st_sq) * x64 + (ss_sq / st_sq) * z_t64
    lam = torch.sqrt(ss_sq * diff_sq / st_sq)

    eps = randn_like(z_t64).to(torch.float64)
    z_s = mu + lam * eps
    return z_s.to(out_dtype)
