# Copyright (c) 2026, Vinay Saji Mathew.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Moment-matching distillation loss (Salimans et al., 2024)."""

import torch
from torch_utils import persistence
from torch_utils import training_stats

from training.distillation.moment_matching.momentmatching_ops import sample_timesteps_mm, sample_conditional_posterior

@persistence.import_hook
def _absolute_momentmatching_import(meta):
    # Older moment-matching snapshots pickled this module with a relative import.
    old = 'from .momentmatching_ops import'
    new = 'from training.distillation.moment_matching.momentmatching_ops import'
    if isinstance(meta.module_src, str) and old in meta.module_src:
        meta.module_src = meta.module_src.replace(old, new)
    return meta


@persistence.persistent_class
class EDMMomentMatchLoss:
    """Alternating optimization, Algorithm 2 of Salimans et al. (2024).

    net is the student g_eta, teacher_net the frozen teacher g_theta and aux_net the
    auxiliary denoiser g_phi. Even steps update phi:
        L(phi) = w(s) * { ||x_tilde - g_phi(z_s)||^2 + ||g_theta(z_s) - g_phi(z_s)||^2 }
    Odd steps update eta:
        L(eta) = w(s) * x_tilde^T sg[g_phi(z_s) - g_theta(z_s)]
    """

    def __init__(
        self,
        teacher_net,
        aux_net,
        k: int = 8,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        sigma_data: float = 0.5,
        weight_mode: str = "edm",
        sync_dropout: bool = True,
    ):
        assert k >= 1, "Student steps k must be >= 1"
        assert weight_mode in (
            "edm", "vlike", "flat",
            "snr", "snr+1", "karras", "sqrt_karras", "truncated-snr", "uniform",
        )

        self.teacher_net = teacher_net.eval().requires_grad_(False)
        self.aux_net = aux_net
        self.k = int(k)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self.sigma_data = float(sigma_data)
        self.weight_mode = weight_mode
        self.sync_dropout = bool(sync_dropout)

        self._step_n = 0

    def set_step_n(self, n: int) -> None:
        self._step_n = int(n)

    def _weight(self, sigma: torch.Tensor) -> torch.Tensor:
        if self.weight_mode == "edm":
            return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        if self.weight_mode == "vlike":
            return (1.0 / (sigma ** 2)) + 1.0
        if self.weight_mode == "flat":
            return torch.ones_like(sigma)
        snr = 1.0 / (sigma ** 2 + 1e-20)
        if self.weight_mode == "snr":
            return snr
        if self.weight_mode == "snr+1":
            return snr + 1.0
        if self.weight_mode == "karras":
            return snr + (1.0 / (self.sigma_data ** 2))
        if self.weight_mode == "sqrt_karras":
            return torch.sqrt(sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data)
        if self.weight_mode == "truncated-snr":
            return torch.clamp(snr, min=1.0)
        assert self.weight_mode == "uniform"
        return torch.ones_like(sigma)

    def __call__(self, net, images, labels=None):
        device = images.device
        batch_size = images.shape[0]
        is_even = (self._step_n % 2 == 0)
        y = images

        ts = sample_timesteps_mm(
            batch_size=batch_size,
            k=self.k,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            device=device,
        )
        sigma_t_vec = ts["sigma_t"]
        sigma_s_vec = ts["sigma_s"]
        sigma_t = sigma_t_vec.reshape(batch_size, 1, 1, 1)

        eps = torch.randn_like(y).to(torch.float64)
        y64 = y.to(torch.float64)
        z_t = y64 + sigma_t * eps
        z_t_f32 = z_t.float()

        # The weight is evaluated at sigma_s.
        weight = self._weight(sigma_s_vec.float()).reshape(batch_size, 1, 1, 1)

        if is_even:
            loss = self._loss_even(net, z_t_f32, sigma_t_vec, sigma_s_vec, labels, weight, batch_size)
        else:
            loss = self._loss_odd(net, z_t_f32, sigma_t_vec, sigma_s_vec, labels, weight, batch_size)

        with torch.no_grad():
            training_stats.report('Loss/mm', loss)
            training_stats.report('MM/sigma_t', sigma_t_vec.float().mean())
            training_stats.report('MM/sigma_s', sigma_s_vec.float().mean())
            training_stats.report('MM/is_even', torch.as_tensor(float(is_even), device=device))

        return loss

    def _loss_even(self, net, z_t_f32, sigma_t_vec, sigma_s_vec, labels, weight, batch_size):
        # Student is frozen on even steps.
        with torch.no_grad():
            x_tilde = net(z_t_f32, sigma_t_vec, labels).to(torch.float32)

        z_s = sample_conditional_posterior(z_t_f32, x_tilde, sigma_t_vec, sigma_s_vec)
        z_s_f32 = z_s.float()

        # Replay the aux dropout RNG for the teacher pass.
        if self.sync_dropout:
            rng_state = torch.cuda.get_rng_state()

        g_phi_zs = self.aux_net(z_s_f32, sigma_s_vec, labels).to(torch.float32)

        with torch.no_grad():
            if self.sync_dropout:
                torch.cuda.set_rng_state(rng_state)
            g_theta_zs = self.teacher_net(z_s_f32, sigma_s_vec, labels).to(torch.float32)

        term1 = ((x_tilde.detach() - g_phi_zs) ** 2).sum(dim=[1, 2, 3])
        term2 = ((g_theta_zs.detach() - g_phi_zs) ** 2).sum(dim=[1, 2, 3])
        per_sample = weight.reshape(-1) * (term1 + term2)
        return per_sample.reshape(batch_size, 1, 1, 1)

    def _loss_odd(self, net, z_t_f32, sigma_t_vec, sigma_s_vec, labels, weight, batch_size):
        x_tilde = net(z_t_f32, sigma_t_vec, labels).to(torch.float32)

        # Stop-gradient through z_s.
        z_s = sample_conditional_posterior(z_t_f32, x_tilde.detach(), sigma_t_vec, sigma_s_vec)
        z_s_f32 = z_s.float()

        with torch.no_grad():
            # Replay the aux dropout RNG for the teacher pass.
            if self.sync_dropout:
                rng_state = torch.cuda.get_rng_state()

            g_phi_zs = self.aux_net(z_s_f32, sigma_s_vec, labels).to(torch.float32)

            if self.sync_dropout:
                torch.cuda.set_rng_state(rng_state)

            g_theta_zs = self.teacher_net(z_s_f32, sigma_s_vec, labels).to(torch.float32)

        direction = (g_phi_zs - g_theta_zs).detach()
        dot = (x_tilde * direction).sum(dim=[1, 2, 3])
        per_sample = weight.reshape(-1) * dot
        return per_sample.reshape(batch_size, 1, 1, 1)
