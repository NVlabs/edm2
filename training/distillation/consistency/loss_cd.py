# Copyright (c) 2026, Vinay Saji Mathew.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Multistep consistency distillation (MSCD) loss, Heek et al. (2024).
Works with both EDM `EDMPrecond` and EDM2 `Precond` teachers/students."""

import math
import pickle

import torch
import dnnlib
from torch_utils import persistence
from training.networks_edm2 import inplace_norm_flag

from training.distillation.consistency.consistency_ops import (
    make_karras_sigmas,
    partition_edges_by_sigma,
    filter_teacher_edges_by_sigma,
    sample_segment_and_teacher_pair,
    heun_hop_edm,
    inv_ddim_edm,
)


def _huber_loss(x: torch.Tensor, delta: float = 1e-4) -> torch.Tensor:
    abs_x = x.abs()
    quad = torch.minimum(abs_x, torch.as_tensor(delta, device=x.device, dtype=x.dtype))
    return 0.5 * (quad * quad) + (abs_x - quad) * delta


def _pseudo_huber_vector_norm(diff: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Per-sample sqrt(||diff||^2 + eps^2) - eps over C, H, W."""
    norm_sq = (diff * diff).sum(dim=[1, 2, 3])
    return torch.sqrt(norm_sq + eps * eps) - eps


@persistence.persistent_class
class EDMConsistencyDistillLoss:
    def __init__(
        self,
        teacher_net,                 # Frozen teacher.
        teacher_pkl_path=None,       # Teacher pickle, used to re-attach the teacher on resume.
        S: int = 8,                  # Student steps.
        T_start: int = 256,          # Initial teacher edges.
        T_end: int = 1024,           # Final teacher edges.
        T_anneal_kimg: int = 750,    # Log-linear anneal horizon from T_start to T_end (kimg).
        rho: float = 7.0,            # Karras exponent.
        sigma_min: float = 2e-3,
        sigma_max: float = 80.0,
        loss_type: str = "pseudo_huber",  # "huber" | "l2" | "l2_root" | "pseudo_huber"
        weight_mode: str = "sqrt_karras",  # "edm" | "vlike" | "flat" | "snr" | "snr+1" | "karras" | "sqrt_karras" | "truncated-snr" | "uniform"
        sigma_data: float = 0.5,
        sampling_mode: str = "vp",   # "uniform" | "vp" | "edm"
        sync_dropout: bool = True,
        terminal_anchor: bool = True,
        terminal_teacher_hop: bool = False,
        student_sigma_mids: tuple = None,  # Interior student sigmas, e.g. (1.1,) for [sigma_max, 1.1, 0].
    ):
        assert S >= 2, "Student steps S must be >= 2"
        assert T_start >= 2 and T_end >= T_start
        assert loss_type in ("huber", "l2", "l2_root", "pseudo_huber")
        assert weight_mode in (
            "edm", "vlike", "flat",
            "snr", "snr+1", "karras", "sqrt_karras", "truncated-snr", "uniform",
        )
        self.teacher_net = teacher_net.eval().requires_grad_(False) if teacher_net is not None else None
        self._teacher_pkl_path = teacher_pkl_path
        self.S = int(S)
        self.T_start = int(T_start)
        self.T_end = int(T_end)
        self.T_anneal_kimg = float(T_anneal_kimg)
        self.rho = float(rho)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.loss_type = loss_type
        self.weight_mode = weight_mode
        self.sigma_data = float(sigma_data)
        assert sampling_mode in ("uniform", "vp", "edm"), f"Invalid sampling_mode: {sampling_mode}"
        self.sampling_mode = sampling_mode
        self.terminal_anchor = bool(terminal_anchor)
        self.terminal_teacher_hop = bool(terminal_teacher_hop)
        self.sync_dropout = bool(sync_dropout)
        self.student_sigma_mids = tuple(student_sigma_mids) if student_sigma_mids else None
        if self.student_sigma_mids is not None:
            assert len(self.student_sigma_mids) == S - 1, (
                f"student_sigma_mids has {len(self.student_sigma_mids)} values but S={S} requires {S-1} interior knots"
            )
            for i in range(len(self.student_sigma_mids)):
                assert self.student_sigma_mids[i] > 0, f"student_sigma_mids[{i}] must be > 0"
            for i in range(len(self.student_sigma_mids) - 1):
                assert self.student_sigma_mids[i] > self.student_sigma_mids[i + 1], (
                    f"student_sigma_mids must be strictly descending: {self.student_sigma_mids}"
                )
            assert self.student_sigma_mids[0] < self.sigma_max, (
                f"student_sigma_mids[0]={self.student_sigma_mids[0]} must be < sigma_max={self.sigma_max}"
            )

        self._global_kimg = 0.0
        self._filter_cache = {}

    def __getstate__(self):
        # Keep the frozen teacher out of checkpoints.
        state = self.__dict__.copy()
        state['teacher_net'] = None
        state['_filter_cache'] = {}
        return state

    def reload_teacher(self, device):
        with dnnlib.util.open_url(self._teacher_pkl_path) as f:
            data = pickle.load(f)
        self.teacher_net = data['ema'].eval().requires_grad_(False).to(device)

    def set_global_kimg(self, kimg: float) -> None:
        self._global_kimg = float(kimg)

    def _current_T_edges(self) -> int:
        if self.T_anneal_kimg <= 0:
            return self.T_end
        ratio = min(max(self._global_kimg / self.T_anneal_kimg, 0.0), 1.0)
        log_T_start = math.log(self.T_start)
        log_T_end = math.log(self.T_end)
        log_T_now = log_T_start + ratio * (log_T_end - log_T_start)
        T_now = int(round(math.exp(log_T_now)))
        T_now = max(self.T_start, min(self.T_end, T_now))
        return T_now

    def _build_student_grid(self, device: torch.device) -> torch.Tensor:
        if self.student_sigma_mids is not None:
            vals = [self.sigma_max] + list(self.student_sigma_mids) + [0.0]
            sigmas = torch.tensor(vals, dtype=torch.float64, device=device)
            return sigmas
        sigmas_prepad = make_karras_sigmas(
            num_nodes=self.S,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
        ).to(device)
        zero = torch.zeros(1, device=device, dtype=sigmas_prepad.dtype)
        sigmas = torch.cat([sigmas_prepad, zero], dim=0)
        return sigmas

    def _build_teacher_grid(self, student_sigmas: torch.Tensor, device: torch.device):
        target_T = self._current_T_edges()

        if target_T in self._filter_cache:
            cached_sigmas, cached_terminal_k = self._filter_cache[target_T]
            return cached_sigmas.to(device), cached_terminal_k

        raw_T = target_T
        while True:
            sigmas_prepad = make_karras_sigmas(
                num_nodes=raw_T,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                rho=self.rho,
            ).to(device)
            zero = torch.zeros(1, device=device, dtype=sigmas_prepad.dtype)
            teacher_full = torch.cat([sigmas_prepad, zero], dim=0)

            teacher_filtered, terminal_k = filter_teacher_edges_by_sigma(
                student_sigmas=student_sigmas,
                teacher_sigmas=teacher_full,
            )
            T_eff = teacher_filtered.shape[0] - 1
            if T_eff >= target_T:
                break
            raw_T += 1

        self._filter_cache[target_T] = (teacher_filtered.clone(), terminal_k)
        return teacher_filtered.to(device), terminal_k

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

        y = images

        student_sigmas = self._build_student_grid(device=device)

        teacher_sigmas, terminal_k = self._build_teacher_grid(
            student_sigmas=student_sigmas, device=device,
        )

        sigma_bounds = partition_edges_by_sigma(
            student_sigmas=student_sigmas,
            teacher_sigmas=teacher_sigmas,
        )

        sample_dict = sample_segment_and_teacher_pair(
            sigma_bounds=sigma_bounds,
            teacher_sigmas=teacher_sigmas,
            student_sigmas=student_sigmas,
            batch_size=batch_size,
            device=device,
            terminal_k=terminal_k,
            sampling_mode=self.sampling_mode,
            rho=self.rho,
            terminal_anchor=self.terminal_anchor,
        )

        sigma_t_vec = sample_dict["sigma_t"]
        sigma_s_teacher_vec = sample_dict["sigma_s"]
        sigma_bdry_vec = sample_dict["sigma_bdry"]
        is_terminal = sample_dict["is_terminal"].bool()
        is_boundary_snap = sample_dict["is_boundary_snap"].bool()
        is_general = (~is_terminal) & (~is_boundary_snap)

        sigma_s_eff = sigma_s_teacher_vec.clone()
        sigma_s_eff = torch.where(is_boundary_snap, sigma_bdry_vec, sigma_s_eff)
        sigma_s_eff = torch.where(
            is_general,
            torch.maximum(sigma_s_teacher_vec, sigma_bdry_vec),
            sigma_s_eff,
        )

        sigma_t = sigma_t_vec.to(torch.float64).view(batch_size, 1, 1, 1)
        sigma_s = sigma_s_eff.to(torch.float64).view(batch_size, 1, 1, 1)
        sigma_bdry = sigma_bdry_vec.to(torch.float64).view(batch_size, 1, 1, 1)

        eps = torch.randn_like(y).to(torch.float64)
        y64 = y.to(torch.float64)
        x_t = y64 + sigma_t * eps

        non_terminal = ~is_terminal

        x_s_teach = torch.zeros_like(x_t)
        x_ref_bdry = torch.zeros_like(x_t)
        sigma_ref_vec = sigma_t_vec.new_zeros(batch_size).to(torch.float64)
        tol = 1e-12

        if non_terminal.any():
            idx = non_terminal
            with torch.no_grad():
                x_s_teach_nt = heun_hop_edm(
                    net=self.teacher_net,
                    x_t=x_t[idx],
                    sigma_t=sigma_t_vec[idx],
                    sigma_s=sigma_s_eff[idx],
                    class_labels=labels[idx] if labels is not None else None,
                )
            x_s_teach[idx] = x_s_teach_nt

        if self.terminal_teacher_hop and is_terminal.any():
            with torch.no_grad():
                x_ref_bdry[is_terminal] = self.teacher_net(
                    x_t[is_terminal].float(),
                    sigma_t_vec[is_terminal],
                    labels[is_terminal] if labels is not None else None,
                ).to(torch.float64)
        else:
            x_ref_bdry[is_terminal] = y64[is_terminal]
        sigma_ref_vec[is_terminal] = 0.0

        x_ref_bdry[is_boundary_snap] = x_s_teach[is_boundary_snap]
        sigma_ref_vec[is_boundary_snap] = sigma_s_eff[is_boundary_snap]

        # Replay the student's dropout RNG on the target pass.
        if self.sync_dropout:
            rng_state = torch.cuda.get_rng_state()

        x_hat_t = net(x_t.float(), sigma_t, labels).to(torch.float32)

        if is_general.any():
            with torch.no_grad():
                # Suspend in-place weight renormalization so this no-grad pass cannot mutate MPConv weights.
                token = inplace_norm_flag.set(False)
                try:
                    if self.sync_dropout:
                        torch.cuda.set_rng_state(rng_state)
                        target_x = x_t.clone()
                        target_x[is_general] = x_s_teach[is_general]
                        target_sigma = sigma_t.clone()
                        target_sigma[is_general] = sigma_s[is_general]
                        x_hat_full = net(
                            target_x.float(), target_sigma, labels,
                        ).to(torch.float64)
                        x_hat_s_ng = x_hat_full[is_general]
                    else:
                        net.eval()
                        x_hat_s_ng = net(
                            x_s_teach[is_general].float(),
                            sigma_s[is_general],
                            labels[is_general] if labels is not None else None,
                        ).to(torch.float64)
                        net.train()
                finally:
                    inplace_norm_flag.reset(token)

            ratio_s_b = sigma_bdry[is_general] / torch.clamp(sigma_s[is_general], min=tol)
            x_ref_bdry[is_general] = x_hat_s_ng + ratio_s_b * (x_s_teach[is_general] - x_hat_s_ng)
            sigma_ref_vec[is_general] = sigma_bdry_vec[is_general]

        x_hat_t_star = inv_ddim_edm(
            x_ref=x_ref_bdry,
            x_t=x_t,
            sigma_t=sigma_t_vec,
            sigma_ref=sigma_ref_vec,
        ).to(torch.float32)

        weight = self._weight(sigma_t_vec.to(torch.float32))
        weight = weight.view(batch_size, 1, 1, 1)
        diff = x_hat_t - x_hat_t_star
        if self.loss_type == "huber":
            per_elem = _huber_loss(diff)
            loss = weight * per_elem
        elif self.loss_type == "pseudo_huber":
            per_sample = _pseudo_huber_vector_norm(diff, eps=1e-4)
            per_elem = per_sample.view(batch_size, 1, 1, 1)
            loss = weight * per_elem
        elif self.loss_type == "l2_root":
            per_sample = torch.sqrt(torch.clamp((diff * diff).sum(dim=[1, 2, 3]), min=1e-12))
            per_elem = per_sample.view(batch_size, 1, 1, 1)
            loss = weight * per_elem
        else:
            per_elem = diff * diff
            loss = weight * per_elem

        return loss
