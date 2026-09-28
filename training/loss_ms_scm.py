# Copyright (c) 2026, Vinay Saji Mathew.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Continuous-time multistep sCD loss for TrigFlow (segment-conditioned sCM)."""

import pickle
import torch
import dnnlib
from torch_utils import persistence
from torch_utils import training_stats
from training.networks_trigflow import inplace_norm_flag
from training.segment_schedule import (
    build_boundaries,
    lookup_segment,
    sample_time_option_a,
    sample_time_option_b,
    t_max_from_sigma,
)


@persistence.persistent_class
class TrigFlowMSSCDLoss:
    """TrigFlowSCMLoss generalized to M segments; reduces to it at M=1.

    The JVP runs in train mode so the target sees the same dropout; only the
    in-place weight renormalization is suspended.
    """

    def __init__(
        self,
        *,
        mode='scd',
        num_segments=8,
        boundary_schedule='uniform_t',
        proposal_mode='global',
        sigma_max=80.0,
        P_mean=-1.0,
        P_std=1.6,
        sigma_data=0.5,
        tangent_c=0.1,
        tangent_warmup_iters=10000,
        teacher_net=None,
        teacher_pkl_path=None,
        use_jvp_finite_diff=False,
        jvp_eps=1e-3,
        sync_dropout=True,
    ):
        self.mode = str(mode).lower()
        self.num_segments = int(num_segments)
        self.boundary_schedule = str(boundary_schedule)
        self.proposal_mode = str(proposal_mode).lower()
        self.sigma_max = float(sigma_max)
        self.P_mean = float(P_mean)
        self.P_std = float(P_std)
        self.sigma_data = float(sigma_data)
        self.tangent_c = float(tangent_c)
        self.tangent_warmup_iters = int(tangent_warmup_iters)
        self.teacher_net = teacher_net
        self.teacher_pkl_path = teacher_pkl_path
        self.use_jvp_finite_diff = bool(use_jvp_finite_diff)
        self.jvp_eps = float(jvp_eps)
        self.sync_dropout = bool(sync_dropout)
        self.cur_iter = 0
        self.t_max = t_max_from_sigma(self.sigma_max, self.sigma_data)

        if self.mode not in ('scd', 'sct'):
            raise ValueError(f'Unsupported MS-sCD mode "{mode}"; expected "scd" or "sct".')
        if self.num_segments < 1:
            raise ValueError(f'num_segments must be >= 1, got {self.num_segments}')
        if self.proposal_mode not in ('global', 'segment_balanced'):
            raise ValueError(f'Unknown proposal_mode "{proposal_mode}"')
        if self.mode == 'scd' and self.teacher_net is None and self.teacher_pkl_path is None:
            raise ValueError('sCD mode requires a frozen teacher network or teacher_pkl_path.')

    def __getstate__(self):
        state = dict(self.__dict__)
        state['teacher_net'] = None
        return state

    def reload_teacher(self, device):
        if self.teacher_pkl_path is None:
            raise RuntimeError('Cannot reload teacher: teacher_pkl_path is None.')
        with dnnlib.util.open_url(self.teacher_pkl_path, verbose=False) as f:
            data = pickle.load(f)
        self.teacher_net = data['ema'].eval().requires_grad_(False).to(device)
        del data

    def _boundaries(self, device, dtype):
        return build_boundaries(
            self.num_segments, self.t_max,
            schedule=self.boundary_schedule, device=device, dtype=dtype,
        )

    def _sample_time(self, batch_size, device, dtype, boundaries):
        if self.proposal_mode == 'segment_balanced':
            return sample_time_option_b(batch_size, boundaries, device, dtype)
        return sample_time_option_a(
            batch_size, device, dtype,
            P_mean=self.P_mean, P_std=self.P_std, sigma_data=self.sigma_data,
        )

    def _compute_dxt_dt(self, x0, z, t, labels):
        x_t = torch.cos(t) * x0 + torch.sin(t) * z
        if self.mode == 'sct':
            return torch.cos(t) * z - torch.sin(t) * x0
        assert self.teacher_net is not None
        with torch.no_grad():
            _, F_teacher = self.teacher_net(x_t, t.flatten(), labels, return_F=True)
        return self.sigma_data * F_teacher.to(torch.float32)

    def _jvp_finite_difference(self, net, x_t, t, t_k, labels, dxt_dt, delta_t):
        eps = self.jvp_eps
        t_flat = t.flatten()
        t_k_flat = t_k.flatten()
        cos_dt = torch.cos(delta_t)
        sin_dt = torch.sin(delta_t)
        v_x = cos_dt * sin_dt * dxt_dt
        v_t = (cos_dt * sin_dt).reshape(-1)

        was_training = net.training
        token = inplace_norm_flag.set(False)
        fork_devices = [x_t.device] if x_t.device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=fork_devices):
                _, _, F_plus = net(
                    x_t + eps * v_x,
                    t_flat + eps * v_t,
                    labels,
                    t_k=t_k_flat,
                    return_logvar=True,
                    return_F=True,
                    force_fp32=True,
                )
            with torch.random.fork_rng(devices=fork_devices):
                _, _, F_minus = net(
                    x_t - eps * v_x,
                    t_flat - eps * v_t,
                    labels,
                    t_k=t_k_flat,
                    return_logvar=True,
                    return_F=True,
                    force_fp32=True,
                )
        finally:
            inplace_norm_flag.reset(token)
            net.train(was_training)
        return (F_plus.to(torch.float32) - F_minus.to(torch.float32)) / (2 * eps)

    def _jvp_flow(self, net, x_t, t, t_k, labels, dxt_dt, delta_t):
        t_flat = t.flatten()
        t_k_flat = t_k.flatten()
        cos_dt = torch.cos(delta_t)
        sin_dt = torch.sin(delta_t)
        v_x = cos_dt * sin_dt * dxt_dt
        v_t = (cos_dt * sin_dt).reshape(-1)

        def model_wrapper(x_in, t_in):
            _, logvar, F_out = net(
                x_in,
                t_in,
                labels,
                t_k=t_k_flat,
                return_logvar=True,
                return_F=True,
            )
            return F_out, logvar

        # Single dual-number forward, so primal and tangent share dropout masks.
        was_training = net.training
        token = inplace_norm_flag.set(False)
        try:
            F_theta, F_theta_jvp, logvar = torch.func.jvp(
                model_wrapper,
                (x_t, t_flat),
                (v_x, v_t),
                has_aux=True,
            )
        finally:
            inplace_norm_flag.reset(token)
            net.train(was_training)
        return F_theta, F_theta_jvp, logvar

    def __call__(self, net, images, labels=None):
        net_module = net.module if hasattr(net, 'module') else net
        x0 = images
        batch = x0.shape[0]
        device = x0.device

        boundaries = self._boundaries(device, torch.float32)
        t = self._sample_time(batch, device, torch.float32, boundaries)
        z = torch.randn_like(x0) * self.sigma_data
        x_t = torch.cos(t) * x0 + torch.sin(t) * z

        seg_k, t_k, _t_k1, delta_t = lookup_segment(t, boundaries)

        dxt_dt = self._compute_dxt_dt(x0, z, t, labels)

        if self.use_jvp_finite_diff:
            _, logvar, F_theta = net_module(
                x_t, t.flatten(), labels, t_k=t_k, return_logvar=True, return_F=True, force_fp32=True
            )
            F_theta_jvp = self._jvp_finite_difference(net_module, x_t, t, t_k, labels, dxt_dt, delta_t)
        else:
            F_theta, F_theta_jvp, logvar = self._jvp_flow(net_module, x_t, t, t_k, labels, dxt_dt, delta_t)

        F_theta = F_theta.to(torch.float64)
        F_theta_jvp = F_theta_jvp.detach().to(torch.float64)
        F_theta_minus = F_theta.detach()
        logvar = logvar.to(torch.float64)
        x_t64 = x_t.to(torch.float64)
        dxt_dt64 = dxt_dt.to(torch.float64)
        delta_t64 = delta_t.to(torch.float64)

        r = min(1.0, float(self.cur_iter) / max(self.tangent_warmup_iters, 1))
        cos_dt = torch.cos(delta_t64)
        sin_dt = torch.sin(delta_t64)

        # Tangent target; cos(dt) sin(dt) is already folded into F_theta_jvp.
        g = -cos_dt * cos_dt * (self.sigma_data * F_theta_minus - dxt_dt64)
        g = g - r * (cos_dt * sin_dt * x_t64 + self.sigma_data * F_theta_jvp)

        g_norm = torch.linalg.vector_norm(g, dim=(1, 2, 3), keepdim=True)
        g = g / (g_norm + self.tangent_c)

        resid = F_theta - F_theta_minus - g
        D = float(x0[0].numel())
        loss = resid.square() / logvar.exp() / D + logvar

        with torch.no_grad():
            training_stats.report('Loss/scm_loss', loss.mean())
            training_stats.report('Loss/scm_resid_rms', resid.reshape(batch, -1).square().mean(dim=1).sqrt().mean())
            training_stats.report('Loss/logvar', logvar.mean())
            training_stats.report('Loss/tangent_warmup_r', torch.tensor(r, device=device))
            training_stats.report('Loss/segment_k_mean', seg_k.to(torch.float32).mean())
            training_stats.report('Loss/delta_t_mean', delta_t.mean())
            # Report every segment (0 if empty) so all ranks reduce the same stats.
            loss_per_sample = loss.reshape(batch, -1).mean(dim=1)
            seg_idx = seg_k.reshape(-1).to(torch.int64)
            seg_sums = torch.zeros(self.num_segments, device=device, dtype=loss_per_sample.dtype)
            seg_sums.index_add_(0, seg_idx, loss_per_sample)
            seg_counts = torch.bincount(seg_idx, minlength=self.num_segments).to(loss_per_sample.dtype)
            seg_means = seg_sums / seg_counts.clamp(min=1)
            for k_idx in range(self.num_segments):
                training_stats.report(f'Loss/segment_{k_idx}_loss', seg_means[k_idx])
        return loss
