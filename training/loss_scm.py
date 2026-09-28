# Copyright (c) 2026, Vinay Saji Mathew.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Continuous-time sCM loss (sCD / sCT) for TrigFlow, following Lu & Song (2025)."""

import pickle
import torch
import dnnlib
from torch_utils import persistence
from torch_utils import training_stats
from training.networks_trigflow import inplace_norm_flag


@persistence.persistent_class
class TrigFlowSCMLoss:
    """TrigFlow sCM loss with JVP tangent target (sCD / sCT)."""

    def __init__(
        self,
        *,
        mode='scd',
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
        logvar_per_sample=False,
    ):
        self.mode = str(mode).lower()
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
        # The loop sums the loss over pixels, so logvar is counted D times per
        # image; logvar/D keeps it per-sample (needed with RAdam on CIFAR-10).
        self.logvar_per_sample = bool(logvar_per_sample)
        self.cur_iter = 0

        if self.mode not in ('scd', 'sct'):
            raise ValueError(f'Unsupported sCM mode "{mode}"; expected "scd" or "sct".')
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

    def _sample_time(self, batch_size, device, dtype):
        tau = torch.randn([batch_size, 1, 1, 1], device=device, dtype=dtype) * self.P_std + self.P_mean
        t = torch.atan(tau.exp() / self.sigma_data)
        return t

    def _compute_dxt_dt(self, x0, z, t, labels):
        x_t = torch.cos(t) * x0 + torch.sin(t) * z
        if self.mode == 'sct':
            return torch.cos(t) * z - torch.sin(t) * x0
        assert self.teacher_net is not None
        with torch.no_grad():
            _, F_teacher = self.teacher_net(x_t, t.flatten(), labels, return_F=True)
        return self.sigma_data * F_teacher.to(torch.float32)

    def _jvp_finite_difference(self, net, x_t, t, labels, dxt_dt):
        eps = self.jvp_eps
        t_flat = t.flatten()
        v_x = torch.cos(t) * torch.sin(t) * dxt_dt
        v_t = (torch.cos(t) * torch.sin(t)).flatten()

        was_training = net.training
        token = inplace_norm_flag.set(False)
        fork_devices = [x_t.device] if x_t.device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=fork_devices):
                _, _, F_plus = net(
                    x_t + eps * v_x,
                    t_flat + eps * v_t,
                    labels,
                    return_logvar=True,
                    return_F=True,
                    force_fp32=True,
                )
            with torch.random.fork_rng(devices=fork_devices):
                _, _, F_minus = net(
                    x_t - eps * v_x,
                    t_flat - eps * v_t,
                    labels,
                    return_logvar=True,
                    return_F=True,
                    force_fp32=True,
                )
        finally:
            inplace_norm_flag.reset(token)
            net.train(was_training)
        return (F_plus.to(torch.float32) - F_minus.to(torch.float32)) / (2 * eps)

    def _jvp_flow(self, net, x_t, t, labels, dxt_dt):
        t_flat = t.flatten()
        v_x = torch.cos(t) * torch.sin(t) * dxt_dt
        v_t = (torch.cos(t) * torch.sin(t)).flatten()

        def model_wrapper(x_in, t_in):
            _, logvar, F_out = net(
                x_in,
                t_in,
                labels,
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

        t = self._sample_time(batch, device, torch.float32)
        z = torch.randn_like(x0) * self.sigma_data
        x_t = torch.cos(t) * x0 + torch.sin(t) * z

        dxt_dt = self._compute_dxt_dt(x0, z, t, labels)

        if self.use_jvp_finite_diff:
            _, logvar, F_theta = net_module(
                x_t, t.flatten(), labels, return_logvar=True, return_F=True, force_fp32=True
            )
            F_theta_jvp = self._jvp_finite_difference(net_module, x_t, t, labels, dxt_dt)
        else:
            F_theta, F_theta_jvp, logvar = self._jvp_flow(net_module, x_t, t, labels, dxt_dt)

        F_theta = F_theta.to(torch.float64)
        F_theta_jvp = F_theta_jvp.detach().to(torch.float64)
        F_theta_minus = F_theta.detach()
        logvar = logvar.to(torch.float64)
        t64 = t.to(torch.float64)
        x_t64 = x_t.to(torch.float64)
        dxt_dt64 = dxt_dt.to(torch.float64)

        r = min(1.0, float(self.cur_iter) / max(self.tangent_warmup_iters, 1))
        cos_t = torch.cos(t64)
        sin_t = torch.sin(t64)

        g = -cos_t * cos_t * (self.sigma_data * F_theta_minus - dxt_dt64)
        g = g - r * (cos_t * sin_t * x_t64 + self.sigma_data * F_theta_jvp)

        g_norm = torch.linalg.vector_norm(g, dim=(1, 2, 3), keepdim=True)
        g = g / (g_norm + self.tangent_c)

        resid = F_theta - F_theta_minus - g
        D = float(x0[0].numel())
        logvar_term = logvar / D if self.logvar_per_sample else logvar
        loss = resid.square() / logvar.exp() / D + logvar_term

        with torch.no_grad():
            training_stats.report('Loss/scm_loss', loss.mean())
            training_stats.report('Loss/scm_resid_rms', resid.reshape(batch, -1).square().mean(dim=1).sqrt().mean())
            training_stats.report('Loss/logvar', logvar.mean())
            training_stats.report('Loss/tangent_warmup_r', torch.tensor(r, device=device))
        return loss
