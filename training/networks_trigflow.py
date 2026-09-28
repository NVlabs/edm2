# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modifications copyright (c) 2026, Vinay Saji Mathew.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""TrigFlow U-Net for ImageNet-64: EDM2 magnitude-preserving layers with
positional time embeddings and adaptive double normalization."""

import copy
from contextvars import ContextVar
import numpy as np
import torch
from torch_utils import persistence
from torch_utils import misc


@persistence.import_hook
def _upgrade_emb_t_k_dtype(meta):
    # Older snapshots applied emb_t_k directly; fp16 snapshots then fail on dtype.
    old = 'emb = emb + self.emb_t_k(self.emb_pos(segment_labels))'
    new = ('emb = emb + torch.nn.functional.linear('
           'self.emb_pos(segment_labels), self.emb_t_k.weight.float()).to(emb.dtype)')
    if isinstance(meta.module_src, str) and old in meta.module_src:
        meta.module_src = meta.module_src.replace(old, new)
    return meta

# False suspends MPConv's in-place forced weight normalization (target/JVP passes).
inplace_norm_flag = ContextVar("inplace_norm_flag", default=True)


def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


def resample(x, f=[1, 1], mode="keep"):
    if mode == "keep":
        return x
    f = np.float32(f)
    assert f.ndim == 1 and len(f) % 2 == 0
    pad = (len(f) - 1) // 2
    f = f / f.sum()
    f = np.outer(f, f)[np.newaxis, np.newaxis, :, :]
    f = misc.const_like(x, f)
    c = x.shape[1]
    if mode == "down":
        return torch.nn.functional.conv2d(x, f.tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))
    assert mode == "up"
    return torch.nn.functional.conv_transpose2d(
        x, (f * 4).tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,)
    )


def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596


def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t**2)


def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t**2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.cat([wa * a, wb * b], dim=dim)


@persistence.persistent_class
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000):
        super().__init__()
        half = num_channels // 2
        freqs = torch.arange(half, dtype=torch.float32)
        freqs = freqs / max(half - 1, 1)
        freqs = (1 / max_positions) ** freqs
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        y = x.to(torch.float32).ger(self.freqs.to(torch.float32))
        y = torch.cat([y.cos(), y.sin()], dim=1)
        return (y * np.sqrt(2)).to(x.dtype)


@persistence.persistent_class
class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                if inplace_norm_flag.get():
                    self.weight.copy_(normalize(w))
        w = normalize(w)
        w = w * (gain / np.sqrt(w[0].numel()))
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1] // 2,))


@persistence.persistent_class
class Block(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        flavor="enc",
        resample_mode="keep",
        resample_filter=[1, 1],
        attention=False,
        channels_per_head=64,
        dropout=0,
        res_balance=0.3,
        attn_balance=0.3,
        clip_act=256,
        dout_resolutions=None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_filter = resample_filter
        self.resample_mode = resample_mode
        self.num_heads = out_channels // channels_per_head if attention else 0
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.dout_resolutions = dout_resolutions
        self.conv_res0 = MPConv(out_channels if flavor == "enc" else in_channels, out_channels, kernel=[3, 3])
        self.emb_linear = MPConv(emb_channels, out_channels * 2, kernel=[])
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3, 3])
        self.conv_skip = MPConv(in_channels, out_channels, kernel=[1, 1]) if in_channels != out_channels else None
        self.attn_qkv = MPConv(out_channels, out_channels * 3, kernel=[1, 1]) if self.num_heads != 0 else None
        self.attn_proj = MPConv(out_channels, out_channels, kernel=[1, 1]) if self.num_heads != 0 else None

    def forward(self, x, emb):
        x = resample(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1)

        y = self.conv_res0(mp_silu(x))
        y = normalize(y, dim=1)
        s, b = self.emb_linear(emb).chunk(2, dim=1)
        scale = normalize(s, dim=1)
        shift = normalize(b, dim=1)
        y = y * scale.unsqueeze(2).unsqueeze(3).to(y.dtype) + shift.unsqueeze(2).unsqueeze(3).to(y.dtype)
        y = mp_silu(y)
        if self.training and self.dropout != 0:
            if self.dout_resolutions is None or y.shape[-1] in self.dout_resolutions:
                y = torch.nn.functional.dropout(y, p=self.dropout)
        y = self.conv_res1(y)

        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        if self.num_heads != 0:
            y = self.attn_qkv(x)
            y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3])
            q, k, v = normalize(y, dim=2).unbind(3)
            w = torch.einsum("nhcq,nhck->nhqk", q, k / np.sqrt(q.shape[2])).softmax(dim=3)
            y = torch.einsum("nhqk,nhck->nhcq", w, v)
            y = self.attn_proj(y.reshape(*x.shape))
            x = mp_sum(x, y, t=self.attn_balance)

        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x


@persistence.persistent_class
class TrigFlowUNet(torch.nn.Module):
    def __init__(
        self,
        img_resolution,
        img_channels,
        label_dim,
        model_channels=192,
        channel_mult=[1, 2, 3, 4],
        channel_mult_noise=None,
        channel_mult_emb=None,
        num_blocks=3,
        attn_resolutions=[16, 8],
        label_balance=0.5,
        concat_balance=0.5,
        segment_conditioning=False,
        **block_kwargs,
    ):
        super().__init__()
        cblock = [model_channels * x for x in channel_mult]
        cnoise = model_channels * channel_mult_noise if channel_mult_noise is not None else cblock[0]
        cemb = model_channels * channel_mult_emb if channel_mult_emb is not None else max(cblock)
        self.label_balance = label_balance
        self.concat_balance = concat_balance
        self.out_gain = torch.nn.Parameter(torch.zeros([]))

        self.emb_pos = PositionalEmbedding(cnoise)
        self.emb_noise = MPConv(cnoise, cemb, kernel=[])
        # Zero-init embedding of the segment bottom t_k (MS-sCD only).
        self.segment_conditioning = bool(segment_conditioning)
        if self.segment_conditioning:
            self.emb_t_k = torch.nn.Linear(cnoise, cemb, bias=False)
            torch.nn.init.zeros_(self.emb_t_k.weight)
        else:
            self.emb_t_k = None
        self.emb_label = MPConv(label_dim, cemb, kernel=[]) if label_dim != 0 else None

        self.enc = torch.nn.ModuleDict()
        cout = img_channels + 1
        for level, channels in enumerate(cblock):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f"{res}x{res}_conv"] = MPConv(cin, cout, kernel=[3, 3])
            else:
                self.enc[f"{res}x{res}_down"] = Block(
                    cout, cout, cemb, flavor="enc", resample_mode="down", **block_kwargs
                )
            for idx in range(num_blocks):
                cin = cout
                cout = channels
                self.enc[f"{res}x{res}_block{idx}"] = Block(
                    cin, cout, cemb, flavor="enc", attention=(res in attn_resolutions), **block_kwargs
                )

        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, channels in reversed(list(enumerate(cblock))):
            res = img_resolution >> level
            if level == len(cblock) - 1:
                self.dec[f"{res}x{res}_in0"] = Block(cout, cout, cemb, flavor="dec", attention=True, **block_kwargs)
                self.dec[f"{res}x{res}_in1"] = Block(cout, cout, cemb, flavor="dec", **block_kwargs)
            else:
                self.dec[f"{res}x{res}_up"] = Block(
                    cout, cout, cemb, flavor="dec", resample_mode="up", **block_kwargs
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f"{res}x{res}_block{idx}"] = Block(
                    cin, cout, cemb, flavor="dec", attention=(res in attn_resolutions), **block_kwargs
                )
        self.out_conv = MPConv(cout, img_channels, kernel=[3, 3])

    def forward(self, x, noise_labels, class_labels, segment_labels=None):
        emb = self.emb_noise(self.emb_pos(noise_labels))
        if segment_labels is not None:
            if self.emb_t_k is None:
                raise RuntimeError(
                    'segment_labels were provided but this network was built without '
                    'segment_conditioning=True (no emb_t_k head).'
                )
            # Plain Linear: run in fp32, since snapshots may store fp16 weights.
            seg_emb = self.emb_pos(segment_labels)
            emb = emb + torch.nn.functional.linear(seg_emb, self.emb_t_k.weight.float()).to(emb.dtype)
        if self.emb_label is not None:
            emb = mp_sum(emb, self.emb_label(class_labels * np.sqrt(class_labels.shape[1])), t=self.label_balance)
        emb = mp_silu(emb)

        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        for name, block in self.enc.items():
            x = block(x) if "conv" in name else block(x, emb)
            skips.append(x)

        for name, block in self.dec.items():
            if "block" in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb)
        x = self.out_conv(x, gain=self.out_gain)
        return x


@persistence.persistent_class
class TrigFlowPrecond(torch.nn.Module):
    def __init__(
        self,
        img_resolution,
        img_channels,
        label_dim,
        use_fp16=True,
        sigma_data=0.5,
        logvar_channels=128,
        **unet_kwargs,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        self.unet = TrigFlowUNet(
            img_resolution=img_resolution, img_channels=img_channels, label_dim=label_dim, **unet_kwargs
        )
        self.logvar_pos = PositionalEmbedding(logvar_channels)
        self.logvar_linear = MPConv(logvar_channels, 1, kernel=[])

    def forward(self, x, t, class_labels=None, t_k=None, force_fp32=False, return_logvar=False, return_F=False, **unet_kwargs):
        x = x.to(torch.float32)
        t = t.to(torch.float32).reshape(-1, 1, 1, 1)
        # f = cos(t - t_k) x_t - sin(t - t_k) sigma_d F; t_k=None is the single-segment model.
        if t_k is None:
            delta_t = t
            segment_labels = None
        else:
            t_k_t = t_k.to(torch.float32).reshape(-1, 1, 1, 1)
            delta_t = t - t_k_t
            segment_labels = t_k_t.flatten()
        class_labels = (
            None
            if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=x.device)
            if class_labels is None
            else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        )
        # bf16 rather than fp16: JVP tangents overflow fp16.
        dtype = torch.bfloat16 if (self.use_fp16 and not force_fp32 and x.device.type == "cuda") else torch.float32

        c_skip = torch.cos(delta_t)
        c_out = -torch.sin(delta_t) * self.sigma_data
        c_in = 1 / self.sigma_data

        x_in = (c_in * x).to(dtype)
        F_x = self.unet(x_in, t.flatten(), class_labels, segment_labels=segment_labels, **unet_kwargs)
        F_x32 = F_x.to(torch.float32)
        D_x = c_skip * x + c_out * F_x32

        if return_logvar:
            logvar = self.logvar_linear(self.logvar_pos(t.flatten())).reshape(-1, 1, 1, 1)
            if return_F:
                return D_x, logvar, F_x32
            return D_x, logvar
        if return_F:
            return D_x, F_x32
        return D_x


def transfer_edm2_weights(trigflow_net, edm2_net):
    """Copy compatible image-path weights from EDM2 Precond into TrigFlowPrecond."""
    src_sd = edm2_net.state_dict()
    dst_sd = trigflow_net.state_dict()

    excluded_prefixes = ("unet.emb_fourier.", "logvar_fourier.")
    excluded_exact = {"unet.emb_noise.weight", "logvar_linear.weight"}
    copyable_suffixes = (
        ".conv_res0.weight",
        ".conv_res1.weight",
        ".conv_skip.weight",
        ".attn_qkv.weight",
        ".attn_proj.weight",
    )
    copyable_exact = {
        "unet.enc.64x64_conv.weight",
        "unet.out_conv.weight",
        "unet.out_gain",
        "unet.emb_label.weight",
    }

    copied = []
    skipped = []
    for key, src_val in src_sd.items():
        if any(key.startswith(p) for p in excluded_prefixes) or key in excluded_exact:
            skipped.append((key, "excluded"))
            continue
        if key not in dst_sd:
            skipped.append((key, "missing_in_dst"))
            continue
        if dst_sd[key].shape != src_val.shape:
            skipped.append((key, f"shape_mismatch:{tuple(src_val.shape)}->{tuple(dst_sd[key].shape)}"))
            continue

        should_copy = key in copyable_exact or any(key.endswith(suf) for suf in copyable_suffixes)
        if not should_copy:
            skipped.append((key, "not_whitelisted"))
            continue

        dst_sd[key] = copy.deepcopy(src_val).to(dst_sd[key].dtype)
        copied.append(key)

    trigflow_net.load_state_dict(dst_sd, strict=False)
    return copied, skipped


def reset_logvar_linear(net):
    """Re-initialize the adaptive-weight head."""
    logvar_channels = int(net.logvar_pos.freqs.numel() * 2)
    device = net.logvar_linear.weight.device if hasattr(net.logvar_linear, 'weight') else next(net.parameters()).device
    net.logvar_linear = MPConv(logvar_channels, 1, kernel=[]).to(device)


def init_segment_embedding_zero(net):
    """Zero emb_t_k; no-op if absent."""
    unet = getattr(net, 'unet', net)
    emb = getattr(unet, 'emb_t_k', None)
    if emb is not None and hasattr(emb, 'weight'):
        with torch.no_grad():
            emb.weight.zero_()


def apply_resolution_dropout(net, dropout=0.45, max_resolution=16):
    """Apply dropout only at resolutions <= max_resolution."""
    unet = net.unet
    img_resolution = int(net.img_resolution)
    active_resolutions = []
    cur = img_resolution
    while cur >= 8:
        if cur <= max_resolution:
            active_resolutions.append(cur)
        cur //= 2
    active_resolutions = tuple(sorted(set(active_resolutions), reverse=True))
    for part_name in ('enc', 'dec'):
        part = getattr(unet, part_name, None)
        if part is None:
            continue
        for block_key, block in part.items():
            block.dropout = float(dropout)
            block.dout_resolutions = active_resolutions
