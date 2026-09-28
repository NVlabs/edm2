# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modifications copyright (c) 2026, Vinay Saji Mathew.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""TrigFlow DDPM++ (SongUNet) for CIFAR-10, with positional time embedding
and adaptive double normalization."""

import numpy as np
import torch
from torch_utils import persistence
from torch.nn.functional import silu


def weight_init(shape, mode, fan_in, fan_out):
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


def pnorm(x, dim=1, eps=1e-8):
    # PGGAN pixel norm (Karras et al., 2018), eps inside the sqrt, as cited by sCM.
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


@persistence.persistent_class
class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x + self.bias.to(x.dtype)
        return x


@persistence.persistent_class
class Conv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        bias=True,
        up=False,
        down=False,
        resample_filter=[1, 1],
        fused_resample=False,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels * kernel * kernel, fan_out=out_channels * kernel * kernel)
        self.weight = (
            torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight)
            if kernel
            else None
        )
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(
                x,
                f.mul(4).tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=max(f_pad - w_pad, 0),
            )
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad + f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(
                    x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad
                )
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x + b.reshape(1, -1, 1, 1)
        return x


@persistence.persistent_class
class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = x.contiguous()
        return torch.nn.functional.group_norm(
            x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps
        )


@persistence.persistent_class
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


@persistence.persistent_class
class UNetBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        up=False,
        down=False,
        attention=False,
        num_heads=None,
        channels_per_head=64,
        dropout=0,
        skip_scale=1,
        eps=1e-5,
        resample_filter=[1, 1],
        resample_proj=False,
        init=dict(),
        init_zero=dict(init_weight=0),
        init_attn=None,
        dout_resolutions=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.dout_resolutions = dout_resolutions

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init
        )
        self.affine = Linear(in_features=emb_channels, out_features=out_channels * 2, **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init
            )

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(
                in_channels=out_channels,
                out_channels=out_channels * 3,
                kernel=1,
                **(init_attn if init_attn is not None else init),
            )
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        scale, shift = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype).chunk(chunks=2, dim=1)
        x = silu(self.norm1(x) * pnorm(scale, dim=1) + pnorm(shift, dim=1))

        drop_p = self.dropout if (self.dout_resolutions is None or x.shape[-1] in self.dout_resolutions) else 0
        x = self.conv1(torch.nn.functional.dropout(x, p=drop_p, training=self.training))
        x = (x + (self.skip(orig) if self.skip is not None else orig)) * self.skip_scale

        if self.num_heads:
            y = self.qkv(self.norm2(x))
            y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3])
            q, k, v = y.unbind(3)
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            w = torch.einsum("nhcq,nhck->nhqk", q, k / np.sqrt(q.shape[2])).softmax(dim=3).to(x.dtype)
            a = torch.einsum("nhqk,nhck->nhcq", w, v)
            x = (self.proj(a.reshape(*x.shape)) + x) * self.skip_scale
        return x


@persistence.persistent_class
class TrigFlowDDPMPPUNet(torch.nn.Module):
    def __init__(
        self,
        img_resolution,
        in_channels,
        out_channels,
        label_dim=0,
        augment_dim=0,
        model_channels=128,
        channel_mult=[2, 2, 2],
        channel_mult_emb=4,
        num_blocks=4,
        attn_resolutions=[16],
        dropout=0.10,
        label_dropout=0,
        channels_per_head=64,
        resample_filter=[1, 1],
        dout_resolutions=None,
        segment_conditioning=False,
    ):
        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels,
            num_heads=1,
            channels_per_head=channels_per_head,
            dropout=dropout,
            skip_scale=np.sqrt(0.5),
            eps=1e-6,
            resample_filter=resample_filter,
            resample_proj=True,
            init=init,
            init_zero=init_zero,
            init_attn=init_attn,
            dout_resolutions=dout_resolutions,
        )

        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True)
        self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        # Zero-init embedding of the segment bottom t_k (MS-sCD only).
        self.segment_conditioning = bool(segment_conditioning)
        if self.segment_conditioning:
            self.map_t_k = Linear(in_features=noise_channels, out_features=noise_channels, bias=False, **init)
            torch.nn.init.zeros_(self.map_t_k.weight)
        else:
            self.map_t_k = None

        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f"{res}x{res}_conv"] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs
                )
        skips = [block.out_channels for block in self.enc.values()]

        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f"{res}x{res}_in1"] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = idx == num_blocks and res in attn_resolutions
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            if level == 0:
                self.dec[f"{res}x{res}_aux_norm"] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f"{res}x{res}_aux_conv"] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels, segment_labels=None, augment_labels=None):
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        if segment_labels is not None:
            if self.map_t_k is None:
                raise RuntimeError(
                    'segment_labels were provided but this network was built without '
                    'segment_conditioning=True (no map_t_k head).'
                )
            seg_emb = self.map_noise(segment_labels)
            seg_emb = seg_emb.reshape(seg_emb.shape[0], 2, -1).flip(1).reshape(*seg_emb.shape)
            emb = emb + self.map_t_k(seg_emb)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        skips = []
        for name, block in self.enc.items():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)

        aux = None
        tmp = None
        for name, block in self.dec.items():
            if "aux_norm" in name:
                tmp = block(x)
            elif "aux_conv" in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux


@persistence.persistent_class
class TrigFlowDDPMPPPrecond(torch.nn.Module):
    def __init__(
        self,
        img_resolution,
        img_channels,
        label_dim,
        use_fp16=True,
        sigma_data=0.5,
        logvar_channels=128,
        model_channels=128,
        channel_mult=[2, 2, 2],
        channel_mult_emb=4,
        num_blocks=4,
        attn_resolutions=[16],
        dropout=0.10,
        label_dropout=0,
        augment_dim=0,
        **unet_kwargs,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        self.unet = TrigFlowDDPMPPUNet(
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            augment_dim=augment_dim,
            model_channels=model_channels,
            channel_mult=channel_mult,
            channel_mult_emb=channel_mult_emb,
            num_blocks=num_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            label_dropout=label_dropout,
            **unet_kwargs,
        )
        self.logvar_pos = PositionalEmbedding(logvar_channels)
        self.logvar_linear = Linear(logvar_channels, 1)

    def forward(
        self,
        x,
        t,
        class_labels=None,
        t_k=None,
        augment_labels=None,
        force_fp32=False,
        return_logvar=False,
        return_F=False,
        **unet_kwargs,
    ):
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
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == "cuda") else torch.float32

        c_skip = torch.cos(delta_t)
        c_out = -torch.sin(delta_t) * self.sigma_data
        c_in = 1 / self.sigma_data

        x_in = (c_in * x).to(dtype)
        F_x = self.unet(x_in, t.flatten(), class_labels, segment_labels=segment_labels, augment_labels=augment_labels, **unet_kwargs)
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


def reset_logvar_linear(net):
    """Re-initialize the adaptive-weight head."""
    logvar_channels = int(net.logvar_pos.num_channels)
    device = net.logvar_linear.weight.device
    net.logvar_linear = Linear(logvar_channels, 1).to(device)


def init_segment_embedding_zero(net):
    """Zero map_t_k; no-op if absent."""
    unet = getattr(net, 'unet', net)
    emb = getattr(unet, 'map_t_k', None)
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
        for block in part.values():
            if isinstance(block, UNetBlock):
                block.dropout = float(dropout)
                block.dout_resolutions = active_resolutions
