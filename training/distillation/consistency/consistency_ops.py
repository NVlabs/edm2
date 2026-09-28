# Copyright (c) 2026, Vinay Saji Mathew.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Sigma grids, edge sampling, Heun teacher hops and inverse DDIM for MSCD.
ODE arithmetic is done in float64, network evaluations in float32."""

from typing import Dict, Tuple

import torch


@torch.no_grad()
def make_karras_sigmas(
    num_nodes: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
) -> torch.Tensor:
    """Descending Karras grid [sigma_0 > ... > sigma_{num_nodes-1}], float64, without the trailing 0."""
    assert num_nodes >= 1, "num_nodes must be >= 1"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    step_indices = torch.arange(num_nodes, dtype=torch.float64, device=device)
    sigma_min_r = float(sigma_min)
    sigma_max_r = float(sigma_max)
    rho_r = float(rho)
    sigmas = (sigma_max_r ** (1.0 / rho_r) + step_indices / max(num_nodes - 1, 1) * (sigma_min_r ** (1.0 / rho_r) - sigma_max_r ** (1.0 / rho_r))) ** rho_r
    return sigmas


def filter_teacher_edges_by_sigma(
    student_sigmas: torch.Tensor,
    teacher_sigmas: torch.Tensor,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, int]:
    """Drop teacher edges whose upper sigma coincides with an interior student sigma.
    The first and terminal edges are always kept.

    Returns (teacher_sigmas_cd, terminal_k_cd): the filtered grid (descending, ending
    in 0) and the index of its terminal edge.
    """
    assert student_sigmas.ndim == 1 and teacher_sigmas.ndim == 1
    T = len(teacher_sigmas) - 1
    assert T >= 1

    terminal_k_full = None
    for k in range(T - 1, -1, -1):
        if teacher_sigmas[k] > 0 and teacher_sigmas[k + 1] == 0:
            terminal_k_full = k
            break
    if terminal_k_full is None:
        terminal_k_full = T - 1

    student_interior = student_sigmas[1:-1]

    kept = []
    for k in range(T):
        sigma_k = teacher_sigmas[k]
        if k == 0 or k == terminal_k_full:
            kept.append(k)
            continue
        match = False
        for s in student_interior:
            if torch.isclose(
                sigma_k,
                s,
                rtol=eps,
                atol=eps * max(1.0, float(abs(sigma_k)), float(abs(s))),
            ):
                match = True
                break
        if match:
            continue
        kept.append(k)

    kept_idx = torch.tensor(kept, dtype=torch.long, device=teacher_sigmas.device)

    teacher_sigmas_cd = torch.cat(
        [teacher_sigmas[kept_idx], teacher_sigmas[-1:].clone()], dim=0
    )

    terminal_k_cd = len(kept) - 1

    return teacher_sigmas_cd, int(terminal_k_cd)


def partition_edges_by_sigma(student_sigmas: torch.Tensor, teacher_sigmas: torch.Tensor) -> torch.Tensor:
    """Segment j gets the teacher edges whose upper sigma lies in (sigma_s[j+1], sigma_s[j]].
    Both grids are descending and end in 0. Returns an (S, 2) LongTensor of inclusive
    [k_start, k_end] per segment.
    """
    assert student_sigmas.ndim == 1 and teacher_sigmas.ndim == 1
    S = len(student_sigmas) - 1
    T = len(teacher_sigmas) - 1
    bounds = []
    if not torch.all(student_sigmas[:-1] >= student_sigmas[1:]):
        raise AssertionError("student_sigmas must be descending")
    if not torch.all(teacher_sigmas[:-1] >= teacher_sigmas[1:]):
        raise AssertionError("teacher_sigmas must be descending")
    for j in range(S):
        upper = student_sigmas[j]
        lower = student_sigmas[j + 1]
        mask = (teacher_sigmas[:-1] <= upper) & (teacher_sigmas[:-1] > lower)
        idx = mask.nonzero(as_tuple=False).view(-1)
        if len(idx) == 0:
            diffs = (teacher_sigmas[:-1] - lower).abs()
            k_near = int(torch.argmin(diffs).item())
            bounds.append((k_near, k_near))
        else:
            k_start = int(idx.min().item())
            k_end = int(idx.max().item())
            bounds.append((k_start, k_end))
    return torch.tensor(bounds, dtype=torch.long, device=student_sigmas.device)


def compute_importance_weights(
    teacher_sigmas: torch.Tensor,
    rho: float,
    mode: str = "vp",
    P_mean: float = -1.2,
    P_std: float = 1.2,
    terminal_anchor: bool = True,
) -> torch.Tensor:
    """Normalized sampling probabilities over the T teacher edges.
    mode is "uniform", "vp" or "edm" (log-normal with P_mean, P_std).
    """
    sigmas = teacher_sigmas[:-1].float()
    T = len(sigmas)

    if mode == "uniform":
        weights = torch.ones(T, device=sigmas.device, dtype=torch.float32)
    elif mode == "vp":
        exponent = 1.0 - 1.0 / rho
        weights = (sigmas + 1e-10) ** exponent / (1.0 + sigmas ** 2)
    elif mode == "edm":
        log_sigmas = torch.log(sigmas + 1e-10)
        log_prob = -0.5 * ((log_sigmas - P_mean) / P_std) ** 2
        weights = (sigmas + 1e-10) ** (-1.0 / rho) * torch.exp(log_prob)
    else:
        raise ValueError(f"Unknown sampling mode: {mode}. Use 'uniform', 'vp', or 'edm'.")

    weights = weights / weights.sum().clamp(min=1e-10)

    # Terminal-edge anchoring: the edge sigma_min -> 0 always gets probability 1/T.
    if terminal_anchor and T > 1 and mode != "uniform":
        target_p = 1.0 / T
        non_term = weights[:-1]
        non_term_sum = non_term.sum().clamp(min=1e-10)
        weights[:-1] = non_term * (1.0 - target_p) / non_term_sum
        weights[-1] = target_p

    return weights


def sample_segment_and_teacher_pair(
    sigma_bounds: torch.Tensor,
    teacher_sigmas: torch.Tensor,
    student_sigmas: torch.Tensor,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator = None,
    terminal_k: int = None,
    sampling_mode: str = "vp",
    rho: float = 7.0,
    P_mean: float = -1.2,
    P_std: float = 1.2,
    terminal_anchor: bool = True,
) -> Dict[str, torch.Tensor]:
    """Sample (j, k_t, k_s) for consistency distillation using sigma-anchored segments."""
    S = sigma_bounds.shape[0]
    T = len(teacher_sigmas) - 1
    sigma_bounds = sigma_bounds.to(device)
    teacher_sigmas = teacher_sigmas.to(device)

    use_importance = sampling_mode in ("vp", "edm")
    if use_importance:
        edge_weights = compute_importance_weights(
            teacher_sigmas, rho, mode=sampling_mode, P_mean=P_mean, P_std=P_std,
            terminal_anchor=terminal_anchor,
        ).to(device)
    else:
        edge_weights = None

    if use_importance:
        k_t = torch.multinomial(edge_weights, batch_size, replacement=True, generator=generator)

        k_starts = sigma_bounds[:, 0].contiguous()
        k_ends = sigma_bounds[:, 1]

        step_j = torch.searchsorted(k_starts, k_t, right=True) - 1
        step_j = step_j.clamp(min=0, max=S-1)

        k0 = k_starts[step_j]
        k1 = k_ends[step_j]
        seg_len_j = (k1 - k0 + 1).clamp(min=1)

        local_idx = k_t - k0
        n_rel = seg_len_j - local_idx
        n_rel = n_rel.clamp(min=1)
    else:
        step_j = torch.randint(low=0, high=S, size=(batch_size,), device=device, dtype=torch.long, generator=generator)

        k0 = sigma_bounds[step_j, 0]
        k1 = sigma_bounds[step_j, 1]
        seg_len_j = (k1 - k0 + 1).clamp(min=1)

        u = torch.empty(batch_size, device=device, dtype=torch.float32)
        if generator is not None:
            u.uniform_(0.0, 1.0, generator=generator)
        else:
            u.uniform_(0.0, 1.0)
        n_rel = torch.floor(u * seg_len_j.float() + 1.0).to(torch.long)
        n_rel = torch.minimum(n_rel, seg_len_j)
        k_t = k1 - (n_rel - 1)

    k_s = (k_t + 1).clamp(max=T)

    sigma_t = teacher_sigmas[k_t]
    sigma_s = teacher_sigmas[k_s]
    sigma_bdry = student_sigmas[step_j + 1]

    if terminal_k is None:
        terminal_k = T - 1
    is_terminal = (k_t == terminal_k)
    is_boundary_snap = (~is_terminal) & (n_rel == 1) & (step_j < (S - 1))

    return {
        "step_j": step_j,
        "k_t": k_t,
        "k_s": k_s,
        "sigma_t": sigma_t,
        "sigma_s": sigma_s,
        "sigma_bdry": sigma_bdry,
        "n_rel": n_rel,
        "is_terminal": is_terminal,
        "is_boundary_snap": is_boundary_snap,
    }


def _expand_sigma_to_bchw(sigma: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    s = torch.as_tensor(sigma, device=like.device, dtype=like.dtype)
    if s.ndim == 0:
        s = s.reshape(1, 1, 1, 1)
    elif s.ndim == 1:
        s = s.reshape(-1, 1, 1, 1)
    return s


def inv_ddim_edm(
    x_ref: torch.Tensor,
    x_t: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_ref: torch.Tensor,
) -> torch.Tensor:
    """Inverse DDIM: x_hat*_t = (x_ref * sigma_t - x_t * sigma_ref) / (sigma_t - sigma_ref).
    Computed in float64 to avoid cancellation.
    """
    assert x_ref.shape == x_t.shape, "x_ref and x_t must have the same shape"
    out_dtype = x_t.dtype
    x_ref64 = x_ref.to(torch.float64)
    x_t64 = x_t.to(torch.float64)
    sigma_t_b = _expand_sigma_to_bchw(sigma_t, x_t64)
    sigma_ref_b = _expand_sigma_to_bchw(sigma_ref, x_t64)
    if torch.any(sigma_t_b == 0):
        bad_idx = (sigma_t_b == 0).nonzero(as_tuple=False)[:, 0].unique()
        sigma_t_flat = sigma_t if sigma_t.ndim <= 1 else sigma_t_b[:, 0, 0, 0]
        bad_vals = [(int(i.item()), float(sigma_t_flat[i].item())) for i in bad_idx[:5]]
        raise ValueError(
            f"inv_ddim_edm received sigma_t == 0. Avoid sigma=0 when backsolving.\n"
            f"  Affected samples (first 5): {bad_vals}"
        )
    denom = sigma_t_b - sigma_ref_b
    if torch.any(denom.abs() < 1e-12):
        bad_idx = (denom.abs() < 1e-12).nonzero(as_tuple=False)[:, 0].unique()
        sigma_t_flat = sigma_t if sigma_t.ndim <= 1 else sigma_t_b[:, 0, 0, 0]
        sigma_ref_flat = sigma_ref if sigma_ref.ndim <= 1 else sigma_ref_b[:, 0, 0, 0]
        bad_vals = [
            (int(i.item()), float(sigma_t_flat[i].item()), float(sigma_ref_flat[i].item()))
            for i in bad_idx[:5]
        ]
        raise ValueError(
            f"inv_ddim_edm denominator is zero (sigma_ref ~= sigma_t). Drop or resample this pair.\n"
            f"  Affected samples (first 5, format: [idx, sigma_t, sigma_ref]):\n"
            f"    {bad_vals}"
        )
    x_hat_star_t = (x_ref64 * sigma_t_b - x_t64 * sigma_ref_b) / denom
    return x_hat_star_t.to(out_dtype)


@torch.no_grad()
def heun_hop_edm(
    net,
    x_t: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_s: torch.Tensor,
    class_labels: torch.Tensor = None,
) -> torch.Tensor:
    """Deterministic Heun step of the teacher PF-ODE from sigma_t to sigma_s."""
    assert isinstance(x_t, torch.Tensor)
    out_dtype = x_t.dtype
    x64 = x_t.to(torch.float64)

    sigma_t_r = torch.as_tensor(sigma_t, device=x64.device)
    sigma_s_r = torch.as_tensor(sigma_s, device=x64.device)
    sigma_t_b = _expand_sigma_to_bchw(sigma_t_r, x64)
    sigma_s_b = _expand_sigma_to_bchw(sigma_s_r, x64)

    if torch.any(sigma_t_b == 0) or torch.any(sigma_s_b == 0):
        raise ValueError("heun_hop_edm received sigma=0. Avoid sigma=0 in teacher hops.")

    denoised_t = net(x64.float(), sigma_t_r, class_labels=class_labels).to(torch.float64)
    k1 = (x64 - denoised_t) / sigma_t_b

    x_eul = x64 + (sigma_s_b - sigma_t_b) * k1

    denoised_s = net(x_eul.float(), sigma_s_r, class_labels=class_labels).to(torch.float64)
    k2 = (x_eul - denoised_s) / sigma_s_b

    x_s = x64 + 0.5 * (sigma_s_b - sigma_t_b) * (k1 + k2)
    return x_s.to(out_dtype)
