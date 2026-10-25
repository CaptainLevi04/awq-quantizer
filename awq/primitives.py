"""
AWQ quantization primitives.

Provides scale/zero-point computation, quantize-dequantize helpers, and
per-group observers (min-max and MSE-shrink).
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Scale / zero-point helpers
# ---------------------------------------------------------------------------

def calc_qparams(mn, mx, q_min: int = -8, q_max: int = 7):
    """Compute per-group asymmetric scale and int8 zero-point.

    Args:
        mn: Minimum values tensor (shape: [rows, groups]).
        mx: Maximum values tensor (shape: [rows, groups]).
        q_min: Minimum quantized value (inclusive).
        q_max: Maximum quantized value (inclusive).

    Returns:
        Tuple of (scale, zero_point) tensors; zero_point is int8.
    """
    mn = torch.min(mn, torch.zeros_like(mn))
    mx = torch.max(mx, torch.zeros_like(mx))
    bit_range = torch.tensor(q_max - q_min, dtype=mx.dtype, device=mx.device)
    scale = (mx - mn) / bit_range
    zp = q_min - (mn / scale)
    zp = torch.clamp(zp, q_min, q_max)
    eps = torch.finfo(scale.dtype).eps
    scale = torch.where(
        scale == 0,
        torch.tensor(eps, dtype=scale.dtype, device=scale.device),
        scale,
    )
    zp = torch.round(torch.clamp(zp, -128, 127)).to(torch.int8)
    return scale, zp


# ---------------------------------------------------------------------------
# Quantize-dequantize
# ---------------------------------------------------------------------------

def _qdq_grouped(x, scale, zp, q_min: int = -8, q_max: int = 7):
    """Quantize-dequantize a grouped tensor in floating point.

    Args:
        x: Input tensor unflattened to (..., num_groups, group_size).
        scale: Scale per group (broadcastable to x).
        zp: Zero-point per group (broadcastable to x, int8 stored as float for math).
        q_min: Clamp lower bound.
        q_max: Clamp upper bound.

    Returns:
        Fake-quantized tensor with the same shape and dtype as x.
    """
    scaled = x / scale
    scaled = scaled + zp.to(x.dtype)
    q = torch.round(torch.clamp(scaled, q_min, q_max)).to(x.dtype)
    return (q - zp.to(scale.dtype)) * scale


def fake_quant(w, scale, zp, group_size: int = 128, q_min: int = -8, q_max: int = 7):
    """Apply grouped fake-quantization to a weight matrix.

    Args:
        w: Weight tensor of shape [out_features, in_features].
        scale: Scale tensor of shape [out_features, num_groups].
        zp: Zero-point tensor of shape [out_features, num_groups].
        group_size: Number of elements per quantization group.
        q_min: Minimum quantized integer value.
        q_max: Maximum quantized integer value.

    Returns:
        Fake-quantized weight with the same dtype as w.
    """
    x = w.unflatten(-1, (-1, group_size))
    out = _qdq_grouped(x, scale.unsqueeze(-1), zp.unsqueeze(-1), q_min, q_max)
    return out.flatten(-2).to(w.dtype)


# ---------------------------------------------------------------------------
# Observers
# ---------------------------------------------------------------------------

@torch.no_grad()
def minmax_observer(w, group_size: int = 128):
    """Per-group min-max observer.

    Args:
        w: Weight tensor of shape [out_features, in_features].
        group_size: Number of elements per quantization group.

    Returns:
        Tuple of (min, max) tensors of shape [out_features, num_groups].
    """
    g = w.unflatten(-1, (-1, group_size))
    return g.amin(dim=-1), g.amax(dim=-1)


@torch.no_grad()
def mse_observer(
    w,
    group_size: int = 128,
    maxshrink: float = 0.20,
    patience: int = 5,
    grid: float = 100.0,
    norm: float = 2.4,
    q_min: int = -8,
    q_max: int = 7,
):
    """Per-group MSE-shrink observer.

    Iteratively shrinks the observed range to minimise quantization MSE.

    Args:
        w: Weight tensor of shape [out_features, in_features].
        group_size: Number of elements per quantization group.
        maxshrink: Maximum fractional shrinkage (0–1).
        patience: Stop early if no improvement for this many steps.
        grid: Number of grid steps for the shrinkage sweep.
        norm: Lp-norm exponent for the error metric.
        q_min: Minimum quantized integer value.
        q_max: Maximum quantized integer value.

    Returns:
        Tuple of (best_min, best_max) tensors of shape [out_features, num_groups].
    """
    obs = w.unflatten(-1, (-1, group_size))
    mn, mx = obs.amin(dim=-1), obs.amax(dim=-1)
    best_err = torch.full_like(mn, torch.finfo(mn.dtype).max)
    best_mn, best_mx = mn.clone(), mx.clone()
    no_improve = 0
    for i in range(int(maxshrink * grid)):
        p = 1 - i / grid
        smn, smx = mn * p, mx * p
        sc, zp = calc_qparams(smn, smx, q_min, q_max)
        q = _qdq_grouped(obs, sc.unsqueeze(-1), zp.unsqueeze(-1), q_min, q_max).to(obs.dtype)
        err = torch.sum((q - obs).abs().pow(norm), dim=-1)
        improved = err < best_err
        if torch.any(improved):
            best_err[improved] = err[improved]
            best_mn[improved] = smn[improved]
            best_mx[improved] = smx[improved]
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    return best_mn, best_mx
