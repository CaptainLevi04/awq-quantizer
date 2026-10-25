"""
AWQ activation smoothing.

Implements the scale-search grid and the ``awq_smooth_mapping`` routine that
migrates outlier difficulty from activations to weights before quantization.
"""

import torch
import torch.nn.functional as F
from torch.utils._pytree import tree_map

from awq.primitives import calc_qparams, fake_quant, minmax_observer


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def grid_params(duo_scaling, n_grid: int):
    """Generate (ratio, use_duo) pairs for the AWQ scale-search grid.

    Args:
        duo_scaling: Scaling mode.  ``"both"`` interleaves activation-only and
            duo steps; ``False`` uses activation-only; any truthy value uses
            duo scaling exclusively (with one activation-only step prepended).
        n_grid: Total number of grid points.

    Returns:
        List of ``(ratio, use_duo)`` tuples.
    """
    if duo_scaling == "both":
        n = int(n_grid / 2)
        return [(i / (n - 1), d) for i in range(n) for d in (False, True)]
    if duo_scaling is False:
        return [(i / (n_grid - 1), False) for i in range(n_grid)]
    # duo_scaling is True (or any other truthy value): duo only + one baseline
    return [(0.0, False)] + [(i / (n_grid - 2), True) for i in range(n_grid - 1)]


# ---------------------------------------------------------------------------
# Activation / weight statistics
# ---------------------------------------------------------------------------

@torch.no_grad()
def layer_means(layers, group_size: int = 128):
    """Compute column-wise mean absolute weight magnitudes across layers.

    Weights are normalised per-group before averaging so that layers with
    different scales contribute equally.

    Args:
        layers: Iterable of ``nn.Linear`` modules.
        group_size: Quantization group size used for normalisation.

    Returns:
        1-D ``float64`` tensor of length ``in_features``.
    """
    total_sum, total_count = 0, 0
    for layer in layers:
        w = layer.weight.clone()
        shape = w.shape
        w = w.reshape(-1, group_size)
        w.abs_()
        w.div_(w.amax(dim=1, keepdim=True) + 1e-6)
        w = w.reshape(shape)
        total_count += w.size(0)
        total_sum += w.sum(0, dtype=torch.float64)
    return total_sum / total_count


# ---------------------------------------------------------------------------
# Argument cache
# ---------------------------------------------------------------------------

def to_device(obj, device):
    """Recursively move all tensors in a nested structure to *device*."""
    return tree_map(lambda t: t.to(device) if isinstance(t, torch.Tensor) else t, obj)


class ArgCache:
    """Stores forward-pass (args, kwargs) pairs on a target device."""

    def __init__(self, device):
        self.device = device
        self.items = []

    def add(self, args, kwargs):
        """Detach and cache one set of forward arguments."""
        detach = lambda t: t.detach().to(self.device) if isinstance(t, torch.Tensor) else t
        self.items.append(tree_map(detach, (args, kwargs)))

    def __len__(self):
        return len(self.items)


# ---------------------------------------------------------------------------
# Reference forward pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_parent(parent, cache: ArgCache, exec_device):
    """Run *parent* on all cached inputs and return output tensors.

    Args:
        parent: An ``nn.Module`` (attention or MLP sub-block).
        cache: ``ArgCache`` populated during the statistics collection pass.
        exec_device: Device on which to run the forward pass.

    Returns:
        List of output tensors (one per cached sample).
    """
    outs = []
    for args, kwargs in cache.items:
        args, kwargs = to_device((args, kwargs), exec_device)
        out = parent(*args, **kwargs)
        outs.append(out[0] if isinstance(out, tuple) else out)
    return outs


# ---------------------------------------------------------------------------
# MSE loss helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def mse_loss_between(fp_outs, q_outs):
    """Compute mean MSE loss between two lists of output tensors.

    Args:
        fp_outs: Reference (float) outputs.
        q_outs: Quantized outputs (moved to the same device as *fp_outs*).

    Returns:
        Scalar float MSE value.
    """
    loss = torch.tensor(0.0, device=fp_outs[0].device)
    n = torch.tensor(0, device=fp_outs[0].device)
    for a, b in zip(fp_outs, q_outs):
        loss += F.mse_loss(a, b.to(a.device), reduction="sum")
        n += a.numel()
    return (loss / n).item()


# ---------------------------------------------------------------------------
# AWQ smoothing
# ---------------------------------------------------------------------------

@torch.no_grad()
def awq_smooth_mapping(
    smooth_layer,
    balance_layers,
    parent,
    cache: ArgCache,
    x_sum,
    x_count,
    exec_device,
    group_size: int = 128,
    n_grid: int = 20,
    duo_scaling="both",
    q_min: int = -8,
    q_max: int = 7,
):
    """Search for the optimal AWQ smoothing scale for one mapping.

    Iterates over a grid of (ratio, use_duo) pairs, applies a candidate scale
    vector to *smooth_layer* and *balance_layers*, fake-quantizes the balanced
    weights, and selects the scale that minimises output MSE w.r.t. the
    floating-point reference.

    Args:
        smooth_layer: The LayerNorm (or Linear) whose output is divided by
            the scale (e.g., ``input_layernorm``).
        balance_layers: List of Linear layers whose weights are multiplied by
            the scale (e.g., ``[q_proj, k_proj, v_proj]``).
        parent: Parent module whose forward pass produces the outputs we
            compare (e.g., the full attention or MLP block).
        cache: Cached forward arguments for *parent*.
        x_sum: Sum of per-token absolute activations (shape: ``[in_features]``).
        x_count: Number of tokens accumulated in *x_sum*.
        exec_device: Device for running forward passes.
        group_size: Quantization group size.
        n_grid: Number of grid search points.
        duo_scaling: Grid mode (``"both"``, ``True``, or ``False``).
        q_min: Minimum quantized integer value.
        q_max: Maximum quantized integer value.

    Returns:
        Tuple ``(initial_mse, best_mse)`` — useful for logging improvement.
    """
    x_mean = x_sum.to(exec_device) / x_count.to(exec_device)
    w_mean = (
        layer_means(balance_layers, group_size).to(exec_device)
        if duo_scaling
        else None
    )
    orig_w = {bl: bl.weight.clone() for bl in balance_layers}
    fp_outs = run_parent(parent, cache, exec_device)
    assert all(o.isfinite().all() for o in fp_outs), "NaN/inf in reference outputs"

    best_err, best_scales, initial_err = float("inf"), None, None

    for ratio, use_duo in grid_params(duo_scaling, n_grid):
        if use_duo:
            scales = (
                x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)
            ).clamp(min=1e-4)
        else:
            scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)

        scales = scales / (scales.max() * scales.min()).sqrt()
        scales[torch.isinf(scales)] = 1
        scales[torch.isnan(scales)] = 1

        sv = scales.view(1, -1).to(exec_device)
        for bl in balance_layers:
            bl.weight.data.copy_(orig_w[bl].to(sv.device) * sv)
            mn, mx = minmax_observer(bl.weight.data, group_size)
            sc, zp = calc_qparams(mn, mx, q_min, q_max)
            bl.weight.data = (
                fake_quant(bl.weight.data, sc, zp, group_size, q_min, q_max) / sv
            ).to(bl.weight.dtype)

        loss = mse_loss_between(fp_outs, run_parent(parent, cache, exec_device))
        if initial_err is None:
            initial_err = loss
        if loss < best_err:
            best_err, best_scales = loss, scales.clone()

    if best_scales is None:
        raise RuntimeError(
            "No finite loss in grid search (NaNs in forward pass?)"
        )

    best_scales = best_scales.detach().cpu()

    # Apply best scales to balance layers
    for bl in balance_layers:
        bl.weight.data.copy_(
            orig_w[bl].to(bl.weight.device)
            * best_scales.to(bl.weight.device).view(1, -1)
        )

    # Absorb inverse scale into smooth_layer
    s = best_scales.to(smooth_layer.weight.device)
    if smooth_layer.weight.ndim == 1:
        smooth_layer.weight.data.copy_(smooth_layer.weight.data / s)
    else:
        w = smooth_layer.weight.data
        w[-s.size(0) :] = (w[-s.size(0) :] / s.view(-1, 1)).to(w.dtype)

    if getattr(smooth_layer, "bias", None) is not None:
        smooth_layer.bias.data.copy_(smooth_layer.bias.data / s)

    return initial_err, best_err
