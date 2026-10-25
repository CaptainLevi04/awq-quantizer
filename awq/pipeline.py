"""
Layer-by-layer AWQ quantization pipeline.

Captures calibration activations, runs the AWQ smoothing pass for each
transformer layer, and records per-layer quantization parameters.
"""

import gc

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

from awq.primitives import calc_qparams, mse_observer
from awq.smooth import ArgCache, awq_smooth_mapping, to_device


# ---------------------------------------------------------------------------
# Input capture helpers
# ---------------------------------------------------------------------------

class _Stop(Exception):
    """Raised by ``_Catcher`` to abort the forward pass after the first layer."""


class _Catcher(nn.Module):
    """Wraps the first decoder layer to intercept its inputs."""

    def __init__(self, layer, sink: list):
        super().__init__()
        self.layer = layer
        self.sink = sink
        self.attention_type = getattr(layer, "attention_type", None)

    def forward(self, *args, **kwargs):
        hidden = args[0] if args else kwargs.pop("hidden_states")
        self.sink.append(
            (
                hidden.detach(),
                tree_map(
                    lambda t: t.detach() if isinstance(t, torch.Tensor) else t,
                    kwargs,
                ),
            )
        )
        raise _Stop


@torch.no_grad()
def capture_first_layer_inputs(model, samples, exec_device, cache_device):
    """Run the embedding stack and capture the first decoder-layer inputs.

    Args:
        model: The full causal-LM model.
        samples: List of ``[1, seq_len]`` token-id tensors.
        exec_device: Device for the forward pass (typically ``"cuda"``).
        cache_device: Device where captured tensors are stored between layers.

    Returns:
        Tuple ``(hidden_states, layer_kwargs)`` — one entry per sample.
    """
    layers = model.model.layers
    sink: list = []
    hs: list = []
    lk: list = []

    layers[0] = _Catcher(layers[0], sink)
    try:
        for ids in samples:
            try:
                ids = ids.to(exec_device)
                model(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    use_cache=False,
                )
            except _Stop:
                pass
            h, kw = sink.pop()
            kw.pop("past_key_values", None)
            kw.pop("past_key_value", None)
            kw["use_cache"] = False
            hs.append(h.to(cache_device))
            lk.append(to_device(kw, cache_device))
    finally:
        layers[0] = layers[0].layer

    return hs, lk


def _call_layer(layer, h, kw, exec_device):
    """Forward one decoder layer and return its hidden-state output.

    Args:
        layer: A single ``nn.Module`` decoder layer.
        h: Hidden state tensor (on *cache_device*).
        kw: Keyword arguments dict (on *cache_device*).
        exec_device: Device on which to run the layer.

    Returns:
        Output hidden-state tensor (on *exec_device*).
    """
    out = layer(to_device(h, exec_device), **to_device(kw, exec_device))
    return out[0] if isinstance(out, tuple) else out


# ---------------------------------------------------------------------------
# Attention implementation context manager
# ---------------------------------------------------------------------------

class eager_attention:
    """Context manager that temporarily forces eager attention.

    Some models default to ``sdpa`` or ``flash_attention_2``, which can
    interfere with hook-based statistics collection.
    """

    def __init__(self, model):
        self.cfg = model.config

    def __enter__(self):
        self.old = self.cfg._attn_implementation
        self.cfg._attn_implementation = "eager"

    def __exit__(self, *exc):
        self.cfg._attn_implementation = self.old


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def quantize_model(
    model,
    samples,
    exec_device,
    group_size: int = 128,
    num_bits: int = 4,
    n_grid: int = 20,
    duo_scaling="both",
    cache_device=None,
    log=print,
):
    """Quantize all linear layers of *model* using AWQ.

    Args:
        model: The causal-LM model (on *exec_device*, in eval mode).
        samples: Calibration token tensors from ``load_calibration_samples``.
        exec_device: CUDA/CPU device for layer execution.
        group_size: Quantization group size.
        num_bits: Quantization bit-width (4 or 8).
        n_grid: Number of AWQ grid-search points.
        duo_scaling: AWQ scaling mode (``"both"``, ``True``, or ``False``).
        cache_device: Device for caching activations between layers.
            Defaults to *exec_device* when ``None``.
        log: Callable used to print per-layer progress (default: ``print``).

    Returns:
        Dict mapping ``"model.layers.<i>.<proj_name>"`` to
        ``(scale, zero_point)`` tensors for every quantized linear layer.
    """
    if cache_device is None:
        cache_device = exec_device

    with torch.no_grad(), eager_attention(model):
        return _quantize_model(
            model,
            samples,
            exec_device,
            cache_device=cache_device,
            group_size=group_size,
            num_bits=num_bits,
            n_grid=n_grid,
            duo_scaling=duo_scaling,
            log=log,
        )


def _quantize_model(
    model,
    samples,
    exec_device,
    cache_device,
    group_size: int = 128,
    num_bits: int = 4,
    n_grid: int = 20,
    duo_scaling="both",
    log=print,
):
    """Internal implementation of the layer-by-layer quantization loop.

    Do not call directly; use ``quantize_model`` instead.
    """
    q_min = -(2 ** (num_bits - 1))
    q_max = 2 ** (num_bits - 1) - 1

    layers = model.model.layers
    hs, lk = capture_first_layer_inputs(model, samples, exec_device, cache_device)
    qparams = {}

    for li, layer in enumerate(layers):
        attn = layer.self_attn
        mlp = layer.mlp
        q_proj, k_proj, v_proj, o_proj = (
            attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj
        )
        gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj

        # ------------------------------------------------------------------ #
        # Statistics collection pass
        # ------------------------------------------------------------------ #
        stats: dict = {}
        caches = {
            "attn": ArgCache(cache_device),
            "mlp": ArgCache(cache_device),
            "down": ArgCache(cache_device),
        }

        def stat_hook(name):
            def hook(_m, args, _out):
                a = args[0].abs().detach().flatten(0, -2)
                s = a.float().sum(dim=0).cpu()
                c = torch.tensor(a.size(0))
                if name not in stats:
                    stats[name] = [torch.zeros_like(s), torch.zeros_like(c)]
                stats[name][0] += s
                stats[name][1] += c
            return hook

        def cache_hook(name):
            def hook(_m, args, kwargs):
                caches[name].add(args, kwargs)
            return hook

        handles = [
            q_proj.register_forward_hook(stat_hook("attn")),
            gate.register_forward_hook(stat_hook("mlp")),
            down.register_forward_hook(stat_hook("down")),
            attn.register_forward_pre_hook(cache_hook("attn"), with_kwargs=True),
            mlp.register_forward_pre_hook(cache_hook("mlp"), with_kwargs=True),
            down.register_forward_pre_hook(cache_hook("down"), with_kwargs=True),
        ]

        for h, kw in zip(hs, lk):
            _call_layer(layer, h, kw, exec_device)

        for hd in handles:
            hd.remove()

        # ------------------------------------------------------------------ #
        # AWQ smoothing
        # ------------------------------------------------------------------ #
        mappings = [
            (
                "input_layernorm",
                layer.input_layernorm,
                [q_proj, k_proj, v_proj],
                attn,
                "attn",
            ),
            (
                "post_attention_layernorm",
                layer.post_attention_layernorm,
                [gate, up],
                mlp,
                "mlp",
            ),
            (
                "up_proj",
                up,
                [down],
                down,
                "down",
            ),
        ]

        for name, smooth_layer, balance, parent, key in mappings:
            e0, e1 = awq_smooth_mapping(
                smooth_layer,
                balance,
                parent,
                caches[key],
                stats[key][0],
                stats[key][1],
                exec_device,
                group_size=group_size,
                n_grid=n_grid,
                duo_scaling=duo_scaling,
                q_min=q_min,
                q_max=q_max,
            )
            log(f"  layer {li:2d} {name:25s} mse {e0:.3e} -> {e1:.3e}")
            del caches[key]

        del caches, stats

        # ------------------------------------------------------------------ #
        # Record quantization parameters for each linear projection
        # ------------------------------------------------------------------ #
        prefix = f"model.layers.{li}."
        for lname, lin in [
            ("self_attn.q_proj", q_proj),
            ("self_attn.k_proj", k_proj),
            ("self_attn.v_proj", v_proj),
            ("self_attn.o_proj", o_proj),
            ("mlp.gate_proj", gate),
            ("mlp.up_proj", up),
            ("mlp.down_proj", down),
        ]:
            mn, mx = mse_observer(lin.weight.data, group_size, q_min=q_min, q_max=q_max)
            qparams[prefix + lname] = calc_qparams(mn, mx, q_min, q_max)

        # ------------------------------------------------------------------ #
        # Propagate hidden states to the next layer
        # ------------------------------------------------------------------ #
        hs = [
            _call_layer(layer, h, kw, exec_device).to(cache_device)
            for h, kw in zip(hs, lk)
        ]

        gc.collect()
        if exec_device.type == "cuda":
            torch.cuda.empty_cache()

    return qparams
