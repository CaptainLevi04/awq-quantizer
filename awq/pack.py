"""
Checkpoint packing and saving in compressed-tensors pack-quantized format.

Weights are stored as bit-packed int32 tensors compatible with vLLM /
nm-vllm's ``pack-quantized`` loader.
"""

import json
import math
import os

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Bit-packing helpers
# ---------------------------------------------------------------------------

def pack_to_int32(value: torch.Tensor, num_bits: int, packed_dim: int = 1) -> torch.Tensor:
    """Pack an int8 tensor into int32 words using *num_bits* per element.

    Args:
        value: Integer tensor with dtype ``torch.int8``.
        num_bits: Bits per element (e.g., 4 for W4 quantization).
        packed_dim: Dimension along which to pack (0 = row-wise, 1 = col-wise).

    Returns:
        Packed ``torch.int32`` tensor.
    """
    assert value.dtype == torch.int8, "Input must be int8"
    value = value.to(torch.int32) + (1 << (num_bits - 1))

    if packed_dim == 0:
        value = value.transpose(0, 1)

    rows, cols = value.shape
    packed_cols = math.ceil(cols * num_bits / 32)
    padded = math.ceil(cols / 32) * 32

    if padded > cols:
        value = F.pad(value, (0, padded - cols))

    groups = padded // 32
    vg = value.reshape(rows * groups, 32)
    out = torch.zeros(rows * groups, num_bits, dtype=torch.int32, device=value.device)

    bit_starts = torch.arange(32, device=value.device, dtype=torch.int32) * num_bits
    word_idx = (bit_starts // 32).long()
    bit_off = bit_starts % 32

    out.scatter_add_(
        1,
        word_idx.unsqueeze(0).expand(rows * groups, -1),
        vg << bit_off.unsqueeze(0),
    )

    ov = bit_off + num_bits - 32
    m = ov > 0
    if m.any():
        out.scatter_add_(
            1,
            (word_idx[m] + 1).unsqueeze(0).expand(rows * groups, -1),
            vg[:, m] >> (num_bits - ov[m]).unsqueeze(0),
        )

    out = out.view(rows, groups * num_bits)[:, :packed_cols]
    return out.transpose(0, 1) if packed_dim == 0 else out


def unpack_from_int32(
    value: torch.Tensor,
    num_bits: int,
    shape,
    packed_dim: int = 1,
) -> torch.Tensor:
    """Unpack bit-packed int32 tensor back to int8.

    Args:
        value: Packed ``torch.int32`` tensor produced by ``pack_to_int32``.
        num_bits: Bits per element.
        shape: Original tensor shape (used to determine the unpacked column count).
        packed_dim: Dimension that was packed (must match the ``pack_to_int32`` call).

    Returns:
        Unpacked ``torch.int8`` tensor.
    """
    if packed_dim == 0:
        value = value.transpose(0, 1)

    rows, words = value.shape
    cols = int(shape[packed_dim])

    if words % num_bits:
        value = F.pad(value, (0, num_bits - words % num_bits))
        words = value.shape[1]

    groups = words // num_bits
    vg = value.reshape(rows * groups, num_bits)

    bit_starts = torch.arange(32, device=value.device, dtype=torch.int32) * num_bits
    word_idx = (bit_starts // 32).long()
    bit_off = bit_starts % 32
    lo = torch.clamp(32 - bit_off, max=num_bits)

    out = (vg[:, word_idx] >> bit_off.unsqueeze(0)) & ((1 << lo) - 1).unsqueeze(0)

    m = lo < num_bits
    hi = num_bits - lo[m]
    out[:, m] |= (
        (vg[:, word_idx[m] + 1] & ((1 << hi) - 1).unsqueeze(0))
        << lo[m].unsqueeze(0)
    )

    out = out.view(rows, groups * 32)[:, :cols]
    if packed_dim == 0:
        out = out.transpose(0, 1)

    return (out - (1 << (num_bits - 1))).to(torch.int8)


# ---------------------------------------------------------------------------
# Integer quantization helper
# ---------------------------------------------------------------------------

def quantize_to_int8(
    w: torch.Tensor,
    scale: torch.Tensor,
    zp: torch.Tensor,
    group_size: int,
    num_bits: int = 4,
) -> torch.Tensor:
    """Quantize a weight tensor to int8 using per-group scale and zero-point.

    The quantized range is ``[-(2**(num_bits-1)), 2**(num_bits-1)-1]``, e.g.,
    ``[-8, 7]`` for 4-bit.

    Args:
        w: Float weight tensor of shape ``[out_features, in_features]``.
        scale: Per-group scale of shape ``[out_features, num_groups]``.
        zp: Per-group zero-point (int8) of shape ``[out_features, num_groups]``.
        group_size: Number of elements per quantization group.
        num_bits: Quantization bit-width.

    Returns:
        Quantized ``torch.int8`` tensor of the same spatial shape as *w*.
    """
    q_min = -(2 ** (num_bits - 1))
    q_max = 2 ** (num_bits - 1) - 1

    x = w.unflatten(-1, (-1, group_size))
    scaled = x / scale.unsqueeze(-1)
    scaled = scaled + zp.unsqueeze(-1).to(x.dtype)
    q = torch.round(torch.clamp(scaled, q_min, q_max))
    return q.flatten(-2).to(torch.int8)


# ---------------------------------------------------------------------------
# Quantization config
# ---------------------------------------------------------------------------

def build_quantization_config(num_bits: int, group_size: int) -> dict:
    """Build the ``quantization_config`` dict for ``config.json``.

    Generates a ``compressed-tensors`` / ``pack-quantized`` configuration
    compatible with vLLM and nm-vllm.

    Args:
        num_bits: Quantization bit-width.
        group_size: Quantization group size.

    Returns:
        Dict suitable for embedding in the model's ``config.json``.
    """
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "ignore": ["lm_head"],
        "kv_cache_scheme": None,
        "global_compression_ratio": None,
        "sparsity_config": {},
        "transform_config": {},
        "version": "0.19.0",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "input_activations": None,
                "output_activations": None,
                "format": "pack-quantized",
                "weights": {
                    "num_bits": num_bits,
                    "type": "int",
                    "symmetric": False,
                    "strategy": "group",
                    "group_size": group_size,
                    "dynamic": False,
                    "observer": "mse",
                    "observer_kwargs": {},
                    "actorder": None,
                    "block_structure": None,
                    "scale_dtype": None,
                    "zp_dtype": "torch.int8",
                },
            }
        },
    }


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_checkpoint(
    model,
    tokenizer,
    qparams: dict,
    out_dir: str,
    num_bits: int = 4,
    group_size: int = 128,
) -> dict:
    """Serialize the quantized model to *out_dir* in pack-quantized format.

    Quantized linear layers are stored as:
    - ``<name>.weight_packed``     — bit-packed weights (int32)
    - ``<name>.weight_scale``      — per-group float scales
    - ``<name>.weight_zero_point`` — bit-packed zero-points (int32)
    - ``<name>.weight_shape``      — original weight shape (int64 tensor)

    All other parameters (embeddings, norms, …) are stored as-is.

    Args:
        model: Quantized causal-LM model.
        tokenizer: Tokenizer to save alongside the weights.
        qparams: Dict mapping layer name → ``(scale, zero_point)`` tensors.
        out_dir: Output directory path.
        num_bits: Quantization bit-width.
        group_size: Quantization group size.

    Returns:
        Dict of all tensors written to the safetensors file.
    """
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    tensors = {}
    quantized = set(qparams)

    for name, p in model.state_dict().items():
        mod_name = name.rsplit(".", 1)[0]

        # lm_head is excluded from quantization (kept in float)
        if name == "lm_head.weight":
            continue

        if name.endswith(".weight") and mod_name in quantized:
            scale, zp = qparams[mod_name]
            w = p.detach().cpu()
            scale, zp = scale.cpu(), zp.cpu()

            tensors[mod_name + ".weight_packed"] = pack_to_int32(
                quantize_to_int8(w, scale, zp, group_size, num_bits),
                num_bits,
            ).contiguous()
            tensors[mod_name + ".weight_scale"] = scale.contiguous()
            tensors[mod_name + ".weight_zero_point"] = pack_to_int32(
                zp, num_bits, packed_dim=0
            ).contiguous()
            tensors[mod_name + ".weight_shape"] = torch.tensor(w.shape)
        else:
            tensors[name] = p.detach().cpu().contiguous()

    save_file(
        tensors,
        os.path.join(out_dir, "model.safetensors"),
        metadata={"format": "pt"},
    )

    # Write config.json with embedded quantization config
    cfg = model.config.to_dict()
    cfg["quantization_config"] = build_quantization_config(num_bits, group_size)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Optional generation config
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(out_dir)

    tokenizer.save_pretrained(out_dir)
    return tensors


@torch.no_grad()
def load_dequantized_into(model, ckpt_dir: str, num_bits: int = 4, group_size: int = 128):
    """Load a pack-quantized checkpoint and dequantize weights into *model*.

    Useful for sanity-checking or running inference without a dedicated
    pack-quantized kernel.

    Args:
        model: An unquantized model with the same architecture as the
            checkpoint (weights will be overwritten in-place).
        ckpt_dir: Directory containing ``model.safetensors``.
        num_bits: Quantization bit-width used when the checkpoint was saved.
        group_size: Quantization group size used when the checkpoint was saved.

    Returns:
        The *model* with dequantized weights loaded.
    """
    from safetensors.torch import load_file

    t = load_file(os.path.join(ckpt_dir, "model.safetensors"))
    sd = model.state_dict()

    for key in [k for k in t if k.endswith(".weight_packed")]:
        base = key[: -len(".weight_packed")]
        shape = tuple(t[base + ".weight_shape"].tolist())
        scale = t[base + ".weight_scale"]
        zp = unpack_from_int32(
            t[base + ".weight_zero_point"],
            num_bits,
            (shape[0], scale.shape[1]),
            packed_dim=0,
        )
        qi = unpack_from_int32(t[key], num_bits, shape)
        x = (
            qi.unflatten(-1, (-1, group_size)).to(scale.dtype)
            - zp.unsqueeze(-1).to(scale.dtype)
        )
        sd[base + ".weight"].copy_(
            (x * scale.unsqueeze(-1))
            .flatten(-2)
            .to(sd[base + ".weight"].dtype)
        )

    for k, v in t.items():
        if not k.endswith(
            ("weight_packed", "weight_scale", "weight_zero_point", "weight_shape")
        ):
            sd[k].copy_(v)

    return model
