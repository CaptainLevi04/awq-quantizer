# AWQ Quantizer

A clean, dependency-free implementation of **Activation-aware Weight Quantization (AWQ)** for large language models — no `autoawq`, no `llm-compressor`, just pure PyTorch.

Produces **W4A16 asymmetric** checkpoints in the [compressed-tensors](https://github.com/neuralmagic/compressed-tensors) `pack-quantized` format, ready for inference with [vLLM](https://github.com/vllm-project/vllm) and [nm-vllm](https://github.com/neuralmagic/nm-vllm).

---

## Features

- ✅ AWQ smoothing with configurable grid search and duo-scaling
- ✅ MSE-shrink observer for accurate per-group scale/zero-point
- ✅ Bit-packed int32 output (compressed-tensors `pack-quantized`)
- ✅ vLLM-compatible `config.json` with `quantization_config`
- ✅ Optional Hugging Face Hub upload
- ✅ Single `config.yaml` — no code changes needed between runs

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/<your-username>/awq-quantizer.git
cd awq-quantizer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Edit config.yaml
#    Set model_id, output dir, and optionally your HF token / Hub settings

# 4. Run
python quantize.py

# Or point to a custom config file:
python quantize.py --config configs/llama-3.yaml
```

---

## Config Reference

All behaviour is controlled by `config.yaml`. No code changes are needed between different models or quantization settings.

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `model` | `model_id` | `"Qwen/Qwen2.5-0.5B-Instruct"` | Hugging Face model ID |
| `model` | `hf_token` | `""` | HF access token — leave empty for public models |
| `model` | `torch_dtype` | `"auto"` | Weight dtype: `auto`, `float16`, `bfloat16`, `float32` |
| `output` | `dir` | `"./output/..."` | Directory where the quantized checkpoint is written |
| `quantization` | `num_bits` | `4` | Bit-width (4 or 8) |
| `quantization` | `group_size` | `128` | Elements per quantization group (e.g. 64, 128) |
| `quantization` | `n_grid` | `20` | AWQ scale-search grid points |
| `quantization` | `duo_scaling` | `"both"` | AWQ scaling mode: `"both"`, `true`, or `false` |
| `calibration` | `num_samples` | `512` | Number of calibration sequences |
| `calibration` | `max_seq_length` | `1024` | Maximum token length per sequence |
| `hardware` | `cache_device` | `"cuda"` | Device for caching inter-layer activations (`"cuda"` or `"cpu"`) |
| `hub` | `enabled` | `false` | Upload to Hugging Face Hub after quantization |
| `hub` | `username` | `""` | Your HF username |
| `hub` | `repo_name` | `""` | Repo name (auto-generated from model ID if empty) |
| `hub` | `private` | `false` | Make the Hub repo private |

---

## Output Format

The checkpoint is written in the **compressed-tensors `pack-quantized`** format:

| File | Contents |
|------|----------|
| `model.safetensors` | Bit-packed weights, scales, zero-points, and unquantized tensors |
| `config.json` | Model config with embedded `quantization_config` block |
| `tokenizer.*` | Tokenizer files |
| `generation_config.json` | Generation defaults (if present in the base model) |

Each quantized linear layer is stored as four tensors:

```
model.layers.0.self_attn.q_proj.weight_packed       # int32, bit-packed weights
model.layers.0.self_attn.q_proj.weight_scale        # float, per-group scales
model.layers.0.self_attn.q_proj.weight_zero_point   # int32, bit-packed zero-points
model.layers.0.self_attn.q_proj.weight_shape        # int64, original weight shape
```

This layout is natively understood by vLLM ≥ 0.4 and nm-vllm.

---

## Model Compatibility

Works with any standard decoder-only transformer that exposes the following structure:

```
model.model.layers[i]
    .self_attn.{q_proj, k_proj, v_proj, o_proj}
    .mlp.{gate_proj, up_proj, down_proj}
    .input_layernorm
    .post_attention_layernorm
```

Tested with:

| Family | Example models |
|--------|---------------|
| **Qwen 2.5** | `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct` |
| **Llama 3** | `meta-llama/Meta-Llama-3-8B-Instruct` |
| **Mistral** | `mistralai/Mistral-7B-Instruct-v0.3` |
| **Phi-3** | `microsoft/Phi-3-mini-4k-instruct` |

> **Note:** Models with a non-standard MLP (e.g. `gate` + `experts` in MoE) may need minor pipeline adjustments.

---

## Project Structure

```
awq-quantizer/
├── config.yaml              # All user-facing configuration
├── quantize.py              # Main entry point
├── awq/
│   ├── __init__.py
│   ├── primitives.py        # Quantization math: scale/zp, fake-quant, observers
│   ├── smooth.py            # AWQ smoothing: grid search, ArgCache, awq_smooth_mapping
│   ├── pipeline.py          # Layer-by-layer quantization loop
│   ├── pack.py              # Bit-packing, checkpoint save/load
│   └── calibration.py       # Calibration dataset loading
├── requirements.txt
└── README.md
```

---

## How It Works

```
Load model (fp16/bf16)
        │
        ▼
Load calibration data (512 chat sequences, max 1024 tokens)
        │
        ▼
For each decoder layer:
  1. Collect per-channel activation statistics (x_mean)
  2. Collect per-group weight magnitude statistics (w_mean)
  3. Grid-search for the optimal scale vector s that minimises
     output MSE after fake-quantization (AWQ smoothing)
  4. Absorb s into LayerNorm weights; multiply linear weights by s
  5. Run MSE-shrink observer → compute per-group (scale, zero_point)
        │
        ▼
Pack weights to int32 (4-bit × 8 elements per int32 word)
        │
        ▼
Save compressed-tensors checkpoint + config.json
        │
        ▼
(Optional) Upload to Hugging Face Hub
```

---

## License

MIT
