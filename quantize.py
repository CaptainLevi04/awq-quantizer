"""
AWQ W4A16 Quantization — main entry point.

Usage:
    python quantize.py                        # uses config.yaml in CWD
    python quantize.py --config my_cfg.yaml   # custom config path
"""

import argparse
import os

import torch
import yaml


def _sanity_check(model, tokenizer, ckpt_dir, num_bits, group_size, exec_device):
    """Load the saved checkpoint, dequantize, and run a generation test."""
    from awq.pack import load_dequantized_into
    from transformers import AutoModelForCausalLM

    print("\n── Sanity check ──────────────────────────────────────────────────")
    ref = AutoModelForCausalLM.from_pretrained(
        ckpt_dir,
        torch_dtype=next(model.parameters()).dtype,
        low_cpu_mem_usage=True,
    ).to(exec_device).eval()
    ref = load_dequantized_into(ref, ckpt_dir, num_bits=num_bits, group_size=group_size)

    prompt = "What is the capital of France?"
    inputs = tokenizer(prompt, return_tensors="pt").to(exec_device)
    with torch.no_grad():
        ids = ref.generate(**inputs, max_new_tokens=64, do_sample=False)
    text = tokenizer.decode(ids[0], skip_special_tokens=True)
    print(f"Prompt : {prompt}")
    print(f"Output : {text}")
    print("── Sanity check passed ───────────────────────────────────────────\n")


def main():
    parser = argparse.ArgumentParser(
        description="AWQ W4A16 quantization — reads config.yaml and runs the full pipeline."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)",
    )
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Environment ──────────────────────────────────────────────────────────
    os.environ["USE_TF"] = "0"
    os.environ["USE_FLAX"] = "0"

    hf_token = cfg["model"].get("hf_token") or None
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    # ── Resolve params ───────────────────────────────────────────────────────
    model_id = cfg["model"]["model_id"]
    torch_dtype = cfg["model"].get("torch_dtype", "auto")
    output_dir = cfg["output"]["dir"]

    q = cfg["quantization"]
    num_bits: int = int(q["num_bits"])
    group_size: int = int(q["group_size"])
    n_grid: int = int(q["n_grid"])
    duo_scaling = q["duo_scaling"]  # "both" | true | false

    cal = cfg["calibration"]
    num_samples: int = int(cal["num_samples"])
    max_seq_len: int = int(cal["max_seq_length"])

    hw = cfg.get("hardware", {})
    cache_device_str: str = hw.get("cache_device", "cuda")

    exec_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_device = torch.device(
        cache_device_str if torch.cuda.is_available() else "cpu"
    )

    print(f"Execution device : {exec_device}")
    print(f"Cache device     : {cache_device}")
    print(f"Model            : {model_id}")
    print(f"Quantization     : W{num_bits}A16, group_size={group_size}, n_grid={n_grid}, duo_scaling={duo_scaling}")
    print(f"Calibration      : {num_samples} samples, max_len={max_seq_len}")
    print()

    # ── Load model & tokenizer ───────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading tokenizer …")
    tok = AutoTokenizer.from_pretrained(
        model_id, token=hf_token, trust_remote_code=True
    )

    print("Loading model …")
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            token=hf_token,
            trust_remote_code=True,
        )
        .to(exec_device)
        .eval()
    )
    print(f"Model dtype : {next(model.parameters()).dtype}\n")

    # ── Calibration data ─────────────────────────────────────────────────────
    from awq.calibration import load_calibration_samples

    print("Loading calibration samples …")
    samples = load_calibration_samples(tok, n=num_samples, max_len=max_seq_len)
    total_tokens = sum(s.numel() for s in samples)
    print(f"{len(samples)} samples | {total_tokens:,} tokens\n")

    # ── Quantize ─────────────────────────────────────────────────────────────
    from awq.pipeline import quantize_model

    print("Quantizing …")
    qparams = quantize_model(
        model,
        samples,
        exec_device,
        group_size=group_size,
        num_bits=num_bits,
        n_grid=n_grid,
        duo_scaling=duo_scaling,
        cache_device=cache_device,
    )
    print(f"\n{len(qparams)} linear layers quantized.\n")

    # ── Save ─────────────────────────────────────────────────────────────────
    from awq.pack import save_checkpoint

    print(f"Saving checkpoint to: {output_dir}")
    save_checkpoint(
        model,
        tok,
        qparams,
        output_dir,
        num_bits=num_bits,
        group_size=group_size,
    )
    print("Saved files:", os.listdir(output_dir))

    # ── Sanity check ─────────────────────────────────────────────────────────
    _sanity_check(model, tok, output_dir, num_bits, group_size, exec_device)

    # ── Hub upload (optional) ─────────────────────────────────────────────────
    hub_cfg = cfg.get("hub", {})
    if hub_cfg.get("enabled"):
        from huggingface_hub import HfApi

        username = hub_cfg.get("username", "")
        if not username:
            raise ValueError("hub.username must be set when hub.enabled is true")

        repo_name = hub_cfg.get("repo_name") or (
            f"{model_id.split('/')[-1]}-AWQ-W{num_bits}A16"
        )
        repo_id = f"{username}/{repo_name}"
        private = hub_cfg.get("private", False)

        print(f"Uploading to Hugging Face Hub: {repo_id} …")
        api = HfApi()
        api.create_repo(
            repo_id, repo_type="model", private=private, exist_ok=True
        )
        api.upload_folder(
            folder_path=output_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload AWQ W{num_bits}A16 asymmetric checkpoint",
        )
        print(f"Done: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
