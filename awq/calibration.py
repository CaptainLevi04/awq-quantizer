"""
Calibration data loading for AWQ quantization.

Uses the ``neuralmagic/LLM_compression_calibration`` dataset.
"""


def load_calibration_samples(tokenizer, n: int = 512, max_len: int = 1024):
    """Load and tokenize calibration samples.

    Fetches samples from the ``neuralmagic/LLM_compression_calibration``
    dataset, applies the tokenizer's chat template, and returns tokenized
    tensors ready for the quantization pipeline.

    Args:
        tokenizer: A Hugging Face tokenizer (must support ``apply_chat_template``).
        n: Number of calibration samples to return.
        max_len: Maximum sequence length (longer sequences are truncated).

    Returns:
        List of ``torch.LongTensor`` of shape ``[1, seq_len]``.
    """
    from datasets import load_dataset

    ds = load_dataset("neuralmagic/LLM_compression_calibration", split="train")
    ds = ds.shuffle(seed=42).select(range(min(n * 2, len(ds))))

    samples = []
    for ex in ds:
        text = tokenizer.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        ids = tokenizer(
            text,
            padding=False,
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        samples.append(ids)
        if len(samples) == n:
            break

    return samples
