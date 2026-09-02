import gc
from pathlib import Path

import torch
from torch.profiler import (
    profile,
    ProfilerActivity,
    record_function,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

CONTEXT_LENGTH = 4096
BACKENDS = ["eager", "sdpa"]

OUTPUT_DIR = Path("results/profiler")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def synchronize():
    torch.cuda.synchronize()


def create_exact_length_input(
    tokenizer,
    target_length,
    device,
):
    base_text = (
        "Artificial intelligence systems use GPUs "
        "for parallel computation. "
        "Transformer models contain attention layers "
        "and feed-forward networks. "
        "GPU memory bandwidth, computation, and "
        "communication affect large language model "
        "inference performance. "
    )

    repeated_text = base_text * 500

    encoded = tokenizer(
        repeated_text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = (
        encoded["input_ids"][:, :target_length]
        .to(device)
    )

    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(
            input_ids,
            device=device,
        ),
    }


def run_profile(
    tokenizer,
    inputs,
    backend,
):

    print("\n" + "=" * 70)
    print(f"Backend: {backend}")
    print("=" * 70)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation=backend,
    )

    model.eval()

    print(
        "Configured implementation:",
        getattr(
            model.config,
            "_attn_implementation",
            "unknown",
        ),
    )

    # Warm-up
    with torch.inference_mode():
        for _ in range(2):
            _ = model(
                **inputs,
                use_cache=True,
            )

    synchronize()

    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
    ) as prof:

        with record_function(
            f"prefill_4096_{backend}"
        ):

            with torch.inference_mode():
                output = model(
                    **inputs,
                    use_cache=True,
                )

        synchronize()

    table = (
        prof.key_averages(
            group_by_input_shape=True
        )
        .table(
            sort_by="self_cuda_time_total",
            row_limit=40,
        )
    )

    print(table)

    table_path = (
        OUTPUT_DIR
        / f"prefill_4096_{backend}_operators.txt"
    )

    with open(
        table_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(table)

    trace_path = (
        OUTPUT_DIR
        / f"prefill_4096_{backend}_trace.json"
    )

    prof.export_chrome_trace(
        str(trace_path)
    )

    peak_memory = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    print(
        f"\nPeak memory: {peak_memory:.2f} GB"
    )

    del output
    del model

    gc.collect()
    torch.cuda.empty_cache()
    synchronize()


def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU unavailable."
        )

    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    inputs = create_exact_length_input(
        tokenizer,
        CONTEXT_LENGTH,
        device,
    )

    for backend in BACKENDS:
        run_profile(
            tokenizer,
            inputs,
            backend,
        )


if __name__ == "__main__":
    main()