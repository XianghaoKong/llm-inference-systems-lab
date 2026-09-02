import gc
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

CONTEXT_LENGTHS = [128, 4096]

OUTPUT_DIR = Path("results/profiler")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_exact_length_input(tokenizer, target_length, device):

    base_text = (
        "Artificial intelligence systems use GPUs for parallel computation. "
        "Transformer models contain attention layers and feed-forward networks. "
        "GPU memory bandwidth, computation, and communication affect "
        "large language model inference performance. "
    )

    repeated_text = base_text * 500

    encoded = tokenizer(
        repeated_text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = encoded["input_ids"][:, :target_length].to(device)

    attention_mask = torch.ones_like(
        input_ids,
        device=device,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def warmup(model, inputs, runs=3):

    print("Running warm-up...")

    with torch.inference_mode():

        for _ in range(runs):

            _ = model(
                **inputs,
                use_cache=True,
            )

    torch.cuda.synchronize()

    print("Warm-up complete.")


def profile_prefill(model, inputs, context_length):

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(f"\nProfiling {context_length}-token prefill...")

    torch.cuda.synchronize()

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:

        with record_function(
            f"prefill_{context_length}"
        ):

            with torch.inference_mode():

                outputs = model(
                    **inputs,
                    use_cache=True,
                )

        torch.cuda.synchronize()

    # ------------------------------
    # Operator summary
    # ------------------------------

    table = prof.key_averages(
        group_by_input_shape=True
    ).table(
        sort_by="self_cuda_time_total",
        row_limit=30,
    )

    print("\nTop operators by CUDA time:")
    print(table)

    # ------------------------------
    # Save text table
    # ------------------------------

    table_file = (
        OUTPUT_DIR
        / f"prefill_{context_length}_operators.txt"
    )

    with open(
        table_file,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(table)

    # ------------------------------
    # Export Chrome trace
    # ------------------------------

    trace_file = (
        OUTPUT_DIR
        / f"prefill_{context_length}_trace.json"
    )

    prof.export_chrome_trace(
        str(trace_file)
    )

    peak_memory = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    print(
        f"\nPeak allocated memory: "
        f"{peak_memory:.2f} GB"
    )

    print(f"Operator table saved: {table_file}")
    print(f"Trace saved:          {trace_file}")

    del outputs

    gc.collect()
    torch.cuda.empty_cache()


def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is unavailable."
        )

    device = torch.device("cuda")

    print("=" * 70)
    print("PyTorch Prefill Profiler")
    print("=" * 70)

    print(
        f"GPU          : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"PyTorch      : "
        f"{torch.__version__}"
    )

    print(
        f"CUDA Runtime : "
        f"{torch.version.cuda}"
    )

    print(
        f"Model        : "
        f"{MODEL_NAME}"
    )

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",
    )

    model.eval()

    print(
        f"Attention implementation: "
        f"{getattr(model.config, '_attn_implementation', 'unknown')}"
    )

    # General GPU/model warm-up

    warmup_inputs = create_exact_length_input(
        tokenizer,
        128,
        device,
    )

    warmup(
        model,
        warmup_inputs,
        runs=3,
    )

    del warmup_inputs

    # Profile 128 and 4096

    for context_length in CONTEXT_LENGTHS:

        inputs = create_exact_length_input(
            tokenizer,
            context_length,
            device,
        )

        print(
            f"\nActual input length: "
            f"{inputs['input_ids'].shape[1]}"
        )

        profile_prefill(
            model,
            inputs,
            context_length,
        )

        del inputs


if __name__ == "__main__":
    main()