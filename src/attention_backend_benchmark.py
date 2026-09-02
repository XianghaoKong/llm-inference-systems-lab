import gc
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

CONTEXT_LENGTH = 4096
NUM_RUNS = 5
WARMUP_RUNS = 2

BACKENDS = [
    "eager",
    "sdpa",
]

RESULT_DIR = Path("results/raw")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def synchronize():
    torch.cuda.synchronize()


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

    input_ids = (
        encoded["input_ids"][:, :target_length]
        .to(device)
    )

    attention_mask = torch.ones_like(
        input_ids,
        device=device,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def load_model(attn_backend):

    print(
        f"\nLoading model with "
        f"attn_implementation='{attn_backend}'..."
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation=attn_backend,
    )

    model.eval()

    return model


def warmup(model, inputs):

    print("Warm-up...")

    with torch.inference_mode():

        for _ in range(WARMUP_RUNS):

            _ = model(
                **inputs,
                use_cache=True,
            )

    synchronize()

    print("Warm-up complete.")


def benchmark_prefill(
    model,
    inputs,
    backend,
):

    results = []

    for run in range(
        1,
        NUM_RUNS + 1,
    ):

        gc.collect()

        torch.cuda.reset_peak_memory_stats()

        synchronize()

        start = time.perf_counter()

        with torch.inference_mode():

            outputs = model(
                **inputs,
                use_cache=True,
            )

        synchronize()

        end = time.perf_counter()

        latency_ms = (
            end - start
        ) * 1000

        peak_memory_gb = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        results.append(
            {
                "backend": backend,
                "run": run,
                "context_length": CONTEXT_LENGTH,
                "prefill_latency_ms": latency_ms,
                "peak_memory_gb": peak_memory_gb,
            }
        )

        print(
            f"{backend:>6} | "
            f"Run {run} | "
            f"Prefill={latency_ms:.2f} ms | "
            f"Peak VRAM={peak_memory_gb:.2f} GB"
        )

        del outputs

    return results


def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available."
        )

    device = torch.device("cuda")

    print("=" * 75)
    print("Attention Backend Benchmark")
    print("=" * 75)

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
        f"Context      : "
        f"{CONTEXT_LENGTH} tokens"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    inputs = create_exact_length_input(
        tokenizer,
        CONTEXT_LENGTH,
        device,
    )

    all_results = []

    for backend in BACKENDS:

        model = load_model(
            backend
        )

        print(
            "Configured attention backend:",
            getattr(
                model.config,
                "_attn_implementation",
                "unknown",
            ),
        )

        warmup(
            model,
            inputs,
        )

        backend_results = benchmark_prefill(
            model,
            inputs,
            backend,
        )

        all_results.extend(
            backend_results
        )

        # Remove model before loading next backend
        del model

        gc.collect()
        torch.cuda.empty_cache()
        synchronize()

    df = pd.DataFrame(
        all_results
    )

    raw_path = (
        RESULT_DIR
        / "attention_backend_raw.csv"
    )

    df.to_csv(
        raw_path,
        index=False,
    )

    summary = (
        df.groupby("backend")
        .agg(
            median_prefill_ms=(
                "prefill_latency_ms",
                "median",
            ),
            mean_prefill_ms=(
                "prefill_latency_ms",
                "mean",
            ),
            peak_memory_gb=(
                "peak_memory_gb",
                "max",
            ),
        )
        .reset_index()
    )

    # Add relative speedup
    eager_latency = summary.loc[
        summary["backend"] == "eager",
        "median_prefill_ms",
    ].iloc[0]

    summary["speedup_vs_eager"] = (
        eager_latency
        / summary["median_prefill_ms"]
    )

    summary_path = (
        RESULT_DIR
        / "attention_backend_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {raw_path}"
    )

    print(
        f"Saved: {summary_path}"
    )


if __name__ == "__main__":
    main()