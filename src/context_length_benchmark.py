import gc
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

CONTEXT_LENGTHS = [
    128,
    256,
    512,
    1024,
    2048,
    4096,
]

MAX_NEW_TOKENS = 100
NUM_RUNS = 3

RESULT_DIR = Path("results/raw")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def synchronize():
    torch.cuda.synchronize()


def create_exact_length_input(tokenizer, target_length, device):
    """
    Create an input containing exactly target_length tokens.

    For this performance experiment, semantic quality is not important.
    We use repeated natural-language text, tokenize it once,
    and slice the token IDs to the required length.
    """

    base_text = (
        "Artificial intelligence systems use GPUs for parallel computation. "
        "Transformer models contain attention layers and feed-forward networks. "
        "GPU memory bandwidth, computation, and communication all affect "
        "large language model inference performance. "
    )

    # Repeat enough times to exceed the longest target length.
    repeated_text = base_text * 500

    encoded = tokenizer(
        repeated_text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = encoded["input_ids"]

    if input_ids.shape[1] < target_length:
        raise ValueError(
            f"Generated text only contains {input_ids.shape[1]} tokens, "
            f"which is shorter than target {target_length}."
        )

    # Slice token IDs directly so the input length is exact.
    input_ids = input_ids[:, :target_length].to(device)

    attention_mask = torch.ones_like(
        input_ids,
        device=device,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def measure_ttft(model, inputs):
    """
    Approximate TTFT by measuring the prefill forward pass
    that produces logits for the first generated token.
    """

    synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model(
            **inputs,
            use_cache=True,
        )

        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
        )

    synchronize()

    end = time.perf_counter()

    ttft_ms = (end - start) * 1000

    del outputs
    del next_token

    return ttft_ms


def measure_generation(model, inputs, input_length):
    """
    Measure full generation latency and peak GPU memory.
    """

    gc.collect()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
             min_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )

    synchronize()

    end = time.perf_counter()

    total_latency_s = end - start

    output_tokens = (
        output.shape[1] - input_length
    )

    throughput = (
        output_tokens / total_latency_s
    )

    peak_memory_gb = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    del output

    return {
        "total_latency_s": total_latency_s,
        "output_tokens": output_tokens,
        "throughput_tok_s": throughput,
        "peak_memory_gb": peak_memory_gb,
    }


def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available."
        )

    device = torch.device("cuda")

    print("=" * 75)
    print("Context Length Performance Benchmark")
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

    # ----------------------------------------
    # General warm-up
    # ----------------------------------------

    print("\nRunning warm-up...")

    warmup_inputs = create_exact_length_input(
        tokenizer,
        target_length=128,
        device=device,
    )

    with torch.inference_mode():
        _ = model.generate(
            **warmup_inputs,
            max_new_tokens=20,
            do_sample=False,
        )

    synchronize()

    del warmup_inputs

    print("Warm-up complete.\n")

    results = []

    # ----------------------------------------
    # Context-length experiments
    # ----------------------------------------

    for context_length in CONTEXT_LENGTHS:

        print("=" * 75)
        print(
            f"Testing context length: "
            f"{context_length} tokens"
        )
        print("=" * 75)

        inputs = create_exact_length_input(
            tokenizer,
            target_length=context_length,
            device=device,
        )

        actual_length = (
            inputs["input_ids"].shape[1]
        )

        print(
            f"Actual input tokens: "
            f"{actual_length}"
        )

        for run in range(
            1,
            NUM_RUNS + 1
        ):

            ttft_ms = measure_ttft(
                model,
                inputs,
            )

            generation = measure_generation(
                model,
                inputs,
                actual_length,
            )

            output_tokens = (
                generation["output_tokens"]
            )

            total_latency = (
                generation["total_latency_s"]
            )

            if output_tokens > 1:

                tpot_ms = (
                    (
                        total_latency
                        - ttft_ms / 1000
                    )
                    / (output_tokens - 1)
                    * 1000
                )

            else:
                tpot_ms = float("nan")

            result = {
                "context_length": (
                    context_length
                ),
                "run": run,
                "output_tokens": (
                    output_tokens
                ),
                "ttft_ms": (
                    ttft_ms
                ),
                "tpot_ms": (
                    tpot_ms
                ),
                "total_latency_s": (
                    total_latency
                ),
                "throughput_tok_s": (
                    generation[
                        "throughput_tok_s"
                    ]
                ),
                "peak_memory_gb": (
                    generation[
                        "peak_memory_gb"
                    ]
                ),
            }

            results.append(result)

            print(
                f"Run {run}: "
                f"TTFT={ttft_ms:.2f} ms | "
                f"TPOT={tpot_ms:.2f} ms | "
                f"Throughput="
                f"{generation['throughput_tok_s']:.2f} tok/s | "
                f"Peak VRAM="
                f"{generation['peak_memory_gb']:.2f} GB"
            )

        print()

        del inputs

        gc.collect()
        torch.cuda.empty_cache()

    # ----------------------------------------
    # Save raw results
    # ----------------------------------------

    df = pd.DataFrame(results)

    raw_file = (
        RESULT_DIR
        / "context_length_raw.csv"
    )

    df.to_csv(
        raw_file,
        index=False,
    )

    # ----------------------------------------
    # Aggregate results
    # ----------------------------------------

    summary = (
        df.groupby(
            "context_length"
        )
        .agg(
            median_output_tokens=(
                "output_tokens",
                "median",
            ),
            median_ttft_ms=(
                "ttft_ms",
                "median",
            ),
            median_tpot_ms=(
                "tpot_ms",
                "median",
            ),
            median_latency_s=(
                "total_latency_s",
                "median",
            ),
            median_throughput_tok_s=(
                "throughput_tok_s",
                "median",
            ),
            peak_memory_gb=(
                "peak_memory_gb",
                "max",
            ),
        )
        .reset_index()
    )

    summary_file = (
        RESULT_DIR
        / "context_length_summary.csv"
    )

    summary.to_csv(
        summary_file,
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
        f"\nRaw results saved to: "
        f"{raw_file}"
    )

    print(
        f"Summary saved to: "
        f"{summary_file}"
    )


if __name__ == "__main__":
    main()