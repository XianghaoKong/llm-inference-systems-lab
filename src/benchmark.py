import gc
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 100
NUM_RUNS = 5
WARMUP_TOKENS = 20

RESULT_PATH = Path("results/raw")
RESULT_PATH.mkdir(parents=True, exist_ok=True)


def synchronize():
    """
    CUDA operations are asynchronous.
    Synchronizing ensures timing includes actual GPU execution.
    """
    torch.cuda.synchronize()


def measure_ttft(model, inputs):
    """
    Approximate Time To First Token (TTFT).

    A causal LM forward pass over the complete prompt performs
    the prefill stage and produces logits for the first output token.
    """

    synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model(
            **inputs,
            use_cache=True,
        )

        # Select the first output token using greedy decoding.
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
    Measure complete generation latency, throughput and peak GPU memory.
    """

    gc.collect()

    torch.cuda.reset_peak_memory_stats()

    synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    synchronize()

    end = time.perf_counter()

    total_latency = end - start

    output_tokens = output.shape[1] - input_length

    throughput = output_tokens / total_latency

    peak_memory_gb = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    del output

    return {
        "total_latency_s": total_latency,
        "output_tokens": output_tokens,
        "throughput_tok_s": throughput,
        "peak_memory_gb": peak_memory_gb,
    }


def main():

    print("=" * 70)
    print("LLM Inference Performance Benchmark")
    print("=" * 70)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    device = torch.device("cuda")

    print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"CUDA Runtime  : {torch.version.cuda}")
    print(f"Model         : {MODEL_NAME}")

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

    messages = [
        {
            "role": "user",
            "content":
                "Explain why GPU parallel computing "
                "is useful for deep learning."
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(device)

    input_length = inputs.input_ids.shape[1]

    print(f"Input tokens  : {input_length}")

    # ------------------------------------------------
    # Warm-up
    # ------------------------------------------------

    print("\nRunning GPU warm-up...")

    with torch.inference_mode():
        _ = model.generate(
            **inputs,
            max_new_tokens=WARMUP_TOKENS,
            do_sample=False,
        )

    synchronize()

    print("Warm-up complete.")

    # ------------------------------------------------
    # Benchmark
    # ------------------------------------------------

    results = []

    print("\nStarting benchmark...\n")

    for run in range(1, NUM_RUNS + 1):

        ttft_ms = measure_ttft(
            model,
            inputs,
        )

        generation_metrics = measure_generation(
            model,
            inputs,
            input_length,
        )

        output_tokens = generation_metrics["output_tokens"]
        total_latency = generation_metrics["total_latency_s"]

        # Approximate average time per output token after the first token.
        if output_tokens > 1:
            tpot_ms = (
                (total_latency - ttft_ms / 1000)
                / (output_tokens - 1)
                * 1000
            )
        else:
            tpot_ms = float("nan")

        result = {
            "run": run,
            "input_tokens": input_length,
            "output_tokens": output_tokens,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            **generation_metrics,
        }

        results.append(result)

        print(
            f"Run {run}: "
            f"TTFT={ttft_ms:.2f} ms | "
            f"TPOT={tpot_ms:.2f} ms | "
            f"Latency={total_latency:.3f} s | "
            f"Throughput="
            f"{generation_metrics['throughput_tok_s']:.2f} tok/s | "
            f"Peak VRAM="
            f"{generation_metrics['peak_memory_gb']:.2f} GB"
        )

    # ------------------------------------------------
    # Save raw results
    # ------------------------------------------------

    df = pd.DataFrame(results)

    output_file = (
        RESULT_PATH
        / "baseline_benchmark.csv"
    )

    df.to_csv(
        output_file,
        index=False,
    )

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print(
        f"Median TTFT       : "
        f"{df['ttft_ms'].median():.2f} ms"
    )

    print(
        f"Median TPOT       : "
        f"{df['tpot_ms'].median():.2f} ms"
    )

    print(
        f"Median Throughput : "
        f"{df['throughput_tok_s'].median():.2f} tokens/s"
    )

    print(
        f"Median Latency    : "
        f"{df['total_latency_s'].median():.3f} s"
    )

    print(
        f"Peak VRAM         : "
        f"{df['peak_memory_gb'].max():.2f} GB"
    )

    print(f"\nResults saved to: {output_file}")

    print("=" * 70)


if __name__ == "__main__":
    main()