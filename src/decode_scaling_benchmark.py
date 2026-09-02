import gc
import platform
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

INPUT_TOKENS = 128
OUTPUT_LENGTHS = [32, 128, 256, 512]

WARMUP_RUNS = 2
MEASURED_RUNS = 5

DEVICE = "cuda"
DTYPE = torch.float16

BACKENDS = ["eager", "flash"]

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results" / "raw"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RUNS_CSV = RESULTS_DIR / "decode_scaling_runs.csv"
TOKENS_CSV = RESULTS_DIR / "decode_token_latency.csv"
SUMMARY_CSV = RESULTS_DIR / "decode_scaling_summary.csv"


# ============================================================
# Utilities
# ============================================================

def gb(x):
    return x / (1024 ** 3)


def backend_context(backend):
    if backend == "flash":
        return sdpa_kernel(SDPBackend.FLASH_ATTENTION)

    return nullcontext()


def cuda_timed_forward(fn):
    """
    Measure one GPU operation using CUDA events.
    Includes the forward pass and token selection performed inside fn().
    """

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()

    start.record()
    result = fn()
    end.record()

    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)

    return result, elapsed_ms


# ============================================================
# Controlled 128-token prompt
# ============================================================

def build_input(tokenizer):
    text = (
        "Large language model inference contains a prefill phase and an "
        "autoregressive decode phase. During decoding, previously computed "
        "key and value tensors are stored in a cache so that earlier tokens "
        "do not need to be recomputed. This benchmark studies GPU inference "
        "performance under a controlled workload. "
    )

    # Repeat semantic text so we definitely have >128 tokens.
    text = (text + " ") * 20

    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = encoded.input_ids[:, :INPUT_TOKENS]

    if input_ids.shape[1] != INPUT_TOKENS:
        raise RuntimeError(
            f"Expected {INPUT_TOKENS} tokens, "
            f"but got {input_ids.shape[1]}"
        )

    attention_mask = torch.ones_like(input_ids)

    return (
        input_ids.to(DEVICE),
        attention_mask.to(DEVICE),
    )


# ============================================================
# One autoregressive generation run
# ============================================================

@torch.inference_mode()
def run_generation(
    model,
    input_ids,
    attention_mask,
    output_tokens,
    backend,
):
    """
    Manual autoregressive decoding.

    Prefill:
        prompt -> first output token + KV cache

    Decode:
        one token at a time using past_key_values
    """

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    allocated_before = torch.cuda.memory_allocated()


    # --------------------------------------------------------
    # PREFILL
    # Produces output token #1
    # --------------------------------------------------------

    def prefill_step():
        with backend_context(backend):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        return outputs.past_key_values, next_token


    (past_key_values, next_token), prefill_ms = cuda_timed_forward(
        prefill_step
    )

    torch.cuda.synchronize()

    allocated_after_prefill = torch.cuda.memory_allocated()


    # --------------------------------------------------------
    # DECODE
    # token #2 ... token #N
    # --------------------------------------------------------

    current_attention_mask = attention_mask

    decode_times = []

    for token_index in range(2, output_tokens + 1):

        current_attention_mask = torch.cat(
            [
                current_attention_mask,
                torch.ones(
                    (1, 1),
                    dtype=current_attention_mask.dtype,
                    device=DEVICE,
                ),
            ],
            dim=1,
        )

        def decode_step():
            with backend_context(backend):
                outputs = model(
                    input_ids=next_token,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            new_token = torch.argmax(
                outputs.logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            return outputs.past_key_values, new_token


        (past_key_values, new_token), latency_ms = cuda_timed_forward(
            decode_step
        )

        next_token = new_token

        decode_times.append(
            {
                "token_index": token_index,
                "latency_ms": latency_ms,
            }
        )


    torch.cuda.synchronize()

    final_allocated = torch.cuda.memory_allocated()
    peak_allocated = torch.cuda.max_memory_allocated()


    # First output token comes from prefill.
    total_decode_ms = sum(
        item["latency_ms"]
        for item in decode_times
    )

    e2e_ms = prefill_ms + total_decode_ms

    decode_token_count = max(output_tokens - 1, 0)

    if decode_token_count > 0:
        decode_tok_s = (
            decode_token_count /
            (total_decode_ms / 1000.0)
        )
    else:
        decode_tok_s = float("nan")


    result = {
        "prefill_ms": prefill_ms,
        "total_decode_ms": total_decode_ms,
        "e2e_ms": e2e_ms,
        "decode_tok_s": decode_tok_s,

        "allocated_before_gb": gb(allocated_before),
        "allocated_after_prefill_gb": gb(
            allocated_after_prefill
        ),
        "final_allocated_gb": gb(final_allocated),
        "peak_allocated_gb": gb(peak_allocated),
    }


    # Release KV cache before next run.
    del past_key_values
    del next_token

    gc.collect()
    torch.cuda.empty_cache()

    return result, decode_times


# ============================================================
# Load one backend at a time
# ============================================================

def load_model(backend):
    if backend == "eager":
        attention_impl = "eager"

    elif backend == "flash":
        attention_impl = "sdpa"

    else:
        raise ValueError(
            f"Unknown backend: {backend}"
        )

    print(
        f"\nLoading model with "
        f"attn_implementation={attention_impl}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        attn_implementation=attention_impl,
    )

    model = model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# Main benchmark
# ============================================================

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    torch.manual_seed(0)

    print("=" * 70)
    print("E3 — Autoregressive Decode Scaling")
    print("=" * 70)

    print(f"Model:          {MODEL_NAME}")
    print(f"GPU:            {torch.cuda.get_device_name(0)}")
    print(f"PyTorch:        {torch.__version__}")
    print(f"Transformers:   {transformers.__version__}")
    print(f"CUDA runtime:   {torch.version.cuda}")
    print(f"Python:         {platform.python_version()}")

    print(f"Input tokens:   {INPUT_TOKENS}")
    print(f"Output lengths: {OUTPUT_LENGTHS}")
    print(f"Warmups:        {WARMUP_RUNS}")
    print(f"Measured runs:  {MEASURED_RUNS}")
    print("=" * 70)


    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    input_ids, attention_mask = build_input(
        tokenizer
    )

    print(
        f"Actual input length: "
        f"{input_ids.shape[1]} tokens"
    )


    all_runs = []
    all_token_times = []


    # ========================================================
    # Eager then Flash
    # ========================================================

    for backend in BACKENDS:

        print("\n" + "=" * 70)
        print(f"BACKEND: {backend.upper()}")
        print("=" * 70)

        model = load_model(backend)


        for output_tokens in OUTPUT_LENGTHS:

            print(
                f"\n--- Output length: "
                f"{output_tokens} tokens ---"
            )


            # ------------------------------------------------
            # Warmup
            # ------------------------------------------------

            for warmup in range(WARMUP_RUNS):

                print(
                    f"Warmup "
                    f"{warmup + 1}/{WARMUP_RUNS}"
                )

                run_generation(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_tokens=output_tokens,
                    backend=backend,
                )


            # ------------------------------------------------
            # Measured runs
            # ------------------------------------------------

            for run_idx in range(
                1,
                MEASURED_RUNS + 1,
            ):

                print(
                    f"Measured run "
                    f"{run_idx}/{MEASURED_RUNS}"
                )

                run_result, token_times = run_generation(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_tokens=output_tokens,
                    backend=backend,
                )


                run_row = {
                    "backend": backend,
                    "input_tokens": INPUT_TOKENS,
                    "output_tokens": output_tokens,
                    "run": run_idx,
                    **run_result,
                }

                all_runs.append(run_row)


                for item in token_times:

                    all_token_times.append(
                        {
                            "backend": backend,
                            "input_tokens": INPUT_TOKENS,
                            "output_tokens": output_tokens,
                            "run": run_idx,
                            "token_index": item[
                                "token_index"
                            ],
                            "latency_ms": item[
                                "latency_ms"
                            ],
                        }
                    )


                print(
                    f"  Prefill: "
                    f"{run_result['prefill_ms']:.3f} ms"
                )

                print(
                    f"  Decode:  "
                    f"{run_result['total_decode_ms']:.3f} ms"
                )

                print(
                    f"  E2E:     "
                    f"{run_result['e2e_ms']:.3f} ms"
                )

                print(
                    f"  Decode throughput: "
                    f"{run_result['decode_tok_s']:.2f} tok/s"
                )

                print(
                    f"  Peak allocated: "
                    f"{run_result['peak_allocated_gb']:.3f} GB"
                )


        # Unload before loading next backend.
        del model

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


    # ========================================================
    # Save raw run data
    # ========================================================

    runs_df = pd.DataFrame(all_runs)
    tokens_df = pd.DataFrame(all_token_times)

    runs_df.to_csv(
        RUNS_CSV,
        index=False,
    )

    tokens_df.to_csv(
        TOKENS_CSV,
        index=False,
    )


    # ========================================================
    # Aggregate summary
    # ========================================================

    summaries = []

    for backend in BACKENDS:

        for output_tokens in OUTPUT_LENGTHS:

            run_subset = runs_df[
                (runs_df["backend"] == backend)
                &
                (
                    runs_df["output_tokens"]
                    == output_tokens
                )
            ]

            token_subset = tokens_df[
                (tokens_df["backend"] == backend)
                &
                (
                    tokens_df["output_tokens"]
                    == output_tokens
                )
            ]

            latencies = token_subset[
                "latency_ms"
            ].to_numpy()


            summaries.append(
                {
                    "backend": backend,
                    "input_tokens": INPUT_TOKENS,
                    "output_tokens": output_tokens,

                    "prefill_median_ms":
                        run_subset[
                            "prefill_ms"
                        ].median(),

                    "e2e_median_ms":
                        run_subset[
                            "e2e_ms"
                        ].median(),

                    "decode_median_total_ms":
                        run_subset[
                            "total_decode_ms"
                        ].median(),

                    "mean_tpot_ms":
                        np.mean(latencies),

                    "p50_tpot_ms":
                        np.percentile(
                            latencies,
                            50,
                        ),

                    "p95_tpot_ms":
                        np.percentile(
                            latencies,
                            95,
                        ),

                    "decode_tok_s_median":
                        run_subset[
                            "decode_tok_s"
                        ].median(),

                    "peak_memory_gb_median":
                        run_subset[
                            "peak_allocated_gb"
                        ].median(),

                    "post_prefill_memory_gb_median":
                        run_subset[
                            "allocated_after_prefill_gb"
                        ].median(),

                    "final_memory_gb_median":
                        run_subset[
                            "final_allocated_gb"
                        ].median(),
                }
            )


    summary_df = pd.DataFrame(summaries)

    summary_df.to_csv(
        SUMMARY_CSV,
        index=False,
    )


    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(RUNS_CSV)
    print(TOKENS_CSV)
    print(SUMMARY_CSV)


if __name__ == "__main__":
    main()