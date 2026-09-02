from pathlib import Path
import gc
import platform

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

INPUT_LENGTHS = [64, 128, 256]
OUTPUT_TOKENS = 520

WARMUP_RUNS = 2
MEASURED_RUNS = 5

DEVICE = "cuda"
DTYPE = torch.float16

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results" / "raw"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = RESULTS_DIR / "e3b_boundary_token_latency.csv"
BOUNDARY_CSV = RESULTS_DIR / "e3b_boundary_summary.csv"


# ============================================================
# Helpers
# ============================================================

def to_mib(x):
    return x / (1024 ** 2)


def build_input(tokenizer, target_length):
    """
    Build a semantic prompt and truncate it to exactly target_length tokens.
    """

    text = (
        "Large language model inference consists of a prefill phase and an "
        "autoregressive decode phase. During decoding, previously computed "
        "key and value tensors are retained in the KV cache. This controlled "
        "experiment studies whether sequence-length boundaries produce "
        "repeatable changes in GPU decode latency. "
    )

    text = (text + " ") * 30

    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = encoded.input_ids[:, :target_length]

    if input_ids.shape[1] != target_length:
        raise RuntimeError(
            f"Expected {target_length} tokens, "
            f"got {input_ids.shape[1]}"
        )

    attention_mask = torch.ones_like(input_ids)

    return (
        input_ids.to(DEVICE),
        attention_mask.to(DEVICE),
    )


def timed_forward(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()

    start.record()
    result = fn()
    end.record()

    torch.cuda.synchronize()

    return result, start.elapsed_time(end)


# ============================================================
# One fixed-length generation
# ============================================================

@torch.inference_mode()
def run_generation(
    model,
    input_ids,
    attention_mask,
    input_length,
):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------------
    # Prefill
    # Produces generated token #1
    # --------------------------------------------------------

    def prefill_step():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
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

    (past_key_values, next_token), prefill_ms = timed_forward(
        prefill_step
    )

    current_attention_mask = attention_mask

    rows = []

    # --------------------------------------------------------
    # Decode token #2 ... #520
    # --------------------------------------------------------

    for token_index in range(2, OUTPUT_TOKENS + 1):

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
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
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

        (past_key_values, new_token), latency_ms = timed_forward(
            decode_step
        )

        next_token = new_token

        # For generated token k:
        #
        # effective sequence length processed by attention
        # = prompt length + k - 1
        #
        # Example:
        # prompt=128, token_index=386
        # effective_seq_len=513
        effective_seq_len = (
            input_length + token_index - 1
        )

        allocated_mib = to_mib(
            torch.cuda.memory_allocated()
        )

        reserved_mib = to_mib(
            torch.cuda.memory_reserved()
        )

        rows.append(
            {
                "input_tokens": input_length,
                "token_index": token_index,
                "effective_seq_len": effective_seq_len,
                "latency_ms": latency_ms,
                "allocated_mib": allocated_mib,
                "reserved_mib": reserved_mib,
            }
        )

    peak_mib = to_mib(
        torch.cuda.max_memory_allocated()
    )

    del past_key_values
    del next_token

    gc.collect()
    torch.cuda.empty_cache()

    return prefill_ms, peak_mib, rows


# ============================================================
# Main
# ============================================================

def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    torch.manual_seed(0)

    print("=" * 72)
    print("E3B — 512-Token Boundary Verification")
    print("=" * 72)

    print(f"Model:          {MODEL_NAME}")
    print(f"GPU:            {torch.cuda.get_device_name(0)}")
    print(f"PyTorch:        {torch.__version__}")
    print(f"Transformers:   {transformers.__version__}")
    print(f"CUDA runtime:   {torch.version.cuda}")
    print(f"Python:         {platform.python_version()}")

    print(f"Input lengths:  {INPUT_LENGTHS}")
    print(f"Output tokens:  {OUTPUT_TOKENS}")
    print(f"Backend:        SDPA Flash")
    print(f"Warmups:        {WARMUP_RUNS}")
    print(f"Measured runs:  {MEASURED_RUNS}")

    print("=" * 72)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        attn_implementation="sdpa",
    )

    model = model.to(DEVICE)
    model.eval()

    all_rows = []

    # ========================================================
    # Prompt lengths
    # ========================================================

    for input_length in INPUT_LENGTHS:

        print("\n" + "=" * 72)
        print(
            f"INPUT LENGTH: {input_length} tokens"
        )
        print("=" * 72)

        input_ids, attention_mask = build_input(
            tokenizer,
            input_length,
        )

        predicted_boundary_token = (
            513 - input_length + 1
        )

        # Since:
        # effective_seq_len = input_length + token_index - 1
        #
        # effective_seq_len = 513
        # token_index = 514 - input_length

        predicted_boundary_token = (
            514 - input_length
        )

        print(
            "Expected 512→513 crossing around "
            f"generated token #{predicted_boundary_token}"
        )

        # ----------------------------------------------------
        # Warmup
        # ----------------------------------------------------

        for warmup_idx in range(
            1,
            WARMUP_RUNS + 1,
        ):

            print(
                f"Warmup "
                f"{warmup_idx}/{WARMUP_RUNS}"
            )

            run_generation(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_length=input_length,
            )

        # ----------------------------------------------------
        # Measured
        # ----------------------------------------------------

        for run_idx in range(
            1,
            MEASURED_RUNS + 1,
        ):

            print(
                f"Measured run "
                f"{run_idx}/{MEASURED_RUNS}"
            )

            prefill_ms, peak_mib, rows = run_generation(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_length=input_length,
            )

            for row in rows:
                row["run"] = run_idx
                row["prefill_ms"] = prefill_ms
                row["peak_allocated_mib"] = peak_mib

                all_rows.append(row)

    # ========================================================
    # Save raw data
    # ========================================================

    df = pd.DataFrame(all_rows)

    df.to_csv(
        RAW_CSV,
        index=False,
    )

    # ========================================================
    # Median across five measured runs
    # ========================================================

    median_df = (
        df.groupby(
            [
                "input_tokens",
                "token_index",
                "effective_seq_len",
            ],
            as_index=False,
        )
        .agg(
            median_latency_ms=(
                "latency_ms",
                "median",
            ),
            p95_latency_ms=(
                "latency_ms",
                lambda x: np.percentile(x, 95),
            ),
            median_allocated_mib=(
                "allocated_mib",
                "median",
            ),
            median_reserved_mib=(
                "reserved_mib",
                "median",
            ),
        )
    )

    # Focus specifically on sequence lengths around 512.
    boundary_df = median_df[
        median_df["effective_seq_len"].between(
            504,
            520,
        )
    ].copy()

    boundary_df.to_csv(
        BOUNDARY_CSV,
        index=False,
    )

    print("\n" + "=" * 72)
    print("BOUNDARY WINDOW: EFFECTIVE SEQUENCE 504–520")
    print("=" * 72)

    print(
        boundary_df.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(RAW_CSV)
    print(BOUNDARY_CSV)

    print("\nE3B complete.")


if __name__ == "__main__":
    main()