import gc
import time
import warnings
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

CONTEXT_LENGTH = 4096
NUM_RUNS = 3
WARMUP_RUNS = 2

RESULT_DIR = Path("results/raw")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


TEST_CONFIGS = [
    {
        "name": "eager_mask",
        "implementation": "eager",
        "sdpa_backend": "none",
        "use_mask": True,
    },
    {
        "name": "sdpa_auto_mask",
        "implementation": "sdpa",
        "sdpa_backend": "auto",
        "use_mask": True,
    },
    {
        "name": "sdpa_auto_nomask",
        "implementation": "sdpa",
        "sdpa_backend": "auto",
        "use_mask": False,
    },
    {
        "name": "sdpa_math_nomask",
        "implementation": "sdpa",
        "sdpa_backend": "math",
        "use_mask": False,
    },
    {
        "name": "sdpa_flash_nomask",
        "implementation": "sdpa",
        "sdpa_backend": "flash",
        "use_mask": False,
    },
    {
        "name": "sdpa_flash_mask",
        "implementation": "sdpa",
        "sdpa_backend": "flash",
        "use_mask": True,
    },
]


def synchronize():
    torch.cuda.synchronize()


def create_input(tokenizer, target_length, device):

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

    return input_ids, attention_mask


def get_backend_context(backend):

    if backend == "auto":
        return nullcontext()

    if backend == "none":
        return nullcontext()

    if backend == "math":
        return sdpa_kernel(
            SDPBackend.MATH
        )

    if backend == "flash":
        return sdpa_kernel(
            SDPBackend.FLASH_ATTENTION
        )

    raise ValueError(
        f"Unknown backend: {backend}"
    )


def build_inputs(
    input_ids,
    attention_mask,
    use_mask,
):

    if use_mask:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    return {
        "input_ids": input_ids,
    }


def load_model(implementation):

    print(
        f"\nLoading model with "
        f"attn_implementation='{implementation}'..."
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation=implementation,
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

    return model


def run_forward(
    model,
    inputs,
    sdpa_backend,
):

    with get_backend_context(
        sdpa_backend
    ):

        with torch.inference_mode():

            output = model(
                **inputs,
                use_cache=True,
            )

    return output


def warmup(
    model,
    inputs,
    sdpa_backend,
):

    for _ in range(WARMUP_RUNS):

        output = run_forward(
            model,
            inputs,
            sdpa_backend,
        )

        del output

    synchronize()


def benchmark_config(
    model,
    config,
    input_ids,
    attention_mask,
):

    inputs = build_inputs(
        input_ids,
        attention_mask,
        config["use_mask"],
    )

    print("\n" + "=" * 75)
    print(f"Test: {config['name']}")
    print("=" * 75)

    print(
        f"Implementation : "
        f"{config['implementation']}"
    )

    print(
        f"SDPA backend   : "
        f"{config['sdpa_backend']}"
    )

    print(
        f"Explicit mask  : "
        f"{config['use_mask']}"
    )

    results = []

    try:

        print("Warm-up...")

        with warnings.catch_warnings(
            record=True
        ) as captured_warnings:

            warnings.simplefilter("always")

            warmup(
                model,
                inputs,
                config["sdpa_backend"],
            )

            for warning in captured_warnings:
                print(
                    "WARNING:",
                    str(warning.message)
                )

        print("Warm-up successful.")

    except Exception as error:

        print(
            "\nCONFIG FAILED DURING WARM-UP"
        )

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        return [
            {
                "config": config["name"],
                "implementation":
                    config["implementation"],
                "sdpa_backend":
                    config["sdpa_backend"],
                "use_mask":
                    config["use_mask"],
                "run": None,
                "prefill_ms": None,
                "peak_memory_gb": None,
                "status": "failed",
                "error": str(error),
            }
        ]

    for run in range(
        1,
        NUM_RUNS + 1
    ):

        gc.collect()

        torch.cuda.reset_peak_memory_stats()

        synchronize()

        try:

            with warnings.catch_warnings(
                record=True
            ) as captured_warnings:

                warnings.simplefilter(
                    "always"
                )

                start = time.perf_counter()

                output = run_forward(
                    model,
                    inputs,
                    config["sdpa_backend"],
                )

                synchronize()

                end = time.perf_counter()

                for warning in captured_warnings:
                    print(
                        "WARNING:",
                        str(warning.message)
                    )

            latency_ms = (
                end - start
            ) * 1000

            peak_memory_gb = (
                torch.cuda.max_memory_allocated()
                / 1024**3
            )

            print(
                f"Run {run}: "
                f"{latency_ms:.2f} ms | "
                f"{peak_memory_gb:.2f} GB"
            )

            results.append(
                {
                    "config": config["name"],
                    "implementation":
                        config["implementation"],
                    "sdpa_backend":
                        config["sdpa_backend"],
                    "use_mask":
                        config["use_mask"],
                    "run": run,
                    "prefill_ms":
                        latency_ms,
                    "peak_memory_gb":
                        peak_memory_gb,
                    "status":
                        "success",
                    "error":
                        "",
                }
            )

            del output

        except Exception as error:

            print(
                f"Run {run} FAILED:"
            )

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

            results.append(
                {
                    "config": config["name"],
                    "implementation":
                        config["implementation"],
                    "sdpa_backend":
                        config["sdpa_backend"],
                    "use_mask":
                        config["use_mask"],
                    "run": run,
                    "prefill_ms": None,
                    "peak_memory_gb": None,
                    "status": "failed",
                    "error": str(error),
                }
            )

    return results


def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU unavailable."
        )

    device = torch.device("cuda")

    print("=" * 75)
    print("SDPA Backend Diagnostic")
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
        f"{CONTEXT_LENGTH}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    input_ids, attention_mask = (
        create_input(
            tokenizer,
            CONTEXT_LENGTH,
            device,
        )
    )

    print(
        f"Actual tokens: "
        f"{input_ids.shape[1]}"
    )

    all_results = []

    # ------------------------------------
    # Eager tests
    # ------------------------------------

    eager_model = load_model(
        "eager"
    )

    eager_configs = [
        config
        for config in TEST_CONFIGS
        if config["implementation"]
        == "eager"
    ]

    for config in eager_configs:

        results = benchmark_config(
            eager_model,
            config,
            input_ids,
            attention_mask,
        )

        all_results.extend(
            results
        )

    del eager_model

    gc.collect()
    torch.cuda.empty_cache()
    synchronize()

    # ------------------------------------
    # SDPA tests
    # ------------------------------------

    sdpa_model = load_model(
        "sdpa"
    )

    sdpa_configs = [
        config
        for config in TEST_CONFIGS
        if config["implementation"]
        == "sdpa"
    ]

    for config in sdpa_configs:

        results = benchmark_config(
            sdpa_model,
            config,
            input_ids,
            attention_mask,
        )

        all_results.extend(
            results
        )

    del sdpa_model

    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------
    # Save raw data
    # ------------------------------------

    df = pd.DataFrame(
        all_results
    )

    raw_path = (
        RESULT_DIR
        / "sdpa_backend_diagnostic_raw.csv"
    )

    df.to_csv(
        raw_path,
        index=False,
    )

    # Only successful measurements
    success_df = df[
        df["status"] == "success"
    ].copy()

    if len(success_df) > 0:

        summary = (
            success_df
            .groupby(
                [
                    "config",
                    "implementation",
                    "sdpa_backend",
                    "use_mask",
                ]
            )
            .agg(
                median_prefill_ms=(
                    "prefill_ms",
                    "median",
                ),
                mean_prefill_ms=(
                    "prefill_ms",
                    "mean",
                ),
                peak_memory_gb=(
                    "peak_memory_gb",
                    "max",
                ),
            )
            .reset_index()
        )

        summary_path = (
            RESULT_DIR
            / "sdpa_backend_diagnostic_summary.csv"
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
            f"\nSummary saved: "
            f"{summary_path}"
        )

    print(
        f"Raw results saved: "
        f"{raw_path}"
    )


if __name__ == "__main__":
    main()