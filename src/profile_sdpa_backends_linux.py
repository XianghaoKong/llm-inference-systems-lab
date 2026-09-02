import gc
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import profile, ProfilerActivity, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CONTEXT_LENGTH = 4096

BACKENDS = {
    "math": SDPBackend.MATH,
    "flash": SDPBackend.FLASH_ATTENTION,
}

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "profiler"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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

    return {
        "input_ids": input_ids,
    }


def warmup(model, inputs, backend):

    with sdpa_kernel(backend):
        with torch.inference_mode():

            for _ in range(2):
                output = model(
                    **inputs,
                    use_cache=True,
                )

                del output

    synchronize()


def run_profile(
    model,
    inputs,
    backend_name,
    backend,
):

    print("\n" + "=" * 75)
    print(f"Profiling backend: {backend_name}")
    print("=" * 75)

    warmup(
        model,
        inputs,
        backend,
    )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    synchronize()

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
            f"prefill_4096_{backend_name}"
        ):

            with sdpa_kernel(
                backend
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
            row_limit=50,
        )
    )

    print(table)

    table_path = (
        OUTPUT_DIR
        / f"prefill_4096_{backend_name}_linux_operators.txt"
    )

    with open(
        table_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(table)

    trace_path = (
        OUTPUT_DIR
        / f"prefill_4096_{backend_name}_linux_trace.json"
    )

    prof.export_chrome_trace(
        str(trace_path)
    )

    peak_memory = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    print(
        f"\nPeak allocated memory: "
        f"{peak_memory:.3f} GB"
    )

    print(
        f"Saved operator table: "
        f"{table_path}"
    )

    print(
        f"Saved trace: "
        f"{trace_path}"
    )

    del output

    gc.collect()
    torch.cuda.empty_cache()


def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU unavailable."
        )

    device = torch.device("cuda")

    print("=" * 75)
    print("SDPA Math vs Flash Profiler")
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

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    inputs = create_input(
        tokenizer,
        CONTEXT_LENGTH,
        device,
    )

    print(
        f"Input tokens : "
        f"{inputs['input_ids'].shape[1]}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="sdpa",
    )

    model.eval()

    print(
        "Configured attention:",
        getattr(
            model.config,
            "_attn_implementation",
            "unknown",
        ),
    )

    for backend_name, backend in BACKENDS.items():

        run_profile(
            model,
            inputs,
            backend_name,
            backend,
        )


if __name__ == "__main__":
    main()