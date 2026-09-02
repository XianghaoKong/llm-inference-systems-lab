import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    print("=" * 60)
    print("LLM Inference Baseline")
    print("=" * 60)

    # ----- Hardware information -----
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    device = torch.device("cuda")

    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(
        f"GPU memory      : "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

    # ----- Load tokenizer -----
    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ----- Load model -----
    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",
    )

    model.eval()

    # ----- Prompt -----
    messages = [
        {
            "role": "user",
            "content": "Explain why GPU parallel computing is useful for deep learning."
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

    print(f"\nInput tokens     : {inputs.input_ids.shape[1]}")

    # ----- Inference -----
    print("\nGenerating...\n")

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    new_tokens = output[0][inputs.input_ids.shape[1]:]

    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    print(response)

    print("\n" + "=" * 60)
    print(f"Generated tokens : {len(new_tokens)}")

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3

    print(f"CUDA allocated   : {allocated:.2f} GB")
    print(f"CUDA reserved    : {reserved:.2f} GB")
    print("=" * 60)


if __name__ == "__main__":
    main()