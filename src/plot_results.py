from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_FILE = Path("results/raw/context_length_summary.csv")
FIGURE_DIR = Path("figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.figure(figsize=(7, 4.5))

    plt.plot(
        x,
        y,
        marker="o",
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = FIGURE_DIR / filename

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {path}")


def main():

    df = pd.read_csv(INPUT_FILE)

    # --------------------------
    # Context vs TTFT
    # --------------------------

    save_plot(
        df["context_length"],
        df["median_ttft_ms"],
        "Input Context Length (tokens)",
        "Median TTFT (ms)",
        "Context Length vs Time to First Token",
        "context_vs_ttft.png",
    )

    # --------------------------
    # Context vs Throughput
    # --------------------------

    save_plot(
        df["context_length"],
        df["median_throughput_tok_s"],
        "Input Context Length (tokens)",
        "Median Throughput (tokens/s)",
        "Context Length vs End-to-End Throughput",
        "context_vs_throughput.png",
    )

    # --------------------------
    # Context vs GPU Memory
    # --------------------------

    save_plot(
        df["context_length"],
        df["peak_memory_gb"],
        "Input Context Length (tokens)",
        "Peak GPU Memory (GB)",
        "Context Length vs Peak GPU Memory",
        "context_vs_memory.png",
    )


if __name__ == "__main__":
    main()