from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

RAW_DIR = SCRIPT_DIR / "results" / "raw"
FIGURE_DIR = SCRIPT_DIR / "results" / "figures" / "e3"

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_CSV = RAW_DIR / "decode_scaling_summary.csv"
TOKEN_CSV = RAW_DIR / "decode_token_latency.csv"


# ============================================================
# Load data
# ============================================================

summary = pd.read_csv(SUMMARY_CSV)
tokens = pd.read_csv(TOKEN_CSV)

print("Loaded summary:")
print(summary)

print("\nToken latency rows:")
print(len(tokens))


# ============================================================
# Figure 1
# Output Length vs End-to-End Latency
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

for backend in ["eager", "flash"]:

    subset = (
        summary[
            summary["backend"] == backend
        ]
        .sort_values("output_tokens")
    )

    ax.plot(
        subset["output_tokens"],
        subset["e2e_median_ms"] / 1000.0,
        marker="o",
        linewidth=2,
        label=backend.capitalize(),
    )

ax.set_xlabel("Generated Output Length (tokens)")
ax.set_ylabel("Median End-to-End Latency (s)")
ax.set_title(
    "E3: Output Length vs End-to-End Generation Latency"
)

ax.set_xticks(
    sorted(summary["output_tokens"].unique())
)

ax.grid(
    True,
    alpha=0.3,
)

ax.legend()

fig.tight_layout()

output_path = (
    FIGURE_DIR
    / "decode_output_vs_total_latency.png"
)

fig.savefig(
    output_path,
    dpi=300,
)

plt.close(fig)

print(f"Saved: {output_path}")


# ============================================================
# Figure 2
# Output Length vs Decode Throughput
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

for backend in ["eager", "flash"]:

    subset = (
        summary[
            summary["backend"] == backend
        ]
        .sort_values("output_tokens")
    )

    ax.plot(
        subset["output_tokens"],
        subset["decode_tok_s_median"],
        marker="o",
        linewidth=2,
        label=backend.capitalize(),
    )

ax.set_xlabel("Generated Output Length (tokens)")
ax.set_ylabel("Median Decode Throughput (tokens/s)")
ax.set_title(
    "E3: Output Length vs Decode Throughput"
)

ax.set_xticks(
    sorted(summary["output_tokens"].unique())
)

ax.grid(
    True,
    alpha=0.3,
)

ax.legend()

fig.tight_layout()

output_path = (
    FIGURE_DIR
    / "decode_output_vs_throughput.png"
)

fig.savefig(
    output_path,
    dpi=300,
)

plt.close(fig)

print(f"Saved: {output_path}")


# ============================================================
# Figure 3
# Token Index vs Per-Token Decode Latency
#
# Use the 512-token workload because it gives the longest
# continuous decode sequence.
#
# Median across the 5 measured runs is calculated for each
# generated-token position.
# ============================================================

LONG_OUTPUT = 512

long_tokens = tokens[
    tokens["output_tokens"] == LONG_OUTPUT
]

median_by_token = (
    long_tokens
    .groupby(
        ["backend", "token_index"],
        as_index=False,
    )["latency_ms"]
    .median()
)

fig, ax = plt.subplots(figsize=(9, 5))

for backend in ["eager", "flash"]:

    subset = (
        median_by_token[
            median_by_token["backend"] == backend
        ]
        .sort_values("token_index")
    )

    ax.plot(
        subset["token_index"],
        subset["latency_ms"],
        linewidth=1.3,
        alpha=0.85,
        label=backend.capitalize(),
    )

ax.set_xlabel("Generated Token Index")
ax.set_ylabel("Median Decode Latency (ms/token)")
ax.set_title(
    "E3: Per-Token Decode Latency Across a 512-Token Generation"
)

ax.set_xlim(
    median_by_token["token_index"].min(),
    median_by_token["token_index"].max(),
)

ax.grid(
    True,
    alpha=0.3,
)

ax.legend()

fig.tight_layout()

output_path = (
    FIGURE_DIR
    / "decode_token_index_vs_latency.png"
)

fig.savefig(
    output_path,
    dpi=300,
)

plt.close(fig)

print(f"Saved: {output_path}")


# ============================================================
# Complete
# ============================================================

print("\nE3 figures complete.")
print(f"Figure directory: {FIGURE_DIR}")