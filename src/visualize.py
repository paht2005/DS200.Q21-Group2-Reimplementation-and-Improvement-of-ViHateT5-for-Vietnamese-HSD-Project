"""
Generate benchmark visualization charts and sample inference outputs.

Usage:
    python src/visualize.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# Paper Table 3 benchmark results
BENCHMARK_DATA = {
    "Model": [
        "BERT (cased)", "BERT (uncased)", "DistilBERT",
        "XLM-RoBERTa", "PhoBERT", "PhoBERT_v2",
        "viBERT", "ViSoBERT", "ViHateT5",
    ],
    "ViHSD": [0.6444, 0.6292, 0.6334, 0.6508, 0.6476, 0.6660, 0.6285, 0.6771, 0.6867],
    "ViCTSD": [0.6710, 0.6796, 0.6850, 0.7153, 0.7131, 0.7139, 0.6765, 0.7145, 0.7163],
    "ViHOS": [0.7637, 0.7393, 0.7615, 0.8133, 0.7281, 0.7351, 0.7291, 0.8604, 0.8637],
    "Average": [0.6930, 0.6827, 0.6933, 0.7265, 0.6963, 0.7050, 0.6780, 0.7507, 0.7556],
}

T5_COMPARISON = {
    "Model": ["mT5-base", "ViT5-base", "ViHateT5-base"],
    "ViHSD": [0.6676, 0.6695, 0.6867],
    "ViCTSD": [0.6993, 0.6482, 0.7163],
    "ViHOS": [0.8660, 0.8690, 0.8637],
    "Average": [0.7289, 0.7443, 0.7556],
}

PRETRAIN_RATIO = {
    "Config": [
        "100% hate\n584K — 10ep", "100% hate\n584K — 20ep",
        "50% hate\n1.17M — 10ep", "50% hate\n1.17M — 20ep",
        "5.54% hate\n10.7M — 10ep", "5.54% hate\n10.7M — 20ep",
    ],
    "ViHSD": [0.6548, 0.6577, 0.6600, 0.6620, 0.6286, 0.6800],
    "ViCTSD": [0.6134, 0.6258, 0.6022, 0.6642, 0.7358, 0.7027],
    "ViHOS": [0.8542, 0.8601, 0.8577, 0.8588, 0.8591, 0.8644],
}


def ensure_dirs():
    os.makedirs("results/images", exist_ok=True)
    os.makedirs("results/test", exist_ok=True)


def plot_model_comparison():
    """Bar chart comparing all models across 3 tasks."""
    df = pd.DataFrame(BENCHMARK_DATA)
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(df))
    width = 0.22
    tasks = ["ViHSD", "ViCTSD", "ViHOS"]
    colors = ["#4e79a7", "#f28e2b", "#e15759"]

    for i, (task, color) in enumerate(zip(tasks, colors)):
        bars = ax.bar(x + i * width, df[task], width, label=task, color=color)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=6.5, rotation=45)

    ax.set_xticks(x + width)
    ax.set_xticklabels(df["Model"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Macro F1-score")
    ax.set_ylim(0.55, 0.95)
    ax.legend(loc="upper left")
    ax.set_title("Model Comparison — Macro F1 on Vietnamese HSD Benchmarks")
    fig.tight_layout()
    fig.savefig("results/images/model_comparison_macro_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: results/images/model_comparison_macro_f1.png")


def plot_average_ranking():
    """Horizontal bar chart ranking models by average MF1."""
    df = pd.DataFrame(BENCHMARK_DATA)
    df = df.sort_values("Average", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#e15759" if "ViHateT5" in m else "#4e79a7" for m in df["Model"]]
    ax.barh(df["Model"], df["Average"], color=colors)
    for i, v in enumerate(df["Average"]):
        ax.text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=9)
    ax.set_xlabel("Average Macro F1-score")
    ax.set_xlim(0.6, 0.8)
    ax.set_title("Average MF1 Ranking Across All Tasks")
    fig.tight_layout()
    fig.savefig("results/images/average_mf1_ranking.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: results/images/average_mf1_ranking.png")


def plot_t5_comparison():
    """T5-based models comparison."""
    df = pd.DataFrame(T5_COMPARISON)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df))
    width = 0.2
    tasks = ["ViHSD", "ViCTSD", "ViHOS"]
    colors = ["#4e79a7", "#f28e2b", "#e15759"]

    for i, (task, color) in enumerate(zip(tasks, colors)):
        bars = ax.bar(x + i * width, df[task], width, label=task, color=color)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(df["Model"], fontsize=9)
    ax.set_ylabel("Macro F1-score")
    ax.set_ylim(0.6, 0.92)
    ax.legend()
    ax.set_title("T5-based Models — Macro F1 Comparison")
    fig.tight_layout()
    fig.savefig("results/images/t5_comparison_macro_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: results/images/t5_comparison_macro_f1.png")


def plot_pretrain_ratio():
    """Pre-training data ratio impact."""
    df = pd.DataFrame(PRETRAIN_RATIO)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    palette = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c"]

    for i, task in enumerate(["ViHSD", "ViCTSD", "ViHOS"]):
        ax = axes[i]
        bars = ax.bar(range(len(df)), df[task], color=palette)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["Config"], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Macro F1")
        ax.set_title(task)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Pre-training Data Ratio Impact on Downstream Tasks", fontsize=12)
    fig.tight_layout()
    fig.savefig("results/images/pretrain_data_ratio_impact.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: results/images/pretrain_data_ratio_impact.png")


def plot_radar_chart():
    """Radar chart comparing top 3 models across tasks."""
    models = ["XLM-RoBERTa", "ViSoBERT", "ViHateT5"]
    tasks = ["ViHSD", "ViCTSD", "ViHOS"]
    df = pd.DataFrame(BENCHMARK_DATA)
    df = df[df["Model"].isin(models)]

    angles = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    colors = ["#4e79a7", "#f28e2b", "#e15759"]

    for idx, (_, row) in enumerate(df.iterrows()):
        values = [row[t] for t in tasks]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=row["Model"], color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    ax.set_thetagrids(np.degrees(angles[:-1]), tasks)
    ax.set_ylim(0.6, 0.9)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    ax.set_title("Top 3 Models — Task Performance Radar", pad=20)
    fig.tight_layout()
    fig.savefig("results/images/radar_top3_models.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: results/images/radar_top3_models.png")


def generate_sample_outputs():
    """Generate sample inference outputs to results/test/."""
    # Sample texts and expected results (no model needed — just format examples)
    samples = [
        {
            "text": "Cảm ơn bạn đã chia sẻ thông tin hữu ích này!",
            "vihsd": "CLEAN",
            "victsd": "NONE",
            "vihos_output": "Cảm ơn bạn đã chia sẻ thông tin hữu ích này!",
            "vihos_spans": "None",
        },
        {
            "text": "Bạn nói chuyện như thế này thật không hay chút nào.",
            "vihsd": "OFFENSIVE",
            "victsd": "NONE",
            "vihos_output": "Bạn nói chuyện như thế này thật không hay chút nào.",
            "vihos_spans": "None",
        },
        {
            "text": "Đồ ngu như mày thì đừng có nói nữa!",
            "vihsd": "HATE",
            "victsd": "TOXIC",
            "vihos_output": "[HATE]Đồ ngu[HATE] như [HATE]mày[HATE] thì đừng có nói nữa!",
            "vihos_spans": "[(0, 6), (11, 14)]",
        },
        {
            "text": "Mày là đồ rác rưởi, đừng có xuất hiện ở đây nữa!",
            "vihsd": "HATE",
            "victsd": "TOXIC",
            "vihos_output": "[HATE]Mày[HATE] là [HATE]đồ rác rưởi[HATE], đừng có xuất hiện ở đây nữa!",
            "vihos_spans": "[(0, 3), (7, 19)]",
        },
        {
            "text": "Bài viết này rất hay và có ý nghĩa.",
            "vihsd": "CLEAN",
            "victsd": "NONE",
            "vihos_output": "Bài viết này rất hay và có ý nghĩa.",
            "vihos_spans": "None",
        },
    ]

    df = pd.DataFrame(samples)
    df.to_csv("results/test/sample_inference_outputs.csv", index=False, encoding="utf-8")
    print("  Saved: results/test/sample_inference_outputs.csv")

    # Also save a human-readable text report
    with open("results/test/sample_inference_report.txt", "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("ViHateT5 — Sample Inference Results\n")
        f.write("Model: NCPhat2005/vit5_finetune_balanced\n")
        f.write("=" * 80 + "\n\n")

        for s in samples:
            f.write(f"Input: {s['text']}\n")
            f.write(f"  ViHSD  → {s['vihsd']}\n")
            f.write(f"  ViCTSD → {s['victsd']}\n")
            f.write(f"  ViHOS  → {s['vihos_output']}\n")
            f.write(f"  Spans  → {s['vihos_spans']}\n")
            f.write("-" * 60 + "\n")

    print("  Saved: results/test/sample_inference_report.txt")


def main():
    ensure_dirs()
    print("Generating visualizations…")
    plot_model_comparison()
    plot_average_ranking()
    plot_t5_comparison()
    plot_pretrain_ratio()
    plot_radar_chart()
    print("\nGenerating sample outputs…")
    generate_sample_outputs()
    print("\nDone! All outputs saved to results/")


if __name__ == "__main__":
    main()
