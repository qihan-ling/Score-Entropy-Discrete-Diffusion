"""
Plan A: Steps-to-commit vs. Surprisal scatter, residual analysis, KL divergence exploration.

Loads SEDD enforce-prefix outputs from LTR_SAP/ and GPT-2 surprisals from sapbenchmark.
Produces:
  1. Scatter plot: steps_to_commit (z-scored) vs. surprisal (z-scored) across filler items
  2. Residual analysis: top/bottom decile residuals classified by word type
  3. KL divergence panels: cumulative KL vs. surprisal, step-wise KL trajectories

Usage:
  python LTR_SAP/analysis/plan_a_scatter.py --output_dir LTR_SAP/analysis/figures/plan_a
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy import stats

from utils import (
    load_all_outputs, extract_word_metrics, load_gpt2_surprisals,
    setup_matplotlib, LTR_SAP_DIR,
)


def build_word_level_table(subset):
    """Build a DataFrame of word-level metrics across all items in a subset."""
    outputs = load_all_outputs(subset)
    if not outputs:
        print(f"  No outputs found for {subset}")
        return pd.DataFrame()

    rows = []
    for out in outputs:
        item = out.get("tokenization", {}).get("sentence", "")
        wm = extract_word_metrics(out)
        wm["sentence"] = item
        wm["item"] = out["commitment_log"][0]["position"] if out["commitment_log"] else None
        rows.append(wm)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Plan A: Steps vs Surprisal analysis")
    parser.add_argument("--output_dir", type=str, default="LTR_SAP/analysis/figures/plan_a")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plt = setup_matplotlib()

    # --- 1. Load SEDD word-level metrics for filler items ---
    print("Loading SEDD filler outputs...")
    filler_df = build_word_level_table("filler")

    if filler_df.empty:
        print("No filler outputs found. Run batch_runner.py first.")
        return

    # --- 2. Load all subsets for a combined view ---
    all_dfs = []
    for subset in ["Agreement", "ClassicGP", "RelativeClause", "AttachmentAmbiguity", "filler"]:
        df = build_word_level_table(subset)
        if not df.empty:
            df["subset"] = subset
            all_dfs.append(df)

    if not all_dfs:
        print("No outputs found. Run batch_runner.py first.")
        return
    combined = pd.concat(all_dfs, ignore_index=True)

    # --- 3. Z-score both metrics ---
    combined = combined.dropna(subset=["steps_to_commit", "surprisal"])
    combined["steps_z"] = stats.zscore(combined["steps_to_commit"])
    combined["surprisal_z"] = stats.zscore(combined["surprisal"])

    # --- Figure 1: Scatter of steps vs surprisal ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(combined["surprisal_z"], combined["steps_z"],
               alpha=0.3, s=10, c="#2196F3", edgecolors="none")

    r, p = stats.pearsonr(combined["surprisal_z"], combined["steps_z"])
    rho, p_rho = stats.spearmanr(combined["surprisal_z"], combined["steps_z"])

    slope, intercept = np.polyfit(combined["surprisal_z"], combined["steps_z"], 1)
    x_line = np.linspace(combined["surprisal_z"].min(), combined["surprisal_z"].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=1.5)

    ax.set_xlabel("Surprisal (z-scored)")
    ax.set_ylabel("Steps-to-commit (z-scored)")
    ax.set_title(f"Steps-to-commit vs Surprisal\nr={r:.3f} (p={p:.2e}), rho={rho:.3f}")
    ax.set_aspect("equal", adjustable="datalim")
    fig.savefig(os.path.join(args.output_dir, "scatter_steps_vs_surprisal.png"))
    plt.close(fig)
    print(f"  Saved scatter_steps_vs_surprisal.png (r={r:.3f}, rho={rho:.3f})")

    # --- Figure 2: Residual analysis ---
    combined["residual"] = combined["steps_z"] - (slope * combined["surprisal_z"] + intercept)
    n_decile = max(1, len(combined) // 10)

    sorted_df = combined.sort_values("residual")
    bottom = sorted_df.head(n_decile).copy()
    top = sorted_df.tail(n_decile).copy()
    bottom["decile"] = "Low steps (given surprisal)"
    top["decile"] = "High steps (given surprisal)"
    residual_df = pd.concat([bottom, top])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, (label, group) in enumerate(residual_df.groupby("decile")):
        ax = axes[i]
        if "word" in group.columns:
            word_counts = group["word"].value_counts().head(15)
            word_counts.plot.barh(ax=ax)
        ax.set_title(label)
        ax.set_xlabel("Count")
    fig.suptitle("Top/Bottom Decile Residuals: Most Common Words")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "residual_word_types.png"))
    plt.close(fig)
    print("  Saved residual_word_types.png")

    # --- Figure 3: Cumulative KL vs Surprisal ---
    combined_kl = combined.dropna(subset=["cumulative_kl", "surprisal"])
    if len(combined_kl) > 10:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(combined_kl["surprisal"], combined_kl["cumulative_kl"],
                   alpha=0.3, s=10, c="#FF9800", edgecolors="none")

        r_kl, p_kl = stats.pearsonr(combined_kl["surprisal"], combined_kl["cumulative_kl"])
        ax.set_xlabel("Surprisal (bits)")
        ax.set_ylabel("Cumulative KL (bits)")
        ax.set_title(f"Cumulative KL vs Surprisal (r={r_kl:.3f})")
        fig.savefig(os.path.join(args.output_dir, "cumkl_vs_surprisal.png"))
        plt.close(fig)
        print(f"  Saved cumkl_vs_surprisal.png (r={r_kl:.3f})")

    # --- Save combined metrics CSV ---
    out_csv = os.path.join(args.output_dir, "word_level_metrics.csv")
    combined.to_csv(out_csv, index=False)
    print(f"  Saved word_level_metrics.csv ({len(combined)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
