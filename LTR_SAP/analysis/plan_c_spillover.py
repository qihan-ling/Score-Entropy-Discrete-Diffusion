"""
Plan C: Spillover analysis and condition-by-measure interaction figures.

Examines whether SEDD denoising metrics at position i predict human reading
times at positions i+1 and i+2 (spillover), and compares spillover profiles
of denoising metrics vs. surprisal across all ET and SPR measures.

Reads:
  LTR_SAP/analysis/data/sedd_spr_merged.csv
  LTR_SAP/analysis/data/sedd_et_merged.csv

Produces:
  - Spillover correlation heatmaps: metric at position i vs. RT at i+k
  - Condition-by-measure interaction plots
  - Cross-measure comparison tables

Usage:
  python LTR_SAP/analysis/plan_c_spillover.py --output_dir LTR_SAP/analysis/figures/plan_c
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from utils import setup_matplotlib, ET_MEASURES


def compute_spillover_correlations(df, predictor_col, outcome_col, max_lag=3):
    """Compute correlation between predictor at position i and outcome at i+k.

    Args:
        df: DataFrame with 'item', 'word_pos', predictor_col, outcome_col
        predictor_col: name of the predictor column
        outcome_col: name of the outcome column
        max_lag: maximum spillover lag

    Returns:
        list of dicts with lag, r, p, n
    """
    results = []
    for lag in range(0, max_lag + 1):
        if lag == 0:
            valid = df.dropna(subset=[predictor_col, outcome_col])
            if len(valid) < 3:
                continue
            r, p = scipy_stats.pearsonr(valid[predictor_col], valid[outcome_col])
            results.append({"lag": lag, "r": r, "p": p, "n": len(valid)})
        else:
            # Create lagged version
            df_shifted = df.copy()
            df_shifted["outcome_lagged"] = df_shifted.groupby("item")[outcome_col].shift(-lag)
            valid = df_shifted.dropna(subset=[predictor_col, "outcome_lagged"])
            if len(valid) < 3:
                continue
            r, p = scipy_stats.pearsonr(valid[predictor_col], valid["outcome_lagged"])
            results.append({"lag": lag, "r": r, "p": p, "n": len(valid)})
    return results


def main():
    parser = argparse.ArgumentParser(description="Plan C: Spillover analysis")
    parser.add_argument("--output_dir", type=str, default="LTR_SAP/analysis/figures/plan_c")
    parser.add_argument("--max_lag", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plt = setup_matplotlib()

    spr_path = "LTR_SAP/analysis/data/sedd_spr_merged.csv"
    et_path = "LTR_SAP/analysis/data/sedd_et_merged.csv"

    # --- SPR spillover ---
    if os.path.exists(spr_path):
        print("Loading SPR data for spillover analysis...")
        spr = pd.read_csv(spr_path)

        predictor_cols = ["steps_to_commit", "surprisal", "cumulative_kl", "gpt2_surprisal"]
        predictor_cols = [c for c in predictor_cols if c in spr.columns]

        all_corrs = []
        for pred in predictor_cols:
            corrs = compute_spillover_correlations(spr, pred, "RT", max_lag=args.max_lag)
            for c in corrs:
                c["predictor"] = pred
                c["outcome"] = "RT"
            all_corrs.extend(corrs)

        if all_corrs:
            corr_df = pd.DataFrame(all_corrs)
            corr_df.to_csv(os.path.join(args.output_dir, "spr_spillover_correlations.csv"), index=False)

            # Plot spillover profiles
            fig, ax = plt.subplots(figsize=(10, 6))
            for pred in predictor_cols:
                sub = corr_df[corr_df["predictor"] == pred]
                ax.plot(sub["lag"], sub["r"], marker="o", label=pred, linewidth=2)
                # Add significance markers
                for _, row in sub.iterrows():
                    if row["p"] < 0.05:
                        ax.annotate("*", (row["lag"], row["r"]), textcoords="offset points",
                                    xytext=(0, 5), ha="center", fontsize=14)
            ax.set_xlabel("Spillover lag (words)")
            ax.set_ylabel("Pearson r with RT")
            ax.set_title("SPR Spillover Profile: SEDD metrics vs GPT-2 surprisal")
            ax.set_xticks(range(args.max_lag + 1))
            ax.set_xticklabels([f"i+{k}" for k in range(args.max_lag + 1)])
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.legend()
            fig.savefig(os.path.join(args.output_dir, "spr_spillover_profile.png"))
            plt.close(fig)
            print("  Saved spr_spillover_profile.png")

    # --- Eye-tracking spillover ---
    if os.path.exists(et_path):
        print("Loading eye-tracking data for spillover analysis...")
        et = pd.read_csv(et_path)

        predictor_cols = ["steps_to_commit", "surprisal", "cumulative_kl"]
        predictor_cols = [c for c in predictor_cols if c in et.columns]
        et_measures = [m for m in ET_MEASURES if m in et.columns]

        all_et_corrs = []
        for pred in predictor_cols:
            for measure in et_measures:
                corrs = compute_spillover_correlations(et, pred, measure, max_lag=args.max_lag)
                for c in corrs:
                    c["predictor"] = pred
                    c["outcome"] = measure
                all_et_corrs.extend(corrs)

        if all_et_corrs:
            et_corr_df = pd.DataFrame(all_et_corrs)
            et_corr_df.to_csv(os.path.join(args.output_dir, "et_spillover_correlations.csv"), index=False)

            # Heatmap: predictor x (measure, lag)
            fig, axes = plt.subplots(len(predictor_cols), 1,
                                     figsize=(12, 4 * len(predictor_cols)), squeeze=False)
            for i, pred in enumerate(predictor_cols):
                ax = axes[i, 0]
                sub = et_corr_df[et_corr_df["predictor"] == pred]

                if sub.empty:
                    continue

                pivot = sub.pivot_table(index="outcome", columns="lag", values="r")
                im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r",
                               vmin=-0.3, vmax=0.3)
                ax.set_xticks(range(pivot.shape[1]))
                ax.set_xticklabels([f"i+{k}" for k in pivot.columns])
                ax.set_yticks(range(pivot.shape[0]))
                ax.set_yticklabels(pivot.index)
                ax.set_title(f"Spillover: {pred}")
                plt.colorbar(im, ax=ax, label="Pearson r")

                # Annotate with significance
                p_pivot = sub.pivot_table(index="outcome", columns="lag", values="p")
                for row_idx in range(pivot.shape[0]):
                    for col_idx in range(pivot.shape[1]):
                        p_val = p_pivot.values[row_idx, col_idx]
                        if p_val < 0.001:
                            marker = "***"
                        elif p_val < 0.01:
                            marker = "**"
                        elif p_val < 0.05:
                            marker = "*"
                        else:
                            marker = ""
                        if marker:
                            ax.text(col_idx, row_idx, marker, ha="center", va="center",
                                    fontsize=10, fontweight="bold")

            fig.suptitle("Eye-tracking spillover correlations")
            fig.tight_layout()
            fig.savefig(os.path.join(args.output_dir, "et_spillover_heatmap.png"))
            plt.close(fig)
            print("  Saved et_spillover_heatmap.png")

    # --- Cross-measure comparison table ---
    print("\nGenerating cross-measure comparison...")
    if os.path.exists(et_path):
        et = pd.read_csv(et_path)
        et_measures = [m for m in ET_MEASURES if m in et.columns]
        predictors = ["steps_to_commit", "surprisal", "cumulative_kl"]
        predictors = [p for p in predictors if p in et.columns]

        cross_table_rows = []
        for pred in predictors:
            for measure in et_measures:
                valid = et.dropna(subset=[pred, measure])
                if len(valid) < 3:
                    continue
                r, p = scipy_stats.pearsonr(valid[pred], valid[measure])
                cross_table_rows.append({
                    "predictor": pred,
                    "measure": measure,
                    "r": r,
                    "p": p,
                    "n": len(valid),
                })

        if cross_table_rows:
            cross_df = pd.DataFrame(cross_table_rows)
            cross_df.to_csv(os.path.join(args.output_dir, "cross_measure_correlations.csv"), index=False)
            print(f"  Saved cross_measure_correlations.csv ({len(cross_df)} rows)")

            # Pivot for display
            pivot = cross_df.pivot_table(index="measure", columns="predictor", values="r")
            print("\nCorrelation with ET measures (Pearson r):")
            print(pivot.to_string(float_format="%.3f"))

    print("\nDone.")


if __name__ == "__main__":
    main()
