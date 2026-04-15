"""
Strict-LTR factor decomposition analysis.

Gate dsigma is constant (--steps 1024). Investigate the two variable factors:
  1. Score sharpness: proxied by final_entropy
  2. Context quality: proxied by position (linear increase in enforce-prefix LTR)

For each subset, correlate final_entropy and position with steps_taken
(and weighted_steps) to characterize their relationships.

Usage:
  python LTR_SAP/analysis/strict_ltr_decompose.py --output_dir LTR_SAP/analysis/results/strict_ltr
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

from utils import (
    get_sap_files, get_subset_name, get_critical_pos_col,
    load_sap_csv, load_all_outputs, compute_weighted_steps,
    setup_matplotlib, condition_palette,
)


def collect_token_level_data(subset, total_steps=1024):
    """Collect token-level data (not word-aggregated) for factor analysis."""
    csv_files = get_sap_files()
    csv_path = None
    for f in csv_files:
        if get_subset_name(f) == subset:
            csv_path = f
            break
    if csv_path is None:
        return pd.DataFrame()

    df_stim = load_sap_csv(csv_path)
    crit_col = get_critical_pos_col(csv_path)
    cond_col = "condition" if "condition" in df_stim.columns else None

    all_rows = []
    conditions = df_stim[cond_col].unique() if cond_col else [None]

    for condition in conditions:
        outputs = load_all_outputs(subset, condition)
        for output in outputs:
            sentence = output["tokenization"]["sentence"]
            if cond_col:
                matching = df_stim[(df_stim[cond_col] == condition) & (df_stim["Sentence"] == sentence)]
            else:
                matching = df_stim[df_stim["Sentence"] == sentence]
            if matching.empty:
                continue

            item_row = matching.iloc[0]
            item_id = item_row.get("item", None)
            crit_pos = int(item_row[crit_col]) if crit_col else None

            entries = compute_weighted_steps(output["commitment_log"], total_steps)
            for entry in entries:
                row = {
                    "item": item_id,
                    "condition": condition,
                    "position": entry["position"],
                    "steps_taken": entry["steps_taken"],
                    "weighted_steps": entry["weighted_steps"],
                    "t_commitment": entry["t_commitment"],
                    "step": entry["step"],
                    "final_entropy": entry.get("final_entropy"),
                    "final_surprisal": entry.get("final_surprisal"),
                    "cumulative_kl": entry.get("cumulative_kl"),
                    "committed_token": entry.get("committed_token"),
                    "correct": entry.get("correct"),
                }
                if crit_pos is not None:
                    row["critical_pos"] = crit_pos
                    row["relative_pos"] = entry["position"] - crit_pos
                all_rows.append(row)

    return pd.DataFrame(all_rows)


def factor_correlations(df, subset, output_dir):
    """Correlate factors with steps_taken and weighted_steps."""
    plt = setup_matplotlib()

    print(f"\n  === Factor correlations ({subset}) ===")

    # Overall correlations
    valid = df.dropna(subset=["steps_taken", "final_entropy", "position"])
    if valid.empty:
        print("  No valid data")
        return

    results = []
    for target in ["steps_taken", "weighted_steps"]:
        for factor in ["final_entropy", "position"]:
            clean = valid[[target, factor]].dropna()
            if len(clean) < 5:
                continue
            rho, p = stats.spearmanr(clean[target], clean[factor])
            r_pearson, p_pearson = stats.pearsonr(clean[target], clean[factor])
            print(f"    {target} vs {factor}: Spearman rho={rho:.3f} (p={p:.4f}), "
                  f"Pearson r={r_pearson:.3f} (p={p_pearson:.4f}), n={len(clean)}")
            results.append({
                "subset": subset, "target": target, "factor": factor,
                "spearman_rho": rho, "p_spearman": p,
                "pearson_r": r_pearson, "p_pearson": p_pearson,
                "n": len(clean),
            })

    if results:
        pd.DataFrame(results).to_csv(output_dir / f"{subset}_factor_correlations.csv", index=False)

    # Scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for col, target in enumerate(["steps_taken", "weighted_steps"]):
        for row, factor in enumerate(["final_entropy", "position"]):
            ax = axes[row, col]
            clean = valid[[target, factor]].dropna()
            ax.scatter(clean[factor], clean[target], alpha=0.3, s=10)
            ax.set_xlabel(factor)
            ax.set_ylabel(target)
            rho, _ = stats.spearmanr(clean[target], clean[factor])
            ax.set_title(f"{target} vs {factor} (rho={rho:.3f})")
    plt.suptitle(f"Factor Decomposition: {subset}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{subset}_factor_scatter.png")
    plt.close()


def factor_by_condition(df, subset, output_dir):
    """Compare factors between conditions at the critical position."""
    if "condition" not in df.columns or "relative_pos" not in df.columns:
        return

    plt = setup_matplotlib()
    conditions = sorted(df["condition"].dropna().unique())
    if len(conditions) < 2:
        return

    crit = df[df["relative_pos"] == 0].copy()
    if crit.empty:
        return

    print(f"\n  === Factor comparison at critical position ({subset}) ===")

    factors = ["steps_taken", "weighted_steps", "final_entropy", "t_commitment"]
    available = [f for f in factors if f in crit.columns]

    results = []
    for factor in available:
        for cond in conditions:
            vals = crit[crit["condition"] == cond][factor].dropna()
            print(f"    {factor} | {cond}: mean={vals.mean():.3f}, sd={vals.std():.3f}")

        groups = [crit[crit["condition"] == c][factor].dropna() for c in conditions]
        if len(groups) == 2 and all(len(g) > 1 for g in groups):
            t_stat, p_val = stats.ttest_ind(groups[0], groups[1])
            diff = groups[0].mean() - groups[1].mean()
            print(f"    -> diff={diff:.3f}, t={t_stat:.3f}, p={p_val:.4f}")
            results.append({
                "subset": subset, "factor": factor,
                "cond_a": conditions[0], "cond_b": conditions[1],
                "mean_a": groups[0].mean(), "mean_b": groups[1].mean(),
                "diff": diff, "t_stat": t_stat, "p_value": p_val,
            })

    if results:
        pd.DataFrame(results).to_csv(output_dir / f"{subset}_factor_by_condition.csv", index=False)

    # Entropy-to-steps ratio analysis
    crit_valid = crit.dropna(subset=["final_entropy", "steps_taken"])
    if len(crit_valid) > 0 and crit_valid["steps_taken"].max() > 0:
        crit_valid = crit_valid.copy()
        crit_valid["entropy_steps_ratio"] = crit_valid["final_entropy"] / crit_valid["steps_taken"].clip(lower=1)
        print(f"\n  === Entropy-to-steps ratio at critical position ===")
        for cond in conditions:
            vals = crit_valid[crit_valid["condition"] == cond]["entropy_steps_ratio"].dropna()
            if len(vals) > 0:
                print(f"    {cond}: mean={vals.mean():.4f}, sd={vals.std():.4f}")


def position_profile(df, subset, output_dir):
    """Plot how steps_taken and entropy vary with position."""
    plt = setup_matplotlib()
    palette = condition_palette()

    if "condition" not in df.columns:
        return

    conditions = sorted(df["condition"].dropna().unique())
    if "relative_pos" not in df.columns:
        return

    window = df[(df["relative_pos"] >= -3) & (df["relative_pos"] <= 3)]
    if window.empty:
        return

    metrics = ["steps_taken", "weighted_steps", "final_entropy"]
    available = [m for m in metrics if m in window.columns]

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        for cond in conditions:
            cond_data = window[window["condition"] == cond]
            means = cond_data.groupby("relative_pos")[metric].mean()
            sems = cond_data.groupby("relative_pos")[metric].sem()
            color = palette.get(cond, None)
            ax.errorbar(means.index, means.values, yerr=sems.values,
                        label=cond, marker="o", color=color, capsize=3)
        ax.set_xlabel("Position relative to critical")
        ax.set_ylabel(metric)
        ax.legend()
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    plt.suptitle(f"Position Profile: {subset}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{subset}_position_profile.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Strict-LTR factor decomposition")
    parser.add_argument("--output_dir", type=str, default="LTR_SAP/analysis/results/strict_ltr")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--steps", type=int, default=1024)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subsets = ["Agreement", "ClassicGP", "RelativeClause", "AttachmentAmbiguity"]
    if args.subset:
        subsets = [args.subset]

    for subset in subsets:
        print(f"\n{'='*70}")
        print(f"Factor Decomposition: {subset}")
        print(f"{'='*70}")

        df = collect_token_level_data(subset, total_steps=args.steps)
        if df.empty:
            print(f"  No data found for {subset}")
            continue

        print(f"  Loaded {len(df)} token-level entries")

        factor_correlations(df, subset, output_dir)
        factor_by_condition(df, subset, output_dir)
        position_profile(df, subset, output_dir)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
