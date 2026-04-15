"""
Cross-experiment comparison scripts.

Compares:
  1. Strict-LTR vs critical-position metrics at the same critical positions
  2. Soft-context and Monte Carlo vs strict-LTR
  3. Condition effects across all experiment types

Reads from LTR_SAP/, LTR_SAP_critical/, and context comparison results.

Usage:
  python LTR_SAP_comparison/compare_experiments.py \
      --output_dir LTR_SAP_comparison/results
"""

import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "LTR_SAP", "analysis"))

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

from utils import (
    get_sap_files, get_subset_name, get_critical_pos_col, load_sap_csv,
    load_all_outputs, extract_word_metrics, compute_weighted_steps,
    load_critical_outputs_by_offset,
    setup_matplotlib, condition_palette,
    LTR_SAP_DIR, LTR_SAP_CRITICAL_DIR,
)


def collect_strict_ltr_at_critical(subset, total_steps=1024):
    """Collect strict-LTR metrics at critical positions for comparison."""
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

    if crit_col is None:
        return pd.DataFrame()

    rows = []
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
            item_id = item_row.get("item")
            crit_pos = int(item_row[crit_col])

            entries = compute_weighted_steps(output["commitment_log"], total_steps)

            # Find entries at and around critical position
            for entry in entries:
                token_pos = entry["position"]
                word_pos = token_pos - 1  # subtract <|endoftext|>
                offset = word_pos - (crit_pos - 1)

                if -2 <= offset <= 3:
                    rows.append({
                        "item": item_id,
                        "condition": condition,
                        "offset": offset,
                        "word_position": word_pos + 1,  # 1-indexed
                        "steps_taken": entry["steps_taken"],
                        "weighted_steps": entry["weighted_steps"],
                        "t_commitment": entry["t_commitment"],
                        "final_entropy": entry.get("final_entropy"),
                        "final_surprisal": entry.get("final_surprisal"),
                        "cumulative_kl": entry.get("cumulative_kl"),
                        "correct": entry.get("correct"),
                        "experiment": "strict_ltr",
                    })

    return pd.DataFrame(rows)


def collect_critical_position_metrics(subset):
    """Collect critical-position metrics for comparison."""
    csv_files = get_sap_files()
    csv_path = None
    for f in csv_files:
        if get_subset_name(f) == subset:
            csv_path = f
            break
    if csv_path is None:
        return pd.DataFrame()

    df_stim = load_sap_csv(csv_path)
    cond_col = "condition" if "condition" in df_stim.columns else None
    conditions = df_stim[cond_col].unique() if cond_col else [None]

    rows = []
    for condition in conditions:
        by_offset = load_critical_outputs_by_offset(subset, condition)
        for offset, outputs in by_offset.items():
            for output in outputs:
                entry = output.get("commitment_log", {})
                if not entry:
                    continue
                tok = output.get("tokenization", {})
                sentence = tok.get("sentence", "")

                if cond_col:
                    matching = df_stim[(df_stim[cond_col] == condition) & (df_stim["Sentence"] == sentence)]
                else:
                    matching = df_stim[df_stim["Sentence"] == sentence]
                item_id = matching.iloc[0].get("item") if not matching.empty else None

                rows.append({
                    "item": item_id,
                    "condition": condition,
                    "offset": offset,
                    "word_position": entry.get("word_position"),
                    "steps_taken": entry.get("steps_taken"),
                    "t_commitment": entry.get("t_commitment"),
                    "final_entropy": entry.get("final_entropy"),
                    "final_surprisal": entry.get("final_surprisal"),
                    "cumulative_kl": entry.get("cumulative_kl"),
                    "correct": entry.get("correct"),
                    "experiment": "critical_position",
                })

    return pd.DataFrame(rows)


def comparison_1_strict_vs_critical(output_dir):
    """Compare strict-LTR and critical-position metrics at same positions."""
    plt = setup_matplotlib()
    palette = condition_palette()

    print(f"\n{'='*60}")
    print("Comparison 1: Strict-LTR vs Critical-Position")
    print(f"{'='*60}")

    subsets = ["Agreement", "ClassicGP", "RelativeClause", "AttachmentAmbiguity"]
    all_comparisons = []

    for subset in subsets:
        strict = collect_strict_ltr_at_critical(subset)
        critical = collect_critical_position_metrics(subset)

        if strict.empty or critical.empty:
            print(f"  {subset}: missing data (strict={len(strict)}, critical={len(critical)})")
            continue

        # Merge on item, condition, offset
        merged = strict.merge(
            critical, on=["item", "condition", "offset"],
            suffixes=("_strict", "_critical"),
            how="inner",
        )

        if merged.empty:
            continue

        print(f"\n  {subset}: {len(merged)} matched entries")

        # Correlation of steps_taken between experiments
        for offset in sorted(merged["offset"].unique()):
            odata = merged[merged["offset"] == offset]
            clean = odata[["steps_taken_strict", "steps_taken_critical"]].dropna()
            if len(clean) >= 5:
                rho, p = stats.spearmanr(clean["steps_taken_strict"], clean["steps_taken_critical"])
                print(f"    offset {offset:+d}: steps correlation rho={rho:.3f}, p={p:.4f}, n={len(clean)}")

        # Condition effects comparison
        conditions = sorted(merged["condition"].dropna().unique())
        if len(conditions) >= 2:
            for exp in ["strict", "critical"]:
                col = f"steps_taken_{exp}"
                crit_data = merged[merged["offset"] == 0]
                groups = [crit_data[crit_data["condition"] == c][col].dropna() for c in conditions]
                if all(len(g) > 1 for g in groups):
                    diff = groups[0].mean() - groups[1].mean()
                    t_stat, p_val = stats.ttest_ind(groups[0], groups[1])
                    print(f"    {exp} condition effect: diff={diff:.3f}, t={t_stat:.3f}, p={p_val:.4f}")

        merged["subset"] = subset
        all_comparisons.append(merged)

    if all_comparisons:
        combined = pd.concat(all_comparisons, ignore_index=True)
        combined.to_csv(output_dir / "strict_vs_critical_comparison.csv", index=False)
        print(f"\n  Saved strict_vs_critical_comparison.csv ({len(combined)} rows)")

        # Scatter: strict vs critical steps at offset=0
        crit_data = combined[combined["offset"] == 0]
        if len(crit_data) > 5:
            fig, ax = plt.subplots(figsize=(8, 8))
            for subset in crit_data["subset"].unique():
                sub = crit_data[crit_data["subset"] == subset]
                ax.scatter(sub["steps_taken_strict"], sub["steps_taken_critical"],
                           alpha=0.5, s=20, label=subset)
            lim = max(crit_data["steps_taken_strict"].max(),
                      crit_data["steps_taken_critical"].max()) * 1.1
            ax.plot([0, lim], [0, lim], "k--", alpha=0.3)
            ax.set_xlabel("Steps (strict-LTR)")
            ax.set_ylabel("Steps (critical-position)")
            ax.set_title("Strict-LTR vs Critical-Position at Critical Word")
            ax.legend()
            fig.savefig(output_dir / "strict_vs_critical_scatter.png")
            plt.close(fig)


def comparison_2_context_modes(output_dir):
    """Summarize context mode comparison results if available."""
    print(f"\n{'='*60}")
    print("Comparison 2: Context Modes Summary")
    print(f"{'='*60}")

    context_results_dir = Path("LTR_SAP/analysis/results/context_comparison")
    if not context_results_dir.exists():
        print("  No context comparison results found. Run compare_context_modes.py first.")
        return

    for csv_file in sorted(context_results_dir.glob("*_context_comparison.csv")):
        subset = csv_file.stem.replace("_context_comparison", "")
        df = pd.read_csv(csv_file)
        print(f"\n  {subset}:")

        modes = ["hard", "soft", "montecarlo", "renoise", "masked"]
        for mode in modes:
            col = f"steps_{mode}"
            if col in df.columns:
                valid = df[col].dropna()
                print(f"    {mode:12s}: mean={valid.mean():.1f}, median={valid.median():.1f}, n={len(valid)}")

    # Copy Wilcoxon test results
    for csv_file in sorted(context_results_dir.glob("*_wilcoxon_tests.csv")):
        df = pd.read_csv(csv_file)
        df.to_csv(output_dir / csv_file.name, index=False)
        print(f"\n  Copied {csv_file.name}")


def comparison_3_condition_effects_across_experiments(output_dir):
    """Compare condition effects across all experiment types."""
    plt = setup_matplotlib()

    print(f"\n{'='*60}")
    print("Comparison 3: Condition Effects Across Experiments")
    print(f"{'='*60}")

    subsets = ["Agreement", "ClassicGP", "RelativeClause", "AttachmentAmbiguity"]

    all_effects = []
    for subset in subsets:
        # Strict-LTR effects
        strict = collect_strict_ltr_at_critical(subset)
        if not strict.empty:
            conditions = sorted(strict["condition"].dropna().unique())
            if len(conditions) >= 2:
                crit = strict[strict["offset"] == 0]
                for metric in ["steps_taken", "weighted_steps", "final_entropy"]:
                    if metric not in crit.columns:
                        continue
                    groups = [crit[crit["condition"] == c][metric].dropna() for c in conditions]
                    if all(len(g) > 1 for g in groups):
                        all_effects.append({
                            "subset": subset,
                            "experiment": "strict_ltr",
                            "metric": metric,
                            "cond_a": conditions[0],
                            "cond_b": conditions[1],
                            "mean_a": groups[0].mean(),
                            "mean_b": groups[1].mean(),
                            "diff": groups[0].mean() - groups[1].mean(),
                            "cohens_d": (groups[0].mean() - groups[1].mean()) / np.sqrt(
                                (groups[0].var() + groups[1].var()) / 2
                            ) if groups[0].var() + groups[1].var() > 0 else 0,
                        })

        # Critical-position effects
        critical = collect_critical_position_metrics(subset)
        if not critical.empty:
            conditions = sorted(critical["condition"].dropna().unique())
            if len(conditions) >= 2:
                crit = critical[critical["offset"] == 0]
                for metric in ["steps_taken", "final_entropy"]:
                    if metric not in crit.columns:
                        continue
                    groups = [crit[crit["condition"] == c][metric].dropna() for c in conditions]
                    if all(len(g) > 1 for g in groups):
                        all_effects.append({
                            "subset": subset,
                            "experiment": "critical_position",
                            "metric": metric,
                            "cond_a": conditions[0],
                            "cond_b": conditions[1],
                            "mean_a": groups[0].mean(),
                            "mean_b": groups[1].mean(),
                            "diff": groups[0].mean() - groups[1].mean(),
                            "cohens_d": (groups[0].mean() - groups[1].mean()) / np.sqrt(
                                (groups[0].var() + groups[1].var()) / 2
                            ) if groups[0].var() + groups[1].var() > 0 else 0,
                        })

    if all_effects:
        effects_df = pd.DataFrame(all_effects)
        effects_df.to_csv(output_dir / "condition_effects_all_experiments.csv", index=False)
        print(f"\n  Saved condition_effects_all_experiments.csv ({len(effects_df)} rows)")

        # Print summary
        for subset in effects_df["subset"].unique():
            print(f"\n  {subset}:")
            sub = effects_df[effects_df["subset"] == subset]
            for _, row in sub.iterrows():
                print(f"    {row['experiment']:20s} | {row['metric']:15s} | "
                      f"diff={row['diff']:+.3f} | d={row['cohens_d']:+.3f}")

        # Grouped bar chart of Cohen's d
        fig, ax = plt.subplots(figsize=(14, 6))
        experiments = effects_df["experiment"].unique()
        metrics = effects_df["metric"].unique()
        subsets_list = effects_df["subset"].unique()

        x = np.arange(len(subsets_list))
        width = 0.15
        offset = 0

        for exp in experiments:
            for metric in metrics:
                mask = (effects_df["experiment"] == exp) & (effects_df["metric"] == metric)
                sub = effects_df[mask]
                if sub.empty:
                    continue
                vals = []
                for s in subsets_list:
                    row = sub[sub["subset"] == s]
                    vals.append(row["cohens_d"].values[0] if not row.empty else 0)
                ax.bar(x + offset * width, vals, width,
                       label=f"{exp}/{metric}", alpha=0.8)
                offset += 1

        ax.set_xticks(x + width * (offset - 1) / 2)
        ax.set_xticklabels(subsets_list, rotation=30, ha="right")
        ax.set_ylabel("Cohen's d")
        ax.set_title("Condition Effect Sizes Across Experiments")
        ax.legend(fontsize=7, ncol=2)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(output_dir / "condition_effects_comparison.png")
        plt.close(fig)
        print("  Saved condition_effects_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Cross-experiment comparison")
    parser.add_argument("--output_dir", type=str, default="LTR_SAP_comparison/results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_1_strict_vs_critical(output_dir)
    comparison_2_context_modes(output_dir)
    comparison_3_condition_effects_across_experiments(output_dir)

    print(f"\nAll comparison results saved to {output_dir}")


if __name__ == "__main__":
    main()
