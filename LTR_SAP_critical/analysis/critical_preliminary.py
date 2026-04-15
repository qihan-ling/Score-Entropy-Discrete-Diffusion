"""
Critical-position filler-independent analyses.

Same 4 analyses as strict-LTR preliminary but using position-confound-free
critical-position results. Also analyzes future_scores to see if the model's
look-ahead predictions correlate with condition effects.

Usage:
  python LTR_SAP_critical/analysis/critical_preliminary.py \
      --output_dir LTR_SAP_critical/analysis/results
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "LTR_SAP", "analysis"))

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

from utils import (
    get_sap_files, get_subset_name, get_critical_pos_col,
    load_sap_csv, load_critical_outputs_by_offset,
    load_gpt2_surprisals, load_eye_tracking,
    ET_MEASURES, setup_matplotlib, condition_palette,
    LTR_SAP_CRITICAL_DIR,
)


def collect_critical_metrics(subset):
    """Collect metrics from critical-position outputs for a subset.

    Returns DataFrame with: item, condition, offset, steps_taken, t_commitment,
    final_entropy, final_surprisal, cumulative_kl, word, committed_token, correct
    """
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

    all_rows = []
    for condition in conditions:
        by_offset = load_critical_outputs_by_offset(subset, condition)
        for offset, outputs in by_offset.items():
            for output in outputs:
                entry = output.get("commitment_log", {})
                if not entry:
                    continue
                config = output.get("config", {})
                tok = output.get("tokenization", {})

                # Extract item ID from the sentence match
                sentence = tok.get("sentence", "")
                if cond_col:
                    matching = df_stim[(df_stim[cond_col] == condition) & (df_stim["Sentence"] == sentence)]
                else:
                    matching = df_stim[df_stim["Sentence"] == sentence]

                item_id = matching.iloc[0].get("item") if not matching.empty else None

                row = {
                    "item": item_id,
                    "condition": condition,
                    "offset": offset,
                    "word_position": entry.get("word_position"),
                    "word": entry.get("word"),
                    "steps_taken": entry.get("steps_taken"),
                    "t_commitment": entry.get("t_commitment"),
                    "final_entropy": entry.get("final_entropy"),
                    "final_surprisal": entry.get("final_surprisal"),
                    "cumulative_kl": entry.get("cumulative_kl"),
                    "committed_token": entry.get("committed_token"),
                    "correct": entry.get("correct"),
                }
                all_rows.append(row)

    return pd.DataFrame(all_rows)


def analysis_1_condition_effects(df, subset, output_dir):
    """Test whether SEDD metrics differentiate conditions at offset=0."""
    if "condition" not in df.columns:
        return

    crit = df[df["offset"] == 0].copy()
    conditions = sorted(crit["condition"].dropna().unique())
    if len(conditions) < 2:
        return

    metrics = ["steps_taken", "final_surprisal", "final_entropy", "cumulative_kl"]
    available = [m for m in metrics if m in crit.columns]

    print(f"\n  === Condition effects at critical position ({subset}) ===")
    results = []
    for metric in available:
        groups = []
        for cond in conditions:
            vals = crit[crit["condition"] == cond][metric].dropna()
            groups.append(vals)
            print(f"    {metric} | {cond}: mean={vals.mean():.3f}, sd={vals.std():.3f}, n={len(vals)}")

        if len(groups) == 2 and all(len(g) > 1 for g in groups):
            t_stat, p_val = stats.ttest_ind(groups[0], groups[1])
            u_stat, p_mann = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
            effect = groups[0].mean() - groups[1].mean()
            print(f"    -> diff={effect:.3f}, t={t_stat:.3f}, p={p_val:.4f} "
                  f"(Mann-Whitney p={p_mann:.4f})")
            results.append({
                "subset": subset, "metric": metric,
                "cond_a": conditions[0], "cond_b": conditions[1],
                "mean_a": groups[0].mean(), "mean_b": groups[1].mean(),
                "diff": effect, "t_stat": t_stat,
                "p_ttest": p_val, "p_mannwhitney": p_mann,
            })

    if results:
        pd.DataFrame(results).to_csv(
            output_dir / f"{subset}_condition_effects.csv", index=False
        )


def analysis_2_effect_profile(df, subset, output_dir):
    """Effect direction and spillover profile across offsets."""
    plt = setup_matplotlib()
    palette = condition_palette()

    if "condition" not in df.columns:
        return

    conditions = sorted(df["condition"].dropna().unique())
    if len(conditions) < 2:
        return

    print(f"\n  === Effect profile across offsets ({subset}) ===")

    metrics = ["steps_taken", "final_surprisal", "final_entropy"]
    available = [m for m in metrics if m in df.columns]

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        for cond in conditions:
            cond_data = df[df["condition"] == cond]
            means = cond_data.groupby("offset")[metric].mean()
            sems = cond_data.groupby("offset")[metric].sem()
            color = palette.get(cond, None)
            ax.errorbar(means.index, means.values, yerr=sems.values,
                        label=cond, marker="o", color=color, capsize=3)
        ax.set_xlabel("Offset from critical position")
        ax.set_ylabel(metric)
        ax.legend()
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

        for offset in sorted(df["offset"].unique()):
            groups = [df[(df["condition"] == c) & (df["offset"] == offset)][metric].dropna()
                      for c in conditions]
            if len(groups) == 2 and all(len(g) > 1 for g in groups):
                diff = groups[0].mean() - groups[1].mean()
                print(f"    offset {offset:+d} | {metric}: "
                      f"{conditions[0]}={groups[0].mean():.3f}, "
                      f"{conditions[1]}={groups[1].mean():.3f}, diff={diff:.3f}")

    plt.suptitle(f"Effect Profile: {subset} (Critical-Position)")
    plt.tight_layout()
    plt.savefig(output_dir / f"{subset}_effect_profile.png")
    plt.close()


def analysis_3_rank_correlation(df, subset, output_dir):
    """Spearman correlation between critical-position SEDD metrics and human ET."""
    try:
        et_data = load_eye_tracking()
    except FileNotFoundError:
        print(f"  Skipping rank correlation: ET data not found")
        return

    if "condition" not in df.columns:
        return

    crit = df[df["offset"] == 0].copy()
    conditions = sorted(crit["condition"].dropna().unique())
    sedd_metrics = ["steps_taken", "final_surprisal", "final_entropy"]
    available = [m for m in sedd_metrics if m in crit.columns]

    print(f"\n  === Item-level rank correlations ({subset}) ===")
    results = []

    for cond in conditions:
        cond_sedd = crit[crit["condition"] == cond]
        cond_et = et_data[et_data["cond"] == cond]
        if cond_sedd.empty or cond_et.empty:
            continue

        for sedd_m in available:
            for et_m in ET_MEASURES:
                merged_items = []
                for _, sedd_row in cond_sedd.iterrows():
                    item_id = sedd_row["item"]
                    wpos = sedd_row.get("word_position")
                    if wpos is None:
                        continue
                    et_col = f"{et_m}R{wpos}"
                    if et_col not in cond_et.columns:
                        continue
                    et_vals = cond_et[cond_et["item"] == item_id][et_col].dropna()
                    if et_vals.empty:
                        continue
                    merged_items.append({
                        "item": item_id,
                        "sedd_val": sedd_row[sedd_m],
                        "et_val": et_vals.mean(),
                    })

                if len(merged_items) >= 5:
                    merged_df = pd.DataFrame(merged_items)
                    rho, p_val = stats.spearmanr(merged_df["sedd_val"], merged_df["et_val"])
                    results.append({
                        "subset": subset, "condition": cond,
                        "sedd_metric": sedd_m, "et_metric": et_m,
                        "spearman_rho": rho, "p_value": p_val,
                        "n_items": len(merged_items),
                    })
                    if abs(rho) > 0.3:
                        print(f"    {cond} | {sedd_m} vs {et_m}: "
                              f"rho={rho:.3f}, p={p_val:.4f}, n={len(merged_items)}")

    if results:
        pd.DataFrame(results).to_csv(
            output_dir / f"{subset}_rank_correlations.csv", index=False
        )


def analysis_4_sedd_vs_gpt2(df, subset, output_dir):
    """Compare critical-position SEDD metrics with GPT-2 surprisal."""
    try:
        gpt2_df = load_gpt2_surprisals(subset)
    except FileNotFoundError:
        print(f"  Skipping GPT-2 comparison: file not found for {subset}")
        return

    crit = df[df["offset"] == 0].copy()
    if crit.empty or "final_surprisal" not in crit.columns:
        return

    print(f"\n  === SEDD vs GPT-2 surprisal ({subset}) ===")

    if "item" in gpt2_df.columns and "word_pos" in gpt2_df.columns:
        merged = crit.merge(
            gpt2_df[["item", "word_pos", "surprisal"]].rename(
                columns={"surprisal": "gpt2_surprisal", "word_pos": "word_position"}
            ),
            on=["item", "word_position"],
            how="inner",
        )
        if not merged.empty:
            for col in ["final_surprisal", "steps_taken"]:
                if col not in merged.columns:
                    continue
                clean = merged[[col, "gpt2_surprisal"]].dropna()
                if len(clean) >= 5:
                    rho, p = stats.spearmanr(clean[col], clean["gpt2_surprisal"])
                    print(f"    {col} vs GPT-2: rho={rho:.3f}, p={p:.4f}, n={len(clean)}")


def analysis_5_future_scores(df, subset, output_dir):
    """Analyze whether future position scores differ between conditions."""
    if "condition" not in df.columns:
        return

    conditions = sorted(df["condition"].dropna().unique())
    if len(conditions) < 2:
        return

    # Load the raw outputs to get future_scores
    print(f"\n  === Future scores analysis ({subset}) ===")
    # This analysis requires loading raw JSONs which have future_scores
    # For now, report what percentage of outputs have future_scores data
    for condition in conditions:
        by_offset = load_critical_outputs_by_offset(subset, condition, offsets=[0])
        outputs = by_offset.get(0, [])
        n_with_future = sum(1 for o in outputs if o.get("future_scores"))
        print(f"    {condition}: {n_with_future}/{len(outputs)} outputs have future_scores")


def main():
    parser = argparse.ArgumentParser(description="Critical-position preliminary validation")
    parser.add_argument("--output_dir", type=str, default="LTR_SAP_critical/analysis/results")
    parser.add_argument("--subset", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subsets = ["Agreement", "ClassicGP", "RelativeClause", "AttachmentAmbiguity"]
    if args.subset:
        subsets = [args.subset]

    for subset in subsets:
        print(f"\n{'='*70}")
        print(f"Critical-Position Preliminary Validation: {subset}")
        print(f"{'='*70}")

        df = collect_critical_metrics(subset)
        if df.empty:
            print(f"  No data found for {subset}")
            continue

        print(f"  Loaded {len(df)} entries ({df['offset'].nunique()} offsets)")

        analysis_1_condition_effects(df, subset, output_dir)
        analysis_2_effect_profile(df, subset, output_dir)
        analysis_3_rank_correlation(df, subset, output_dir)
        analysis_4_sedd_vs_gpt2(df, subset, output_dir)
        analysis_5_future_scores(df, subset, output_dir)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
