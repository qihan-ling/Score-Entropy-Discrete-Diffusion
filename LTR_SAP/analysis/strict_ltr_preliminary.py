"""
Strict-LTR filler-independent analyses on experimental items.

Uses weighted_steps (position-deconfounded) as the primary steps metric.

Analyses:
  1. Condition-level effect tests: do SEDD metrics differentiate conditions?
  2. Effect direction alignment: compare SEDD vs human reading time effect signs
  3. Item-level rank correlation: Spearman between SEDD metrics and human measures
  4. SEDD vs GPT-2 surprisal comparison

Usage:
  python LTR_SAP/analysis/strict_ltr_preliminary.py --output_dir LTR_SAP/analysis/results/strict_ltr
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
    load_sap_csv, load_all_outputs, extract_word_metrics,
    compute_weighted_steps, load_gpt2_surprisals,
    load_eye_tracking, load_position_info,
    ET_MEASURES, setup_matplotlib, condition_palette,
)


def collect_strict_ltr_metrics(subset, total_steps=1024):
    """Collect word-level metrics from all strict-LTR outputs for a subset.

    Returns DataFrame with: item, condition, word_pos, word, steps_to_commit,
    weighted_steps, surprisal, entropy, cumulative_kl
    """
    csv_files = get_sap_files()
    csv_path = None
    for f in csv_files:
        if get_subset_name(f) == subset:
            csv_path = f
            break
    if csv_path is None:
        raise ValueError(f"Subset {subset} not found")

    df_stim = load_sap_csv(csv_path)
    crit_col = get_critical_pos_col(csv_path)
    cond_col = "condition" if "condition" in df_stim.columns else None

    all_rows = []
    conditions = df_stim[cond_col].unique() if cond_col else [None]

    for condition in conditions:
        outputs = load_all_outputs(subset, condition)
        if not outputs:
            continue

        if cond_col:
            items_df = df_stim[df_stim[cond_col] == condition]
        else:
            items_df = df_stim

        for output in outputs:
            sentence = output["tokenization"]["sentence"]
            matching = items_df[items_df["Sentence"] == sentence]
            if matching.empty:
                continue

            item_row = matching.iloc[0]
            item_id = item_row.get("item", None)
            crit_pos = int(item_row[crit_col]) if crit_col else None

            commitment_log = compute_weighted_steps(
                output["commitment_log"], total_steps
            )
            word_df = extract_word_metrics(output)

            # Merge weighted_steps into word_df by position
            ws_by_pos = {}
            tc_by_pos = {}
            for entry in commitment_log:
                pos = entry["position"]
                ws_by_pos[pos] = entry.get("weighted_steps", np.nan)
                tc_by_pos[pos] = entry.get("t_commitment", np.nan)

            word_df["item"] = item_id
            word_df["condition"] = condition
            word_df["critical_pos"] = crit_pos
            if crit_pos is not None:
                word_df["relative_pos"] = word_df["word_pos"] - (crit_pos - 1)

            all_rows.append(word_df)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def analysis_1_condition_effects(df, subset, output_dir):
    """Test whether SEDD metrics differentiate conditions at the critical position."""
    plt = setup_matplotlib()

    if "relative_pos" not in df.columns or "condition" not in df.columns:
        print(f"  Skipping {subset}: no condition or critical position info")
        return

    crit = df[df["relative_pos"] == 0].copy()
    conditions = sorted(crit["condition"].dropna().unique())
    if len(conditions) < 2:
        print(f"  Skipping {subset}: fewer than 2 conditions")
        return

    metrics = ["steps_to_commit", "surprisal", "entropy", "cumulative_kl"]
    available_metrics = [m for m in metrics if m in crit.columns]

    print(f"\n  === Condition effects at critical position ({subset}) ===")
    results = []
    for metric in available_metrics:
        groups = []
        for cond in conditions:
            vals = crit[crit["condition"] == cond][metric].dropna()
            groups.append(vals)
            print(f"    {metric} | {cond}: mean={vals.mean():.3f}, sd={vals.std():.3f}, n={len(vals)}")

        if len(groups) == 2 and len(groups[0]) > 1 and len(groups[1]) > 1:
            t_stat, p_val = stats.ttest_ind(groups[0], groups[1])
            u_stat, p_mann = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
            effect = groups[0].mean() - groups[1].mean()
            print(f"    -> diff={effect:.3f}, t={t_stat:.3f}, p={p_val:.4f} (Mann-Whitney p={p_mann:.4f})")
            results.append({
                "subset": subset, "metric": metric,
                "cond_a": conditions[0], "cond_b": conditions[1],
                "mean_a": groups[0].mean(), "mean_b": groups[1].mean(),
                "diff": effect, "t_stat": t_stat, "p_ttest": p_val, "p_mannwhitney": p_mann,
            })

    if results:
        pd.DataFrame(results).to_csv(output_dir / f"{subset}_condition_effects.csv", index=False)


def analysis_2_effect_direction(df, subset, output_dir):
    """Compare effect direction between SEDD metrics and human reading times."""
    if "relative_pos" not in df.columns or "condition" not in df.columns:
        return

    conditions = sorted(df["condition"].dropna().unique())
    if len(conditions) < 2:
        return

    # Compute SEDD effect at ROI 0, +1, +2
    print(f"\n  === Effect direction profile ({subset}) ===")
    sedd_metrics = ["steps_to_commit", "surprisal", "entropy"]
    available = [m for m in sedd_metrics if m in df.columns]

    for roi in [0, 1, 2]:
        roi_data = df[df["relative_pos"] == roi]
        for metric in available:
            g = roi_data.groupby("condition")[metric].mean()
            if len(g) >= 2:
                diff = g.iloc[0] - g.iloc[1]
                print(f"    ROI {roi:+d} | {metric}: {conditions[0]}={g.iloc[0]:.3f}, "
                      f"{conditions[1]}={g.iloc[1]:.3f}, diff={diff:.3f}")


def analysis_3_rank_correlation(df, subset, output_dir):
    """Spearman correlation between SEDD metrics and human reading times at item level."""
    try:
        et_data = load_eye_tracking()
        pos_info = load_position_info()
    except FileNotFoundError:
        print(f"  Skipping rank correlation: ET data not found")
        return

    if "condition" not in df.columns or "item" not in df.columns:
        return

    conditions = sorted(df["condition"].dropna().unique())
    if len(conditions) < 2:
        return

    crit = df[df["relative_pos"] == 0].copy()
    sedd_metrics = ["steps_to_commit", "surprisal", "entropy"]
    available = [m for m in sedd_metrics if m in crit.columns]

    print(f"\n  === Item-level rank correlations ({subset}) ===")
    results = []

    for cond in conditions:
        cond_sedd = crit[crit["condition"] == cond].copy()
        if cond_sedd.empty:
            continue

        # Match with ET data
        cond_et = et_data[et_data["cond"] == cond].copy()
        if cond_et.empty:
            continue

        for sedd_m in available:
            for et_m in ET_MEASURES:
                # Get ET values at critical region
                # ET region columns are like ffdR1..ffdR25
                merged_items = []
                for _, sedd_row in cond_sedd.iterrows():
                    item_id = sedd_row["item"]
                    crit_pos = sedd_row.get("critical_pos")
                    if crit_pos is None:
                        continue
                    et_col = f"{et_m}R{crit_pos}"
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
                    rho, p_val = stats.spearmanr(
                        merged_df["sedd_val"], merged_df["et_val"]
                    )
                    results.append({
                        "subset": subset, "condition": cond,
                        "sedd_metric": sedd_m, "et_metric": et_m,
                        "spearman_rho": rho, "p_value": p_val,
                        "n_items": len(merged_items),
                    })
                    if abs(rho) > 0.3:
                        print(f"    {cond} | {sedd_m} vs {et_m}: rho={rho:.3f}, p={p_val:.4f}, n={len(merged_items)}")

    if results:
        pd.DataFrame(results).to_csv(output_dir / f"{subset}_rank_correlations.csv", index=False)


def analysis_4_sedd_vs_gpt2(df, subset, output_dir):
    """Compare SEDD metrics against GPT-2 surprisal at experimental items."""
    try:
        gpt2_df = load_gpt2_surprisals(subset)
    except FileNotFoundError:
        print(f"  Skipping GPT-2 comparison: file not found for {subset}")
        return

    if "surprisal" not in df.columns:
        return

    print(f"\n  === SEDD vs GPT-2 surprisal ({subset}) ===")

    # GPT-2 data has columns like: item, word_pos, surprisal, ...
    # Match by item and word_pos
    sedd_surp = df[["item", "condition", "word_pos", "surprisal", "steps_to_commit"]].dropna()

    if "item" in gpt2_df.columns and "word_pos" in gpt2_df.columns:
        merged = sedd_surp.merge(
            gpt2_df[["item", "word_pos", "surprisal"]].rename(columns={"surprisal": "gpt2_surprisal"}),
            on=["item", "word_pos"],
            how="inner",
        )
        if not merged.empty:
            rho_surp, p_surp = stats.spearmanr(merged["surprisal"], merged["gpt2_surprisal"])
            rho_steps, p_steps = stats.spearmanr(merged["steps_to_commit"], merged["gpt2_surprisal"])
            print(f"    SEDD surprisal vs GPT-2 surprisal: rho={rho_surp:.3f}, p={p_surp:.4f}")
            print(f"    SEDD steps vs GPT-2 surprisal:     rho={rho_steps:.3f}, p={p_steps:.4f}")

            results = pd.DataFrame([{
                "subset": subset,
                "comparison": "sedd_surprisal_vs_gpt2",
                "spearman_rho": rho_surp, "p_value": p_surp,
            }, {
                "subset": subset,
                "comparison": "sedd_steps_vs_gpt2",
                "spearman_rho": rho_steps, "p_value": p_steps,
            }])
            results.to_csv(output_dir / f"{subset}_sedd_vs_gpt2.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Strict-LTR preliminary validation")
    parser.add_argument("--output_dir", type=str, default="LTR_SAP/analysis/results/strict_ltr")
    parser.add_argument("--subset", type=str, default=None, help="Run for one subset only")
    parser.add_argument("--steps", type=int, default=1024)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subsets = ["Agreement", "ClassicGP", "RelativeClause", "AttachmentAmbiguity"]
    if args.subset:
        subsets = [args.subset]

    for subset in subsets:
        print(f"\n{'='*70}")
        print(f"Strict-LTR Preliminary Validation: {subset}")
        print(f"{'='*70}")

        df = collect_strict_ltr_metrics(subset, total_steps=args.steps)
        if df.empty:
            print(f"  No data found for {subset}")
            continue

        print(f"  Loaded {len(df)} word-level entries")

        analysis_1_condition_effects(df, subset, output_dir)
        analysis_2_effect_direction(df, subset, output_dir)
        analysis_3_rank_correlation(df, subset, output_dir)
        analysis_4_sedd_vs_gpt2(df, subset, output_dir)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
