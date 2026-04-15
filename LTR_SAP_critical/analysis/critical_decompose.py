"""
Critical-position factor decomposition.

Since the position confound is eliminated (every target gets the full noise
schedule starting from step 0), steps_taken variation is driven purely by:
  1. Score sharpness (proxied by final_entropy)
  2. Content difficulty (what the model is predicting, given perfect prefix)

Correlate final_entropy with steps_taken across items and conditions.

Usage:
  python LTR_SAP_critical/analysis/critical_decompose.py \
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
    setup_matplotlib, condition_palette,
)


def collect_critical_metrics(subset):
    """Collect metrics from critical-position outputs."""
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
                tok = output.get("tokenization", {})
                sentence = tok.get("sentence", "")

                if cond_col:
                    matching = df_stim[(df_stim[cond_col] == condition) & (df_stim["Sentence"] == sentence)]
                else:
                    matching = df_stim[df_stim["Sentence"] == sentence]
                item_id = matching.iloc[0].get("item") if not matching.empty else None

                all_rows.append({
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
                    "correct": entry.get("correct"),
                })

    return pd.DataFrame(all_rows)


def entropy_steps_correlation(df, subset, output_dir):
    """Correlate final_entropy with steps_taken (no position confound)."""
    plt = setup_matplotlib()

    print(f"\n  === Entropy-steps correlation ({subset}) ===")

    valid = df.dropna(subset=["steps_taken", "final_entropy"])
    if len(valid) < 5:
        print("  Insufficient data")
        return

    rho, p = stats.spearmanr(valid["steps_taken"], valid["final_entropy"])
    r, p_pearson = stats.pearsonr(valid["steps_taken"], valid["final_entropy"])
    print(f"    Overall: Spearman rho={rho:.3f} (p={p:.4f}), "
          f"Pearson r={r:.3f} (p={p_pearson:.4f}), n={len(valid)}")

    results = [{"subset": subset, "scope": "all", "offset": "all",
                "spearman_rho": rho, "p_spearman": p,
                "pearson_r": r, "p_pearson": p_pearson, "n": len(valid)}]

    # Per offset
    for offset in sorted(valid["offset"].unique()):
        odata = valid[valid["offset"] == offset]
        if len(odata) < 5:
            continue
        rho_o, p_o = stats.spearmanr(odata["steps_taken"], odata["final_entropy"])
        print(f"    Offset {offset:+d}: rho={rho_o:.3f} (p={p_o:.4f}), n={len(odata)}")
        results.append({"subset": subset, "scope": "by_offset", "offset": offset,
                        "spearman_rho": rho_o, "p_spearman": p_o, "n": len(odata)})

    # Per condition at offset=0
    conditions = sorted(valid["condition"].dropna().unique())
    crit = valid[valid["offset"] == 0]
    for cond in conditions:
        cdata = crit[crit["condition"] == cond]
        if len(cdata) < 5:
            continue
        rho_c, p_c = stats.spearmanr(cdata["steps_taken"], cdata["final_entropy"])
        print(f"    {cond} (offset=0): rho={rho_c:.3f} (p={p_c:.4f}), n={len(cdata)}")
        results.append({"subset": subset, "scope": cond, "offset": 0,
                        "spearman_rho": rho_c, "p_spearman": p_c, "n": len(cdata)})

    pd.DataFrame(results).to_csv(
        output_dir / f"{subset}_entropy_steps_correlation.csv", index=False
    )

    # Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    palette = condition_palette()

    ax = axes[0]
    for cond in conditions:
        cdata = valid[valid["condition"] == cond]
        color = palette.get(cond, None)
        ax.scatter(cdata["final_entropy"], cdata["steps_taken"],
                   alpha=0.4, s=15, label=cond, color=color)
    ax.set_xlabel("Final entropy (bits)")
    ax.set_ylabel("Steps taken")
    ax.set_title(f"All offsets (rho={rho:.3f})")
    ax.legend()

    ax = axes[1]
    for cond in conditions:
        cdata = crit[crit["condition"] == cond]
        color = palette.get(cond, None)
        ax.scatter(cdata["final_entropy"], cdata["steps_taken"],
                   alpha=0.5, s=20, label=cond, color=color)
    crit_rho = stats.spearmanr(crit["steps_taken"].dropna(), crit["final_entropy"].dropna())[0] if len(crit.dropna(subset=["steps_taken","final_entropy"])) >= 5 else float("nan")
    ax.set_xlabel("Final entropy (bits)")
    ax.set_ylabel("Steps taken")
    ax.set_title(f"Offset=0 only (rho={crit_rho:.3f})")
    ax.legend()

    plt.suptitle(f"Entropy-Steps Decomposition: {subset}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{subset}_entropy_steps_scatter.png")
    plt.close()


def condition_entropy_comparison(df, subset, output_dir):
    """Compare final_entropy distributions between conditions at each offset."""
    if "condition" not in df.columns:
        return

    conditions = sorted(df["condition"].dropna().unique())
    if len(conditions) < 2:
        return

    print(f"\n  === Entropy by condition across offsets ({subset}) ===")
    results = []

    for offset in sorted(df["offset"].unique()):
        odata = df[df["offset"] == offset]
        groups = [odata[odata["condition"] == c]["final_entropy"].dropna() for c in conditions]

        if len(groups) == 2 and all(len(g) > 1 for g in groups):
            diff = groups[0].mean() - groups[1].mean()
            t_stat, p_val = stats.ttest_ind(groups[0], groups[1])
            print(f"    offset {offset:+d} | entropy: "
                  f"{conditions[0]}={groups[0].mean():.3f}, "
                  f"{conditions[1]}={groups[1].mean():.3f}, "
                  f"diff={diff:.3f}, p={p_val:.4f}")
            results.append({
                "subset": subset, "offset": offset,
                "cond_a": conditions[0], "cond_b": conditions[1],
                "mean_a": groups[0].mean(), "mean_b": groups[1].mean(),
                "diff": diff, "t_stat": t_stat, "p_value": p_val,
            })

    if results:
        pd.DataFrame(results).to_csv(
            output_dir / f"{subset}_entropy_by_condition.csv", index=False
        )


def accuracy_analysis(df, subset, output_dir):
    """Analyze prediction accuracy: does the model commit to the correct token?"""
    if "correct" not in df.columns:
        return

    print(f"\n  === Prediction accuracy ({subset}) ===")

    for offset in sorted(df["offset"].unique()):
        odata = df[df["offset"] == offset]
        correct = odata["correct"].dropna()
        if len(correct) == 0:
            continue
        acc = correct.mean()
        print(f"    offset {offset:+d}: accuracy={acc:.1%} ({int(correct.sum())}/{len(correct)})")

    conditions = sorted(df["condition"].dropna().unique())
    if len(conditions) >= 2:
        crit = df[df["offset"] == 0]
        for cond in conditions:
            correct = crit[crit["condition"] == cond]["correct"].dropna()
            if len(correct) > 0:
                print(f"    offset=0, {cond}: accuracy={correct.mean():.1%}")


def main():
    parser = argparse.ArgumentParser(description="Critical-position factor decomposition")
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
        print(f"Critical-Position Factor Decomposition: {subset}")
        print(f"{'='*70}")

        df = collect_critical_metrics(subset)
        if df.empty:
            print(f"  No data found for {subset}")
            continue

        print(f"  Loaded {len(df)} entries")

        entropy_steps_correlation(df, subset, output_dir)
        condition_entropy_comparison(df, subset, output_dir)
        accuracy_analysis(df, subset, output_dir)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
