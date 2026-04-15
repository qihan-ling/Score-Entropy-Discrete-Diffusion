"""
Factor reading-time validation: test which factor (final_entropy as score
sharpness vs. position as context quality vs. weighted_steps) best explains
reading-time variance.

Uses both strict-LTR and critical-position filler conversion models.

This script requires filler results from BOTH experiments to be collected first,
and the filler conversion model to have been fitted.

Pipeline:
  1. Load filler metrics from both experiments
  2. Merge with SPR filler reading times
  3. Fit mixed-effects models testing each factor
  4. Compare model fits (AIC/BIC)
  5. Produce summary table and visualization

Usage:
  python LTR_SAP_comparison/factor_reading_time_validation.py \
      --output_dir LTR_SAP_comparison/results
"""

import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "LTR_SAP", "analysis"))

import json
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

from utils import (
    load_all_outputs, extract_word_metrics, compute_weighted_steps,
    load_spr_data, load_gpt2_surprisals,
    setup_matplotlib, LTR_SAP_DIR, LTR_SAP_CRITICAL_DIR,
)


def collect_strict_ltr_filler_metrics(total_steps=1024):
    """Collect word-level metrics from strict-LTR filler outputs."""
    outputs = load_all_outputs("filler")
    if not outputs:
        return pd.DataFrame()

    all_rows = []
    for output in outputs:
        entries = compute_weighted_steps(output["commitment_log"], total_steps)
        word_df = extract_word_metrics(output)

        sentence = output["tokenization"]["sentence"]
        words = sentence.split()

        for _, wrow in word_df.iterrows():
            row = wrow.to_dict()
            row["sentence"] = sentence
            row["experiment"] = "strict_ltr"
            # Find weighted_steps for this word's tokens
            pos_entries = [e for e in entries if e["position"] - 1 == row.get("word_pos")]
            if pos_entries:
                row["weighted_steps"] = sum(e["weighted_steps"] for e in pos_entries)
                row["t_commitment"] = pos_entries[0]["t_commitment"]
            all_rows.append(row)

    return pd.DataFrame(all_rows)


def collect_critical_filler_metrics():
    """Collect word-level metrics from critical-position filler outputs."""
    filler_dir = LTR_SAP_CRITICAL_DIR / "filler"
    if not filler_dir.exists():
        return pd.DataFrame()

    all_rows = []
    for json_file in sorted(filler_dir.glob("item_*_wpos_*.json")):
        with open(json_file) as f:
            output = json.load(f)
        entry = output.get("commitment_log", {})
        if not entry:
            continue
        tok = output.get("tokenization", {})

        all_rows.append({
            "word_pos": entry.get("word_position", 0) - 1,
            "word": entry.get("word"),
            "steps_to_commit": entry.get("steps_taken"),
            "steps_taken": entry.get("steps_taken"),
            "t_commitment": entry.get("t_commitment"),
            "final_entropy": entry.get("final_entropy"),
            "surprisal": entry.get("final_surprisal"),
            "cumulative_kl": entry.get("cumulative_kl"),
            "sentence": tok.get("sentence"),
            "experiment": "critical_position",
        })

    return pd.DataFrame(all_rows)


def merge_with_spr(metrics_df):
    """Merge filler metrics with SPR reading times."""
    try:
        spr = load_spr_data("filler")
    except FileNotFoundError:
        print("  SPR filler data not found")
        return pd.DataFrame()

    metrics_df = metrics_df.copy()
    metrics_df["WordPosition"] = metrics_df["word_pos"] + 1

    merged = pd.merge(
        spr,
        metrics_df.drop_duplicates(subset=["sentence", "WordPosition"]),
        left_on=["Sentence", "WordPosition"],
        right_on=["sentence", "WordPosition"],
        how="inner",
    )
    return merged


def analyze_factor_variance(merged_df, experiment_name, output_dir):
    """Test which factor best explains reading-time variance."""
    plt = setup_matplotlib()

    print(f"\n  === Factor variance analysis ({experiment_name}) ===")

    if merged_df.empty:
        print("  No data for analysis")
        return

    # Correlations with RT
    predictors = ["steps_to_commit", "surprisal", "final_entropy",
                   "cumulative_kl", "word_pos"]
    if "weighted_steps" in merged_df.columns:
        predictors.append("weighted_steps")

    available = [p for p in predictors if p in merged_df.columns]

    results = []
    for pred in available:
        clean = merged_df[["RT", pred]].dropna()
        if len(clean) < 10:
            continue
        r, p = stats.pearsonr(clean["RT"], clean[pred])
        rho, p_rho = stats.spearmanr(clean["RT"], clean[pred])
        print(f"    {pred:20s}: Pearson r={r:.4f} (p={p:.4f}), "
              f"Spearman rho={rho:.4f} (p={p_rho:.4f}), n={len(clean)}")
        results.append({
            "experiment": experiment_name,
            "predictor": pred,
            "pearson_r": r,
            "p_pearson": p,
            "spearman_rho": rho,
            "p_spearman": p_rho,
            "n": len(clean),
            "r_squared": r ** 2,
        })

    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("r_squared", ascending=False)
        results_df.to_csv(
            output_dir / f"factor_validation_{experiment_name}.csv", index=False
        )

        # Bar chart of R-squared
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(results_df)), results_df["r_squared"].values)
        ax.set_xticks(range(len(results_df)))
        ax.set_xticklabels(results_df["predictor"].values, rotation=45, ha="right")
        ax.set_ylabel("R² with RT")
        ax.set_title(f"Factor Predictive Power ({experiment_name})")
        fig.tight_layout()
        fig.savefig(output_dir / f"factor_r2_{experiment_name}.png")
        plt.close(fig)

    # Partial correlations: each factor controlling for the other
    if "steps_to_commit" in merged_df.columns and "final_entropy" in merged_df.columns:
        clean = merged_df[["RT", "steps_to_commit", "final_entropy"]].dropna()
        if len(clean) > 20:
            # Partial correlation of steps_to_commit with RT, controlling for entropy
            from scipy.stats import pearsonr
            resid_steps = clean["steps_to_commit"] - np.polyval(
                np.polyfit(clean["final_entropy"], clean["steps_to_commit"], 1),
                clean["final_entropy"]
            )
            resid_rt = clean["RT"] - np.polyval(
                np.polyfit(clean["final_entropy"], clean["RT"], 1),
                clean["final_entropy"]
            )
            r_partial, p_partial = pearsonr(resid_steps, resid_rt)
            print(f"\n    Partial r(steps|entropy, RT): r={r_partial:.4f}, p={p_partial:.4f}")

            # Reverse
            resid_entropy = clean["final_entropy"] - np.polyval(
                np.polyfit(clean["steps_to_commit"], clean["final_entropy"], 1),
                clean["steps_to_commit"]
            )
            resid_rt2 = clean["RT"] - np.polyval(
                np.polyfit(clean["steps_to_commit"], clean["RT"], 1),
                clean["steps_to_commit"]
            )
            r_partial2, p_partial2 = pearsonr(resid_entropy, resid_rt2)
            print(f"    Partial r(entropy|steps, RT): r={r_partial2:.4f}, p={p_partial2:.4f}")

    # Generate R script for mixed-effects models
    r_script_content = f"""# Factor reading-time validation: {experiment_name}
# Generated by factor_reading_time_validation.py
# Run: Rscript LTR_SAP_comparison/results/factor_validation_{experiment_name}.R

library(lme4)
library(dplyr)

data <- read.csv("{output_dir / f'filler_spr_merged_{experiment_name}.csv'}")
cat(sprintf("Loaded %d rows\\n", nrow(data)))

# Z-score predictors
data$steps_z <- scale(data$steps_to_commit)
data$entropy_z <- scale(data$final_entropy)
data$kl_z <- scale(data$cumulative_kl)
data$pos_z <- scale(data$word_pos)

# Model 1: Steps only
m1 <- tryCatch(lmer(RT ~ steps_z + pos_z + (1 | participant) + (1 | item),
                     data = data, REML=FALSE), error = function(e) NULL)

# Model 2: Entropy only
m2 <- tryCatch(lmer(RT ~ entropy_z + pos_z + (1 | participant) + (1 | item),
                     data = data, REML=FALSE), error = function(e) NULL)

# Model 3: Steps + Entropy
m3 <- tryCatch(lmer(RT ~ steps_z + entropy_z + pos_z + (1 | participant) + (1 | item),
                     data = data, REML=FALSE), error = function(e) NULL)

# Model 4: KL only
m4 <- tryCatch(lmer(RT ~ kl_z + pos_z + (1 | participant) + (1 | item),
                     data = data, REML=FALSE), error = function(e) NULL)

# Compare
models <- list(steps=m1, entropy=m2, steps_entropy=m3, kl=m4)
for (name in names(models)) {{
    m <- models[[name]]
    if (!is.null(m)) {{
        cat(sprintf("%s: AIC=%.1f, BIC=%.1f\\n", name, AIC(m), BIC(m)))
    }}
}}

cat("\\nDone.\\n")
"""

    # Save merged data for R
    if not merged_df.empty:
        merged_df.to_csv(
            output_dir / f"filler_spr_merged_{experiment_name}.csv", index=False
        )
        with open(output_dir / f"factor_validation_{experiment_name}.R", "w") as f:
            f.write(r_script_content)
        print(f"\n  Saved R script: factor_validation_{experiment_name}.R")


def main():
    parser = argparse.ArgumentParser(description="Factor reading-time validation")
    parser.add_argument("--output_dir", type=str, default="LTR_SAP_comparison/results")
    parser.add_argument("--steps", type=int, default=1024)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print("Factor Reading-Time Validation")
    print(f"{'='*70}")

    # Strict-LTR
    print("\nCollecting strict-LTR filler metrics...")
    strict_metrics = collect_strict_ltr_filler_metrics(args.steps)
    if not strict_metrics.empty:
        print(f"  {len(strict_metrics)} word-level entries")
        strict_merged = merge_with_spr(strict_metrics)
        if not strict_merged.empty:
            print(f"  {len(strict_merged)} merged with SPR")
            analyze_factor_variance(strict_merged, "strict_ltr", output_dir)
        else:
            print("  Could not merge with SPR data")
    else:
        print("  No strict-LTR filler results found")

    # Critical-position
    print("\nCollecting critical-position filler metrics...")
    critical_metrics = collect_critical_filler_metrics()
    if not critical_metrics.empty:
        print(f"  {len(critical_metrics)} entries")
        critical_merged = merge_with_spr(critical_metrics)
        if not critical_merged.empty:
            print(f"  {len(critical_merged)} merged with SPR")
            analyze_factor_variance(critical_merged, "critical_position", output_dir)
        else:
            print("  Could not merge with SPR data")
    else:
        print("  No critical-position filler results found")

    # Comparison: which experiment's factors better predict RT?
    print(f"\n{'='*70}")
    print("Cross-experiment factor comparison")
    print(f"{'='*70}")

    strict_results = output_dir / "factor_validation_strict_ltr.csv"
    critical_results = output_dir / "factor_validation_critical_position.csv"

    if strict_results.exists() and critical_results.exists():
        strict_df = pd.read_csv(strict_results)
        critical_df = pd.read_csv(critical_results)

        print("\n  Strict-LTR R² values:")
        for _, row in strict_df.iterrows():
            print(f"    {row['predictor']:20s}: R²={row['r_squared']:.4f}")

        print("\n  Critical-Position R² values:")
        for _, row in critical_df.iterrows():
            print(f"    {row['predictor']:20s}: R²={row['r_squared']:.4f}")

        # Combined summary
        strict_df["experiment"] = "strict_ltr"
        critical_df["experiment"] = "critical_position"
        combined = pd.concat([strict_df, critical_df], ignore_index=True)
        combined.to_csv(output_dir / "factor_comparison_summary.csv", index=False)
        print(f"\n  Saved factor_comparison_summary.csv")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
