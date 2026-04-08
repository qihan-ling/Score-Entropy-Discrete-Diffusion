"""
Plan C: Predicted vs. empirical effects and magnitude underestimation test.

Uses the filler-calibrated model coefficients to predict reading times at
experimental items' critical regions (ROI 0, 1, 2). Compares predicted vs.
empirical effects across ALL reading time measures (SPR RT, FFD, GZ, GP, TT,
regin, regout).

Reads:
  LTR_SAP/analysis/data/sedd_spr_merged.csv
  LTR_SAP/analysis/data/sedd_et_merged.csv
  LTR_SAP/analysis/data/filler_model_coefficients.csv

Produces:
  - Predicted vs empirical effect size plots per subset and ROI
  - Magnitude ratio bar charts (predicted/empirical)
  - Correlation tables

Usage:
  python LTR_SAP/analysis/plan_c_regression.py --output_dir LTR_SAP/analysis/figures/plan_c
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from utils import setup_matplotlib, ET_MEASURES, SPR_MEASURES


def compute_condition_effects(df, subset, metric_col, condition_col="condition"):
    """Compute condition effects (difference between conditions) per item and ROI.

    For each subset, defines the contrast:
        Agreement: UNAGREE - AGREE
        ClassicGP: AMB - UAMB (for NPS, NPZ, MVRR subtypes)
        RelativeClause: RC_Obj - RC_Subj
        AttachmentAmbiguity: AttachHigh/Low - AttachMulti
    """
    contrasts = {
        "Agreement": [("UNAGREE", "AGREE")],
        "ClassicGP": [("NPS_AMB", "NPS_UAMB"), ("NPZ_AMB", "NPZ_UAMB"), ("MVRR_AMB", "MVRR_UAMB")],
        "RelativeClause": [("RC_Obj", "RC_Subj")],
        "AttachmentAmbiguity": [("AttachHigh", "AttachMulti"), ("AttachLow", "AttachMulti")],
    }

    if subset not in contrasts:
        return pd.DataFrame()

    results = []
    for cond_a, cond_b in contrasts[subset]:
        df_a = df[df[condition_col] == cond_a]
        df_b = df[df[condition_col] == cond_b]

        if df_a.empty or df_b.empty:
            continue

        for roi in [0, 1, 2]:
            roi_a = df_a[df_a["ROI"] == roi] if "ROI" in df_a.columns else pd.DataFrame()
            roi_b = df_b[df_b["ROI"] == roi] if "ROI" in df_b.columns else pd.DataFrame()

            if roi_a.empty or roi_b.empty:
                continue

            mean_a = roi_a[metric_col].mean()
            mean_b = roi_b[metric_col].mean()
            effect = mean_a - mean_b

            results.append({
                "subset": subset,
                "contrast": f"{cond_a} - {cond_b}",
                "ROI": roi,
                "metric": metric_col,
                "effect_size": effect,
                "n_a": len(roi_a),
                "n_b": len(roi_b),
            })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Plan C: Predicted vs empirical effects")
    parser.add_argument("--output_dir", type=str, default="LTR_SAP/analysis/figures/plan_c")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plt = setup_matplotlib()

    # --- Load data ---
    spr_path = "LTR_SAP/analysis/data/sedd_spr_merged.csv"
    et_path = "LTR_SAP/analysis/data/sedd_et_merged.csv"
    coef_path = "LTR_SAP/analysis/data/filler_model_coefficients.csv"

    # SPR effects
    if os.path.exists(spr_path):
        print("Loading SPR merged data...")
        spr = pd.read_csv(spr_path)

        # Compute empirical RT effects per subset
        all_spr_effects = []
        for subset in spr["subset"].unique():
            if subset == "filler":
                continue
            sub = spr[spr["subset"] == subset]
            for metric in ["RT"]:
                effects = compute_condition_effects(sub, subset, metric)
                if not effects.empty:
                    effects["data_type"] = "empirical"
                    all_spr_effects.append(effects)

            # Predicted effects using SEDD metrics
            for metric in ["steps_to_commit", "surprisal", "cumulative_kl"]:
                if metric in sub.columns:
                    effects = compute_condition_effects(sub, subset, metric)
                    if not effects.empty:
                        effects["data_type"] = f"predicted_{metric}"
                        all_spr_effects.append(effects)

        if all_spr_effects:
            spr_effects = pd.concat(all_spr_effects, ignore_index=True)
            spr_effects.to_csv(os.path.join(args.output_dir, "spr_effects.csv"), index=False)
            print(f"  Saved spr_effects.csv ({len(spr_effects)} rows)")

            # Plot per subset
            for subset in spr_effects["subset"].unique():
                sub_eff = spr_effects[spr_effects["subset"] == subset]
                empirical = sub_eff[sub_eff["data_type"] == "empirical"]

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                for i, roi in enumerate([0, 1, 2]):
                    ax = axes[i]
                    emp_roi = empirical[empirical["ROI"] == roi]

                    if emp_roi.empty:
                        continue

                    contrasts = emp_roi["contrast"].unique()
                    x = np.arange(len(contrasts))
                    width = 0.2

                    # Empirical
                    ax.bar(x - width, emp_roi["effect_size"].values, width, label="Empirical RT")

                    # Predicted with each metric
                    for j, metric in enumerate(["steps_to_commit", "surprisal", "cumulative_kl"]):
                        pred = sub_eff[(sub_eff["data_type"] == f"predicted_{metric}") & (sub_eff["ROI"] == roi)]
                        if not pred.empty:
                            ax.bar(x + width * j, pred["effect_size"].values, width, label=metric)

                    ax.set_title(f"ROI {roi}")
                    ax.set_xticks(x)
                    ax.set_xticklabels(contrasts, rotation=30, ha="right", fontsize=8)
                    if i == 0:
                        ax.legend(fontsize=8)

                fig.suptitle(f"{subset}: Effect sizes by ROI")
                fig.tight_layout()
                fig.savefig(os.path.join(args.output_dir, f"effects_{subset}.png"))
                plt.close(fig)
                print(f"  Saved effects_{subset}.png")

    # --- Eye-tracking effects ---
    if os.path.exists(et_path):
        print("\nLoading eye-tracking merged data...")
        et = pd.read_csv(et_path)

        all_et_effects = []
        for subset in et["subset"].unique():
            if subset == "filler":
                continue
            sub = et[et["subset"] == subset]
            for measure in ET_MEASURES:
                if measure not in sub.columns:
                    continue
                effects = compute_condition_effects(sub, subset, measure)
                if not effects.empty:
                    effects["data_type"] = "empirical"
                    all_et_effects.append(effects)

        if all_et_effects:
            et_effects = pd.concat(all_et_effects, ignore_index=True)
            et_effects.to_csv(os.path.join(args.output_dir, "et_effects.csv"), index=False)
            print(f"  Saved et_effects.csv ({len(et_effects)} rows)")

    # --- Magnitude underestimation analysis ---
    if os.path.exists(coef_path):
        print("\nLoading filler model coefficients...")
        coefs = pd.read_csv(coef_path)
        print(f"  Models: {coefs['model'].unique().tolist()}")
        print(f"  Coefficients loaded:")
        print(coefs.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
