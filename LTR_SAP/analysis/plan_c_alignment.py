"""
Plan C: Align SEDD denoising outputs with human behavioral data.

Merges SEDD word-level metrics (steps_to_commit, cumulative_kl, surprisal, entropy)
with eye-tracking data (all_wide.csv) and SPR data at the item/region level.

Produces merged CSV files ready for regression analysis:
  - sedd_spr_merged.csv: SEDD metrics joined with SPR reading times
  - sedd_et_merged.csv: SEDD metrics joined with eye-tracking measures

Usage:
  python LTR_SAP/analysis/plan_c_alignment.py --output_dir LTR_SAP/analysis/data
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from utils import (
    get_sap_files, get_subset_name, get_critical_pos_col, load_sap_csv,
    load_all_outputs, extract_word_metrics, load_gpt2_surprisals,
    load_eye_tracking, load_spr_data, load_position_info,
    ET_MEASURES, SPR_MEASURES, LTR_SAP_DIR,
)


def build_sedd_metrics_table():
    """Build a unified table of SEDD word-level metrics across all subsets.

    Returns DataFrame with columns:
        subset, condition, item, sentence, word_pos, word,
        steps_to_commit, surprisal, entropy, cumulative_kl, n_tokens
    """
    all_rows = []

    for csv_path in get_sap_files():
        subset = get_subset_name(csv_path)
        df_stim = load_sap_csv(csv_path)
        crit_col = get_critical_pos_col(csv_path)
        cond_col = "condition" if "condition" in df_stim.columns else None

        conditions = df_stim[cond_col].unique().tolist() if cond_col else [None]

        for cond in conditions:
            outputs = load_all_outputs(subset, cond)

            for out in outputs:
                sentence = out["tokenization"]["sentence"]
                match = df_stim[df_stim["Sentence"] == sentence]
                if match.empty:
                    continue

                item = match.iloc[0].get("item", None)
                crit_pos = int(match.iloc[0][crit_col]) if crit_col and crit_col in match.columns else None

                wm = extract_word_metrics(out)
                wm["subset"] = subset
                wm["condition"] = cond
                wm["item"] = item
                wm["sentence"] = sentence
                if crit_pos is not None:
                    wm["critical_pos_1indexed"] = crit_pos
                    wm["ROI"] = wm["word_pos"] - (crit_pos - 1)
                all_rows.append(wm)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def merge_with_gpt2(sedd_df):
    """Merge SEDD metrics with GPT-2 surprisals from sapbenchmark.

    GPT-2 data uses 0-indexed word_pos; SEDD data also uses 0-indexed.
    """
    merged_parts = []
    for subset in sedd_df["subset"].unique():
        try:
            gpt2 = load_gpt2_surprisals(subset)
        except FileNotFoundError:
            print(f"  GPT-2 surprisals not found for {subset}, skipping")
            continue

        gpt2 = gpt2.rename(columns={
            "sum_surprisal": "gpt2_surprisal",
            "sum_surprisal_s": "gpt2_surprisal_s",
            "logfreq": "gpt2_logfreq",
            "logfreq_s": "logfreq_s",
            "length_s": "length_s",
        })
        gpt2["word_pos_0"] = gpt2["word_pos"]
        gpt2_cols = ["Sentence", "word_pos_0", "gpt2_surprisal", "gpt2_surprisal_s",
                     "logfreq_s", "length_s"]
        gpt2_sub = gpt2[[c for c in gpt2_cols if c in gpt2.columns]].copy()

        sedd_sub = sedd_df[sedd_df["subset"] == subset].copy()
        sedd_sub["word_pos_0"] = sedd_sub["word_pos"]

        merged = pd.merge(
            sedd_sub, gpt2_sub,
            left_on=["sentence", "word_pos_0"],
            right_on=["Sentence", "word_pos_0"],
            how="left",
        )
        merged_parts.append(merged)

    if not merged_parts:
        return sedd_df
    return pd.concat(merged_parts, ignore_index=True)


def merge_with_et(sedd_df, et_df, position_info):
    """Merge SEDD metrics with eye-tracking data.

    ET data uses wide format (ffdR1..R25 etc) with absolute word regions.
    We need to map (item, condition) -> word regions and extract measures.
    """
    # Pivot ET data from wide to long at ROI level
    et_long_parts = []
    for measure in ET_MEASURES:
        measure_cols = [c for c in et_df.columns if c.startswith(measure + "R")]
        if not measure_cols:
            continue
        for col in measure_cols:
            region = int(col.replace(measure + "R", ""))
            sub = et_df[["subj", "item", "cond", col]].copy()
            sub = sub.rename(columns={col: measure})
            sub["region"] = region
            et_long_parts.append(sub)

    if not et_long_parts:
        return pd.DataFrame()

    et_long = et_long_parts[0]
    for part in et_long_parts[1:]:
        et_long = pd.merge(et_long, part, on=["subj", "item", "cond", "region"], how="outer")

    # Merge with SEDD on (item, condition, word_pos)
    # word_pos is 0-indexed, region is 1-indexed
    sedd_df = sedd_df.copy()
    sedd_df["region"] = sedd_df["word_pos"] + 1

    merged = pd.merge(
        et_long,
        sedd_df[["item", "condition", "region", "subset", "sentence",
                 "steps_to_commit", "surprisal", "entropy", "cumulative_kl",
                 "word", "n_tokens"]].drop_duplicates(),
        left_on=["item", "cond", "region"],
        right_on=["item", "condition", "region"],
        how="inner",
    )
    return merged


def main():
    parser = argparse.ArgumentParser(description="Plan C: Align SEDD with human data")
    parser.add_argument("--output_dir", type=str, default="LTR_SAP/analysis/data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Step 1: Build SEDD metrics table ---
    print("Building SEDD word-level metrics table...")
    sedd_df = build_sedd_metrics_table()
    if sedd_df.empty:
        print("No SEDD outputs found. Run batch_runner.py first.")
        return
    print(f"  {len(sedd_df)} rows across {sedd_df['subset'].nunique()} subsets")

    # --- Step 2: Merge with GPT-2 ---
    print("Merging with GPT-2 surprisals...")
    sedd_gpt2 = merge_with_gpt2(sedd_df)
    print(f"  {len(sedd_gpt2)} rows after GPT-2 merge")

    # Z-score SEDD metrics
    for col in ["steps_to_commit", "surprisal", "entropy", "cumulative_kl"]:
        if col in sedd_gpt2.columns:
            valid = sedd_gpt2[col].dropna()
            if len(valid) > 1:
                sedd_gpt2[col + "_s"] = scipy_stats.zscore(sedd_gpt2[col].fillna(0))

    sedd_gpt2.to_csv(os.path.join(args.output_dir, "sedd_word_metrics.csv"), index=False)
    print(f"  Saved sedd_word_metrics.csv")

    # --- Step 3: Merge with SPR data ---
    print("Merging with SPR data...")
    spr_parts = []
    for subset in ["filler", "Agreement", "ClassicGP", "RelativeClause", "AttachmentAmbiguity"]:
        spr_name = "Fillers" if subset == "filler" else subset
        try:
            spr = load_spr_data(spr_name)
        except FileNotFoundError:
            print(f"  SPR data not found for {spr_name}")
            continue

        sedd_sub = sedd_gpt2[sedd_gpt2["subset"] == subset].copy()
        if sedd_sub.empty:
            continue

        sedd_sub["WordPosition"] = sedd_sub["word_pos"] + 1  # SPR uses 1-indexed
        merged = pd.merge(
            spr, sedd_sub.drop_duplicates(subset=["sentence", "WordPosition"]),
            left_on=["Sentence", "WordPosition"],
            right_on=["sentence", "WordPosition"],
            how="inner",
        )
        if not merged.empty:
            merged["subset"] = subset
            spr_parts.append(merged)
            print(f"  {subset}: {len(merged)} SPR rows merged")

    if spr_parts:
        spr_merged = pd.concat(spr_parts, ignore_index=True)
        spr_merged.to_csv(os.path.join(args.output_dir, "sedd_spr_merged.csv"), index=False)
        print(f"  Saved sedd_spr_merged.csv ({len(spr_merged)} rows)")

    # --- Step 4: Merge with eye-tracking ---
    print("Merging with eye-tracking data...")
    try:
        et_df = load_eye_tracking()
        position_info = load_position_info()
        et_merged = merge_with_et(sedd_gpt2, et_df, position_info)
        if not et_merged.empty:
            et_merged.to_csv(os.path.join(args.output_dir, "sedd_et_merged.csv"), index=False)
            print(f"  Saved sedd_et_merged.csv ({len(et_merged)} rows)")
        else:
            print("  No ET data merged (check item/condition alignment)")
    except FileNotFoundError as e:
        print(f"  ET data not found: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
