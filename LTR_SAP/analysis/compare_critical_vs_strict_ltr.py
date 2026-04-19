"""
Preliminary analysis: Critical-Position vs Strict-LTR for Agreement subset.

Compares:
1. Basic statistics (steps_taken, accuracy, metric coverage)
2. Steps-to-commit distribution across offsets
3. Condition effects (AGREE vs UNAGREE) at each offset
4. Correlation with human eye-tracking data (ffd, gz, gp, tt)
5. Improvement assessment over strict-LTR

Run:
  python LTR_SAP/analysis/compare_critical_vs_strict_ltr.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent.parent
CRITICAL_DIR = REPO / "LTR_SAP_critical" / "Agreement"
STRICT_LTR_DIR = REPO / "LTR_SAP" / "Agreement"
SAP_CSV = REPO / "SAP_stimuli" / "sap_items_Agreement.csv"
ET_CSV = (
    REPO
    / "Huang_et_al_2024_spr_osf"
    / "material & exp_script"
    / "EM_analysis"
    / "R"
    / "all_wide.csv"
)

OFFSETS = [-2, -1, 0, 1, 2, 3]
ET_MEASURES = ["ffd", "gz", "gp", "tt"]


def load_critical_results():
    """Load all critical-position JSON results for Agreement."""
    rows = []
    for cond in ["AGREE", "UNAGREE"]:
        cond_dir = CRITICAL_DIR / cond
        if not cond_dir.exists():
            continue
        for fpath in sorted(cond_dir.glob("item_*_pos_*.json")):
            with open(fpath) as f:
                d = json.load(f)
            cl = d["commitment_log"]
            fname = fpath.stem
            parts = fname.split("_")
            item_id = int(parts[1])
            offset_str = parts[3]
            offset = int(offset_str)

            rows.append({
                "item": item_id,
                "condition": cond,
                "offset": offset,
                "word_position": cl.get("word_position"),
                "word": cl.get("word"),
                "steps_taken": cl.get("steps_taken"),
                "final_surprisal": cl.get("final_surprisal"),
                "final_entropy": cl.get("final_entropy"),
                "cumulative_kl": cl.get("cumulative_kl"),
                "committed_token": cl.get("committed_token"),
                "target_token": cl.get("target_token"),
                "correct": cl.get("correct"),
                "t_commitment": cl.get("t_commitment"),
            })
    return pd.DataFrame(rows)


def load_strict_ltr_results():
    """Load strict-LTR commitment logs for Agreement, extract critical-region positions."""
    sap = pd.read_csv(SAP_CSV)
    rows = []
    for cond in ["AGREE", "UNAGREE"]:
        cond_dir = STRICT_LTR_DIR / cond
        if not cond_dir.exists():
            continue
        for fpath in sorted(cond_dir.glob("item_*.json")):
            with open(fpath) as f:
                d = json.load(f)

            item_id = int(fpath.stem.split("_")[1])
            sap_row = sap[(sap["item"] == item_id) & (sap["condition"] == cond)]
            if sap_row.empty:
                continue
            crit_pos = int(sap_row.iloc[0]["disambPosition"])
            sentence = sap_row.iloc[0]["Sentence"]
            words = sentence.split()

            for cl_entry in d["commitment_log"]:
                pos = cl_entry["position"]
                word_pos = pos  # token position (1-indexed approx.)

                target_tok = cl_entry.get("target_token")
                # Map token position to word position
                # In strict-LTR, position 1 = first word after <eot>
                # word_position in 1-indexed terms ≈ pos (since BPE can differ)
                # Use the target_token to find offset
                if cl_entry.get("target_token") is not None:
                    for offset in OFFSETS:
                        wp = crit_pos + offset
                        if wp < 1 or wp > len(words):
                            continue
                        # Match by word
                        word = words[wp - 1]
                        # Check if this commitment is for this word position
                        # The token position in strict-LTR starts from 1 (pos 0 = <eot>)
                        # This is approximate - we'll match by word text
                        pass

                rows.append({
                    "item": item_id,
                    "condition": cond,
                    "position": pos,
                    "steps_taken": cl_entry.get("steps_taken"),
                    "final_surprisal": cl_entry.get("final_surprisal"),
                    "final_entropy": cl_entry.get("final_entropy"),
                    "cumulative_kl": cl_entry.get("cumulative_kl", 0),
                    "committed_token": cl_entry.get("committed_token"),
                    "target_token": cl_entry.get("target_token"),
                    "correct": cl_entry.get("correct"),
                })

    df = pd.DataFrame(rows)

    # Now map positions to offsets relative to critical position
    sap = pd.read_csv(SAP_CSV)
    result_rows = []
    for (item, cond), grp in df.groupby(["item", "condition"]):
        sap_row = sap[(sap["item"] == item) & (sap["condition"] == cond)]
        if sap_row.empty:
            continue
        crit_pos = int(sap_row.iloc[0]["disambPosition"])
        sentence = sap_row.iloc[0]["Sentence"]
        words = sentence.split()

        # Build a rough mapping: we need to know which commitment_log position
        # corresponds to which word. In strict-LTR with enforce-prefix,
        # position p corresponds to token position p in full_ids.
        # Token 0 = <eot>, tokens 1..N = sentence tokens.
        # For BPE, word k's first token ≈ position k (for single-token words).
        # We'll use the word index from SAP (1-indexed) and try to match.
        # A simpler approach: for each offset, find the commitment entry whose
        # target_token starts with the word text.
        for offset in OFFSETS:
            wp = crit_pos + offset
            if wp < 1 or wp > len(words):
                continue
            target_word = words[wp - 1]

            # Find the commitment entry for this word position
            # In strict-LTR, word position wp maps to approximately token position wp
            # (since position 0 = <eot>, position 1 = first word token)
            # But BPE can split words. Let's find the entry whose target_token
            # matches the start of the word.
            matched = None
            for _, row in grp.iterrows():
                tt = row.get("target_token", "")
                if tt and tt.strip().lower() == target_word.lower():
                    matched = row
                    break
                elif tt and target_word.lower().startswith(tt.strip().lower()):
                    matched = row
                    break

            if matched is not None:
                result_rows.append({
                    "item": item,
                    "condition": cond,
                    "offset": offset,
                    "word_position": wp,
                    "word": target_word,
                    "steps_taken": matched["steps_taken"],
                    "final_surprisal": matched["final_surprisal"],
                    "final_entropy": matched["final_entropy"],
                    "cumulative_kl": matched["cumulative_kl"],
                    "correct": matched["correct"],
                    "has_metrics": matched["final_surprisal"] is not None and not (
                        isinstance(matched["final_surprisal"], float) and np.isnan(matched["final_surprisal"])
                    ),
                })

    return pd.DataFrame(result_rows)


def load_et_data():
    """Load eye-tracking data for Agreement items."""
    et = pd.read_csv(ET_CSV)
    agree_et = et[et["cond"].isin(["AGREE", "UNAGREE"])].copy()

    sap = pd.read_csv(SAP_CSV)
    crit_map = dict(zip(
        zip(sap["item"], sap["condition"]),
        sap["disambPosition"]
    ))

    rows = []
    for _, row in agree_et.iterrows():
        item = row["item"]
        cond = row["cond"]
        subj = row["subj"]
        crit = crit_map.get((item, cond))
        if crit is None:
            continue

        for offset in OFFSETS:
            wp = crit + offset
            region_idx = wp  # R{wp} columns are 1-indexed
            record = {
                "subj": subj,
                "item": item,
                "condition": cond,
                "offset": offset,
                "word_position": wp,
            }
            for measure in ET_MEASURES:
                col = f"{measure}R{region_idx}"
                if col in row.index:
                    val = row[col]
                    record[measure] = val if pd.notna(val) else np.nan
                else:
                    record[measure] = np.nan
            rows.append(record)

    return pd.DataFrame(rows)


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def main():
    report_lines = []
    def report(line=""):
        print(line)
        report_lines.append(line)

    report_section = lambda title: (
        report(f"\n{'='*70}"),
        report(f"  {title}"),
        report(f"{'='*70}\n"),
    )

    # ── Load data ──
    crit_df = load_critical_results()
    sltr_df = load_strict_ltr_results()
    et_df = load_et_data()

    report_section("1. DATA OVERVIEW")
    report(f"Critical-position results: {len(crit_df)} entries "
           f"({crit_df['item'].nunique()} items × {crit_df['condition'].nunique()} conditions × up to 6 offsets)")
    report(f"Strict-LTR results:       {len(sltr_df)} entries "
           f"({sltr_df['item'].nunique() if len(sltr_df) > 0 else 0} items matched)")
    report(f"Eye-tracking data:        {len(et_df)} entries "
           f"({et_df['subj'].nunique()} subjects × {et_df['item'].nunique()} items)")

    # ── 2. Basic stats comparison ──
    report_section("2. BASIC STATISTICS: CRITICAL-POSITION")

    report(f"{'Offset':>8} {'N':>5} {'Mean Steps':>12} {'Median Steps':>14} {'Accuracy':>10} {'Mean Surp':>12} {'Mean Ent':>10}")
    report("-" * 80)
    for offset in OFFSETS:
        sub = crit_df[crit_df["offset"] == offset]
        n = len(sub)
        if n == 0:
            continue
        steps = sub["steps_taken"]
        acc = sub["correct"].sum() / n * 100
        surp = sub["final_surprisal"].dropna()
        ent = sub["final_entropy"].dropna()
        report(f"{offset:>+8d} {n:>5d} {steps.mean():>12.1f} {steps.median():>14.1f} "
               f"{acc:>9.1f}% {surp.mean():>12.2f} {ent.mean():>10.2f}")

    report(f"\nOverall accuracy: {crit_df['correct'].sum()}/{len(crit_df)} "
           f"({crit_df['correct'].sum()/len(crit_df)*100:.1f}%)")
    report(f"Overall mean steps: {crit_df['steps_taken'].mean():.1f}")
    report(f"Positions with metrics: {crit_df['final_surprisal'].notna().sum()}/{len(crit_df)} "
           f"({crit_df['final_surprisal'].notna().sum()/len(crit_df)*100:.1f}%)")

    # ── 3. Compare with strict-LTR ──
    report_section("3. STRICT-LTR COMPARISON (MATCHED POSITIONS)")

    if len(sltr_df) > 0:
        report(f"{'Offset':>8} {'N_crit':>7} {'N_sltr':>7} {'Crit Steps':>12} {'SLTR Steps':>12} "
               f"{'Crit Acc':>10} {'SLTR Acc':>10} {'SLTR has_met':>13}")
        report("-" * 95)
        for offset in OFFSETS:
            c_sub = crit_df[crit_df["offset"] == offset]
            s_sub = sltr_df[sltr_df["offset"] == offset]
            if len(c_sub) == 0:
                continue
            c_acc = c_sub["correct"].sum() / len(c_sub) * 100 if len(c_sub) > 0 else 0
            s_acc = s_sub["correct"].sum() / len(s_sub) * 100 if len(s_sub) > 0 else 0
            s_has = s_sub["has_metrics"].sum() if "has_metrics" in s_sub.columns and len(s_sub) > 0 else 0
            s_steps = s_sub["steps_taken"].mean() if len(s_sub) > 0 else 0
            report(f"{offset:>+8d} {len(c_sub):>7d} {len(s_sub):>7d} "
                   f"{c_sub['steps_taken'].mean():>12.1f} {s_steps:>12.1f} "
                   f"{c_acc:>9.1f}% {s_acc:>9.1f}% {s_has:>13}")

        report(f"\nKey improvement: Critical-position gives EVERY position a full 1024-step budget.")
        report(f"  Strict-LTR mean steps across offsets: {sltr_df['steps_taken'].mean():.1f}")
        report(f"  Critical-position mean steps:         {crit_df['steps_taken'].mean():.1f}")
        report(f"  Strict-LTR positions with real metrics: "
               f"{sltr_df['has_metrics'].sum() if 'has_metrics' in sltr_df.columns else 'N/A'}/{len(sltr_df)}")
        report(f"  Critical-position positions with metrics: "
               f"{crit_df['final_surprisal'].notna().sum()}/{len(crit_df)}")
    else:
        report("  Could not match strict-LTR positions to offsets.")

    # ── 4. Condition effects ──
    report_section("4. CONDITION EFFECTS (AGREE vs UNAGREE) — CRITICAL-POSITION")

    report(f"{'Offset':>8} {'AGREE steps':>13} {'UNAGREE steps':>15} {'Diff':>8} {'t-stat':>8} {'p-value':>10}")
    report("-" * 75)
    for offset in OFFSETS:
        agree = crit_df[(crit_df["offset"] == offset) & (crit_df["condition"] == "AGREE")]["steps_taken"]
        unagree = crit_df[(crit_df["offset"] == offset) & (crit_df["condition"] == "UNAGREE")]["steps_taken"]
        if len(agree) < 2 or len(unagree) < 2:
            continue
        diff = unagree.mean() - agree.mean()
        t, p = stats.ttest_ind(unagree, agree)
        sig = "*" if p < 0.05 else ("†" if p < 0.1 else "")
        report(f"{offset:>+8d} {agree.mean():>13.1f} {unagree.mean():>15.1f} {diff:>+8.1f} {t:>8.2f} {p:>9.4f} {sig}")

    report("\n  Expected: UNAGREE items should be harder (more steps) at offset 0 (disamb position).")
    report("  Positive diff = UNAGREE takes more steps (expected direction).")

    # Same for surprisal
    report(f"\n{'Offset':>8} {'AGREE surp':>12} {'UNAGREE surp':>14} {'Diff':>8} {'t-stat':>8} {'p-value':>10}")
    report("-" * 75)
    for offset in OFFSETS:
        agree = crit_df[(crit_df["offset"] == offset) & (crit_df["condition"] == "AGREE")]["final_surprisal"].dropna()
        unagree = crit_df[(crit_df["offset"] == offset) & (crit_df["condition"] == "UNAGREE")]["final_surprisal"].dropna()
        if len(agree) < 2 or len(unagree) < 2:
            continue
        diff = unagree.mean() - agree.mean()
        t, p = stats.ttest_ind(unagree, agree)
        sig = "*" if p < 0.05 else ("†" if p < 0.1 else "")
        report(f"{offset:>+8d} {agree.mean():>12.2f} {unagree.mean():>14.2f} {diff:>+8.2f} {t:>8.2f} {p:>9.4f} {sig}")

    # ── 5. Correlation with human eye-tracking ──
    report_section("5. CORRELATION WITH HUMAN EYE-TRACKING DATA")

    # Aggregate ET data by item × condition × offset (mean across subjects)
    et_agg = et_df.groupby(["item", "condition", "offset"])[ET_MEASURES].mean().reset_index()

    merged = crit_df.merge(et_agg, on=["item", "condition", "offset"], how="inner")
    report(f"Merged data: {len(merged)} rows\n")

    if len(merged) > 5:
        sedd_metrics = ["steps_taken", "final_surprisal", "final_entropy", "cumulative_kl"]
        report(f"{'SEDD Metric':>20} {'ET Metric':>10} {'r':>8} {'p':>10} {'N':>5}")
        report("-" * 60)
        for sedd_m in sedd_metrics:
            for et_m in ET_MEASURES:
                valid = merged[[sedd_m, et_m]].dropna()
                if len(valid) < 5:
                    continue
                r, p = stats.pearsonr(valid[sedd_m], valid[et_m])
                sig = "**" if p < 0.01 else ("*" if p < 0.05 else ("†" if p < 0.1 else ""))
                report(f"{sedd_m:>20} {et_m:>10} {r:>8.3f} {p:>10.4f} {len(valid):>5} {sig}")
            report("")

        # Also try per-offset correlations at the critical position (offset=0)
        report("\n  Per-offset=0 correlations (disambiguating word only):")
        report(f"  {'SEDD Metric':>20} {'ET Metric':>10} {'r':>8} {'p':>10} {'N':>5}")
        report("  " + "-" * 58)
        crit0 = merged[merged["offset"] == 0]
        for sedd_m in sedd_metrics:
            for et_m in ET_MEASURES:
                valid = crit0[[sedd_m, et_m]].dropna()
                if len(valid) < 5:
                    continue
                r, p = stats.pearsonr(valid[sedd_m], valid[et_m])
                sig = "**" if p < 0.01 else ("*" if p < 0.05 else ("†" if p < 0.1 else ""))
                report(f"  {sedd_m:>20} {et_m:>10} {r:>8.3f} {p:>10.4f} {len(valid):>5} {sig}")
            report("")

    # ── 6. Steps distribution analysis ──
    report_section("6. STEPS-TO-COMMIT DISTRIBUTION")

    report(f"{'Offset':>8} {'Min':>6} {'Q1':>8} {'Median':>8} {'Q3':>8} {'Max':>6} {'Std':>8}")
    report("-" * 60)
    for offset in OFFSETS:
        sub = crit_df[crit_df["offset"] == offset]["steps_taken"]
        if len(sub) == 0:
            continue
        report(f"{offset:>+8d} {sub.min():>6.0f} {sub.quantile(0.25):>8.0f} {sub.median():>8.0f} "
               f"{sub.quantile(0.75):>8.0f} {sub.max():>6.0f} {sub.std():>8.1f}")

    # ── 7. Summary ──
    report_section("7. SUMMARY & KEY TAKEAWAYS")

    crit_coverage = crit_df["final_surprisal"].notna().sum() / len(crit_df) * 100
    sltr_coverage = (sltr_df["has_metrics"].sum() / len(sltr_df) * 100) if len(sltr_df) > 0 and "has_metrics" in sltr_df.columns else 0

    report(f"1. METRIC COVERAGE: Critical-position achieves {crit_coverage:.0f}% vs strict-LTR's ~{sltr_coverage:.0f}%")
    report(f"   This is the primary improvement — every position gets meaningful metrics.")
    report(f"")
    report(f"2. STEPS VARIATION: Mean={crit_df['steps_taken'].mean():.0f}, "
           f"Std={crit_df['steps_taken'].std():.0f}, "
           f"range [{crit_df['steps_taken'].min()}-{crit_df['steps_taken'].max()}]")
    report(f"   In strict-LTR, later positions were all steps=1 (no variation = no signal).")
    report(f"")
    report(f"3. ACCURACY: {crit_df['correct'].sum()/len(crit_df)*100:.1f}% "
           f"(the model commits to the wrong token most of the time)")
    report(f"   This is expected — steps_taken reflects processing difficulty, not prediction accuracy.")
    report(f"")

    # Check if condition effect at offset 0
    agree_0 = crit_df[(crit_df["offset"] == 0) & (crit_df["condition"] == "AGREE")]["steps_taken"]
    unagree_0 = crit_df[(crit_df["offset"] == 0) & (crit_df["condition"] == "UNAGREE")]["steps_taken"]
    if len(agree_0) >= 2 and len(unagree_0) >= 2:
        t, p = stats.ttest_ind(unagree_0, agree_0)
        diff = unagree_0.mean() - agree_0.mean()
        direction = "expected" if diff > 0 else "UNEXPECTED"
        report(f"4. CONDITION EFFECT at disamb position: UNAGREE-AGREE = {diff:+.1f} steps "
               f"(t={t:.2f}, p={p:.4f}) — {direction} direction")
    report("")

    # Save report
    out_dir = REPO / "LTR_SAP" / "analysis" / "results" / "critical_vs_strict_ltr"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "AGREEMENT_COMPARISON_REPORT.md"
    with open(report_path, "w") as f:
        f.write("# Critical-Position vs Strict-LTR: Agreement Subset\n\n")
        f.write("```\n")
        f.write("\n".join(report_lines))
        f.write("\n```\n")
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
