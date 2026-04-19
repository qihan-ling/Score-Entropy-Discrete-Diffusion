"""
Full analysis of LTR_SAP_critical results.

1. Load all 2055 JSON results across 5 subsets
2. Compute filler conversion factors (OLS regression on SPR data)
3. Per-subset correlation analysis with SPR and eye-tracking
4. Condition effect analysis
5. Generate CRITICAL_POSITION_FULL_REPORT.md

Usage:
  python LTR_SAP_critical/analysis/run_critical_full_analysis.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent.parent
CRITICAL_DIR = REPO / "LTR_SAP_critical"
SAP_DIR = REPO / "SAP_stimuli"
ET_CSV = (
    REPO / "Huang_et_al_2024_spr_osf" / "material & exp_script"
    / "EM_analysis" / "R" / "all_wide.csv"
)
SPR_DIR = REPO / "sapbenchmark" / "analysis" / "spr"
GPT2_DIR = REPO / "sapbenchmark" / "Surprisals" / "data" / "gpt2"
OUTPUT_DIR = CRITICAL_DIR / "analysis" / "results"

SUBSETS = ["Agreement", "ClassicGP", "RelativeClause", "AttachmentAmbiguity"]
OFFSETS = [-2, -1, 0, 1, 2, 3]
ET_MEASURES = ["ffd", "gz", "gp", "tt", "regin", "regout"]

CRIT_COL = {
    "Agreement": "disambPosition",
    "ClassicGP": "disambPosition",
    "RelativeClause": "targetPosition",
    "AttachmentAmbiguity": "disambPosition",
}

SPR_FILE = {
    "Agreement": "AgreementSet.csv",
    "ClassicGP": "ClassicGardenPathSet.csv",
    "RelativeClause": "RelativeClauseSet.csv",
    "AttachmentAmbiguity": "AttachmentSet.csv",
    "filler": "Fillers.csv",
}

GPT2_FILE = {
    "Agreement": "items_Agreement.gpt2.csv.scaled",
    "ClassicGP": "items_ClassicGP.gpt2.csv.scaled",
    "RelativeClause": "items_RelativeClause.gpt2.csv.scaled",
    "AttachmentAmbiguity": "items_AttachmentAmbiguity.gpt2.csv.scaled",
    "filler": "items_filler.gpt2.csv.scaled",
}


# ── Loading ──────────────────────────────────────────────────────────────────

def load_all_critical_results():
    """Load all experimental and filler results into a single DataFrame."""
    rows = []

    for subset in SUBSETS:
        subset_dir = CRITICAL_DIR / subset
        if not subset_dir.exists():
            continue
        sap = pd.read_csv(SAP_DIR / f"sap_items_{subset}.csv")
        crit_col = CRIT_COL[subset]

        for cond_dir in sorted(subset_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            condition = cond_dir.name
            for fpath in sorted(cond_dir.glob("item_*_pos_*.json")):
                try:
                    with open(fpath) as f:
                        d = json.load(f)
                except Exception:
                    continue
                cl = d["commitment_log"]
                fname = fpath.stem
                parts = fname.split("_")
                item_id = int(parts[1])
                offset = int(parts[3])

                sap_row = sap[(sap["item"] == item_id) & (sap["condition"] == condition)]
                crit_pos = int(sap_row.iloc[0][crit_col]) if not sap_row.empty else None

                rows.append({
                    "subset": subset,
                    "item": item_id,
                    "condition": condition,
                    "offset": offset,
                    "word_position": cl.get("word_position"),
                    "crit_position": crit_pos,
                    "word": cl.get("word"),
                    "steps_taken": cl.get("steps_taken"),
                    "final_surprisal": cl.get("final_surprisal"),
                    "final_entropy": cl.get("final_entropy"),
                    "cumulative_kl": cl.get("cumulative_kl"),
                    "correct": cl.get("correct"),
                    "t_commitment": cl.get("t_commitment"),
                    "committed_token": cl.get("committed_token"),
                    "target_token": cl.get("target_token"),
                })

    # Filler
    filler_dir = CRITICAL_DIR / "filler"
    if filler_dir.exists():
        for fpath in sorted(filler_dir.glob("item_*_wpos_*.json")):
            try:
                with open(fpath) as f:
                    d = json.load(f)
            except Exception:
                continue
            cl = d["commitment_log"]
            fname = fpath.stem
            parts = fname.split("_")
            item_id = int(parts[1])
            wpos = int(parts[3])

            rows.append({
                "subset": "filler",
                "item": item_id,
                "condition": None,
                "offset": None,
                "word_position": wpos,
                "crit_position": None,
                "word": cl.get("word"),
                "steps_taken": cl.get("steps_taken"),
                "final_surprisal": cl.get("final_surprisal"),
                "final_entropy": cl.get("final_entropy"),
                "cumulative_kl": cl.get("cumulative_kl"),
                "correct": cl.get("correct"),
                "t_commitment": cl.get("t_commitment"),
                "committed_token": cl.get("committed_token"),
                "target_token": cl.get("target_token"),
            })

    df = pd.DataFrame(rows)
    for col in ["final_surprisal", "final_entropy", "cumulative_kl"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_gpt2_surprisals(subset):
    fname = GPT2_FILE.get(subset)
    if not fname:
        return None
    fpath = GPT2_DIR / fname
    if not fpath.exists():
        return None
    return pd.read_csv(fpath)


def load_spr(subset):
    fname = SPR_FILE.get(subset)
    if not fname:
        return None
    fpath = SPR_DIR / fname
    if not fpath.exists():
        return None
    df = pd.read_csv(fpath)
    if "MD5" in df.columns:
        df["participant"] = df["MD5"]
    return df


# ── Report helpers ───────────────────────────────────────────────────────────

class Report:
    def __init__(self):
        self.lines = []

    def __call__(self, line=""):
        print(line)
        self.lines.append(line)

    def section(self, title):
        self("")
        self(f"{'='*72}")
        self(f"  {title}")
        self(f"{'='*72}")
        self("")

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("# Critical-Position Experiment: Full Analysis Report\n\n")
            f.write("```\n")
            f.write("\n".join(self.lines))
            f.write("\n```\n")


# ── Analyses ─────────────────────────────────────────────────────────────────

def analysis_overview(df, R):
    R.section("1. DATA OVERVIEW")
    R(f"Total results: {len(df)}")
    for subset in SUBSETS + ["filler"]:
        sub = df[df["subset"] == subset]
        if subset == "filler":
            R(f"  {subset}: {len(sub)} entries ({sub['item'].nunique()} items)")
        else:
            R(f"  {subset}: {len(sub)} entries ({sub['item'].nunique()} items, "
              f"conditions: {sorted(sub['condition'].unique())})")
    R("")
    R(f"Overall accuracy: {df['correct'].sum()}/{len(df)} ({df['correct'].sum()/len(df)*100:.1f}%)")
    R(f"Overall mean steps: {df['steps_taken'].mean():.1f} (std={df['steps_taken'].std():.1f})")
    R(f"Positions with surprisal: {df['final_surprisal'].notna().sum()}/{len(df)} "
      f"({df['final_surprisal'].notna().sum()/len(df)*100:.1f}%)")


def analysis_per_subset_stats(df, R):
    R.section("2. PER-SUBSET STATISTICS")
    for subset in SUBSETS + ["filler"]:
        sub = df[df["subset"] == subset]
        if sub.empty:
            continue
        R(f"--- {subset} ---")
        R(f"  N={len(sub)}, items={sub['item'].nunique()}, "
          f"accuracy={sub['correct'].sum()/len(sub)*100:.1f}%")
        R(f"  steps: mean={sub['steps_taken'].mean():.1f}, "
          f"median={sub['steps_taken'].median():.0f}, "
          f"std={sub['steps_taken'].std():.1f}")
        surp = sub["final_surprisal"].dropna()
        R(f"  surprisal: mean={surp.mean():.2f}, std={surp.std():.2f} (N={len(surp)})")
        ent = sub["final_entropy"].dropna()
        R(f"  entropy: mean={ent.mean():.2f}, std={ent.std():.2f}")
        R("")


def analysis_filler_conversion(df, R):
    R.section("3. FILLER CONVERSION FACTORS")

    filler = df[df["subset"] == "filler"].copy()
    if filler.empty:
        R("  No filler data.")
        return {}

    spr = load_spr("filler")
    if spr is None:
        R("  No filler SPR data found.")
        return {}

    gpt2 = load_gpt2_surprisals("filler")

    # Aggregate SPR: mean RT per (item, WordPosition) across participants
    spr_agg = spr.groupby(["item", "WordPosition"])["RT"].mean().reset_index()
    spr_agg.rename(columns={"WordPosition": "word_position", "RT": "mean_RT"}, inplace=True)

    merged = filler.merge(spr_agg, on=["item", "word_position"], how="inner")
    R(f"  Filler x SPR merged: {len(merged)} word-positions")

    # Merge GPT-2 surprisals
    if gpt2 is not None:
        gpt2_sub = gpt2[["item", "word_pos", "sum_surprisal", "logfreq", "length"]].copy()
        gpt2_sub.rename(columns={"word_pos": "word_position",
                                  "sum_surprisal": "gpt2_surprisal"}, inplace=True)
        # GPT-2 word_pos is 0-indexed, filler word_position is 1-indexed
        gpt2_sub["word_position"] = gpt2_sub["word_position"] + 1
        merged = merged.merge(gpt2_sub, on=["item", "word_position"], how="left")
        R(f"  After GPT-2 merge: {len(merged)} rows, "
          f"GPT-2 coverage: {merged['gpt2_surprisal'].notna().sum()}")

    # Add word length from the word column
    merged["word_length"] = merged["word"].str.len()

    # Drop rows with missing values
    cols_needed = ["mean_RT", "steps_taken", "final_surprisal", "word_position", "word_length"]
    clean = merged.dropna(subset=cols_needed).copy()
    R(f"  Clean rows for regression: {len(clean)}")

    if len(clean) < 10:
        R("  Not enough data for regression.")
        return {}

    # Standardize predictors
    for col in ["steps_taken", "final_surprisal", "final_entropy", "cumulative_kl", "word_position", "word_length"]:
        if col in clean.columns:
            s = clean[col].std()
            clean[f"{col}_s"] = (clean[col] - clean[col].mean()) / s if s > 0 else 0

    if "gpt2_surprisal" in clean.columns:
        s = clean["gpt2_surprisal"].std()
        clean["gpt2_surprisal_s"] = (clean["gpt2_surprisal"] - clean["gpt2_surprisal"].mean()) / s if s > 0 else 0

    # OLS regressions
    from numpy.linalg import lstsq

    def ols_report(name, X_cols, y_col="mean_RT"):
        valid = clean.dropna(subset=X_cols + [y_col])
        if len(valid) < 10:
            R(f"  [{name}] Not enough data.")
            return None
        X = valid[X_cols].values
        X = np.column_stack([np.ones(len(X)), X])
        y = valid[y_col].values
        beta, residuals, rank, sv = lstsq(X, y, rcond=None)
        y_pred = X @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        R(f"\n  [{name}] N={len(valid)}, R²={r2:.4f}")
        col_names = ["intercept"] + X_cols
        for i, c in enumerate(col_names):
            R(f"    {c:>25s}: {beta[i]:>10.4f}")
        return {c: beta[i] for i, c in enumerate(col_names)}

    coeffs = {}

    R("\n  --- Model 1: GPT-2 surprisal only ---")
    if "gpt2_surprisal_s" in clean.columns:
        c = ols_report("GPT2", ["gpt2_surprisal_s", "word_position_s", "word_length_s"])
        if c:
            coeffs["gpt2"] = c

    R("\n  --- Model 2: SEDD steps_taken ---")
    c = ols_report("SEDD steps", ["steps_taken_s", "word_position_s", "word_length_s"])
    if c:
        coeffs["steps"] = c

    R("\n  --- Model 3: SEDD final_surprisal ---")
    c = ols_report("SEDD surprisal", ["final_surprisal_s", "word_position_s", "word_length_s"])
    if c:
        coeffs["sedd_surprisal"] = c

    R("\n  --- Model 4: SEDD cumulative_kl ---")
    c = ols_report("SEDD cum_kl", ["cumulative_kl_s", "word_position_s", "word_length_s"])
    if c:
        coeffs["cum_kl"] = c

    R("\n  --- Model 5: SEDD final_entropy ---")
    c = ols_report("SEDD entropy", ["final_entropy_s", "word_position_s", "word_length_s"])
    if c:
        coeffs["entropy"] = c

    R("\n  --- Model 6: Combined (steps + surprisal) ---")
    c = ols_report("Combined", ["steps_taken_s", "final_surprisal_s", "word_position_s", "word_length_s"])
    if c:
        coeffs["combined"] = c

    # Save filler metrics CSV
    out_path = OUTPUT_DIR / "critical_filler_metrics.csv"
    clean.to_csv(out_path, index=False)
    R(f"\n  Saved filler metrics to {out_path.relative_to(REPO)}")

    return coeffs


def analysis_spr_correlation(df, R):
    R.section("4. SPR CORRELATION ANALYSIS (PER SUBSET)")

    sedd_metrics = ["steps_taken", "final_surprisal", "final_entropy", "cumulative_kl"]

    for subset in SUBSETS:
        sub = df[df["subset"] == subset].copy()
        if sub.empty:
            continue

        spr = load_spr(subset)
        gpt2 = load_gpt2_surprisals(subset)
        if spr is None:
            R(f"  [{subset}] No SPR data.")
            continue

        sap = pd.read_csv(SAP_DIR / f"sap_items_{subset}.csv")
        crit_col = CRIT_COL[subset]

        R(f"\n--- {subset} ---")

        # Map SPR ROI to word_position offset relative to critical position
        # SPR ROI=0 means the critical word itself
        # We need to merge by item + condition + ROI offset
        spr_agg = spr.groupby(["item", "AMBUAMB", "ROI"])["RT"].mean().reset_index()
        spr_agg.rename(columns={"ROI": "offset", "RT": "mean_RT"}, inplace=True)

        # Map AMBUAMB to condition names
        # AMBUAMB=1 is the ambiguous/ungrammatical condition
        # Need to figure out the mapping per subset
        # For now, merge by item + offset only (aggregating across conditions in SPR)
        spr_by_item_offset = spr.groupby(["item", "ROI"])["RT"].mean().reset_index()
        spr_by_item_offset.rename(columns={"ROI": "offset", "RT": "mean_RT"}, inplace=True)

        merged = sub.merge(spr_by_item_offset, on=["item", "offset"], how="inner")
        R(f"  Merged (item x offset): {len(merged)} rows")

        if len(merged) < 5:
            R(f"  Not enough merged data.")
            continue

        R(f"  {'SEDD Metric':>20} {'r_pearson':>10} {'p':>10} {'r_spearman':>12} {'p':>10} {'N':>5}")
        R(f"  {'-'*70}")
        for sm in sedd_metrics:
            valid = merged[[sm, "mean_RT"]].dropna()
            if len(valid) < 5:
                continue
            rp, pp = stats.pearsonr(valid[sm], valid["mean_RT"])
            rs, ps = stats.spearmanr(valid[sm], valid["mean_RT"])
            sig = "**" if pp < 0.01 else ("*" if pp < 0.05 else ("†" if pp < 0.1 else ""))
            R(f"  {sm:>20} {rp:>10.3f} {pp:>10.4f} {rs:>12.3f} {ps:>10.4f} {len(valid):>5} {sig}")

        # GPT-2 comparison
        if gpt2 is not None:
            gpt2_sub = gpt2[["item", "condition", "word_pos", "sum_surprisal"]].copy()
            gpt2_sub.rename(columns={"word_pos": "word_position_0idx",
                                      "sum_surprisal": "gpt2_surprisal"}, inplace=True)

            # Map word_pos to offset relative to critical position
            gpt2_with_offset = []
            for _, row in gpt2_sub.iterrows():
                item = row["item"]
                cond = row["condition"]
                wp_0idx = row["word_position_0idx"]
                sap_match = sap[(sap["item"] == item) & (sap["condition"] == cond)]
                if sap_match.empty:
                    continue
                cp = int(sap_match.iloc[0][crit_col])
                offset = (wp_0idx + 1) - cp  # word_pos is 0-indexed, crit_pos is 1-indexed
                gpt2_with_offset.append({
                    "item": item, "condition": cond, "offset": offset,
                    "gpt2_surprisal": row["gpt2_surprisal"],
                })
            gpt2_off = pd.DataFrame(gpt2_with_offset)

            if not gpt2_off.empty:
                gpt2_agg = gpt2_off.groupby(["item", "offset"])["gpt2_surprisal"].mean().reset_index()
                merged_gpt2 = merged.merge(gpt2_agg, on=["item", "offset"], how="left")
                valid_g = merged_gpt2[["gpt2_surprisal", "mean_RT"]].dropna()
                if len(valid_g) >= 5:
                    rp, pp = stats.pearsonr(valid_g["gpt2_surprisal"], valid_g["mean_RT"])
                    R(f"  {'GPT2_surprisal':>20} {rp:>10.3f} {pp:>10.4f} {'':>12} {'':>10} {len(valid_g):>5}")

        # Per-offset breakdown
        R(f"\n  Per-offset correlations (steps_taken vs mean_RT):")
        R(f"  {'Offset':>8} {'r':>8} {'p':>10} {'N':>5}")
        for offset in OFFSETS:
            valid = merged[(merged["offset"] == offset)][["steps_taken", "mean_RT"]].dropna()
            if len(valid) < 4:
                continue
            r, p = stats.pearsonr(valid["steps_taken"], valid["mean_RT"])
            sig = "*" if p < 0.05 else ""
            R(f"  {offset:>+8d} {r:>8.3f} {p:>10.4f} {len(valid):>5} {sig}")

        R(f"\n  Per-offset correlations (final_surprisal vs mean_RT):")
        R(f"  {'Offset':>8} {'r':>8} {'p':>10} {'N':>5}")
        for offset in OFFSETS:
            valid = merged[(merged["offset"] == offset)][["final_surprisal", "mean_RT"]].dropna()
            if len(valid) < 4:
                continue
            r, p = stats.pearsonr(valid["final_surprisal"], valid["mean_RT"])
            sig = "*" if p < 0.05 else ""
            R(f"  {offset:>+8d} {r:>8.3f} {p:>10.4f} {len(valid):>5} {sig}")


def analysis_et_correlation(df, R):
    R.section("5. EYE-TRACKING CORRELATION ANALYSIS (PER SUBSET)")

    sedd_metrics = ["steps_taken", "final_surprisal", "final_entropy", "cumulative_kl"]

    try:
        et = pd.read_csv(ET_CSV)
    except Exception as e:
        R(f"  Could not load eye-tracking data: {e}")
        return

    for subset in SUBSETS:
        sub = df[df["subset"] == subset].copy()
        if sub.empty:
            continue

        sap = pd.read_csv(SAP_DIR / f"sap_items_{subset}.csv")
        crit_col = CRIT_COL[subset]
        conditions = sorted(sub["condition"].unique())

        # Filter ET to relevant conditions
        et_sub = et[et["cond"].isin(conditions)].copy()
        if et_sub.empty:
            R(f"\n--- {subset} ---")
            R(f"  No ET data for conditions {conditions}")
            continue

        # Build critical position map
        crit_map = {}
        for _, row in sap.iterrows():
            crit_map[(row["item"], row["condition"])] = int(row[crit_col])

        # Extract ET at critical offsets
        et_rows = []
        for _, row in et_sub.iterrows():
            item = row["item"]
            cond = row["cond"]
            cp = crit_map.get((item, cond))
            if cp is None:
                continue
            for offset in OFFSETS:
                wp = cp + offset
                record = {"item": item, "condition": cond, "offset": offset}
                for measure in ET_MEASURES:
                    col = f"{measure}R{wp}"
                    record[measure] = row[col] if col in row.index and pd.notna(row[col]) else np.nan
                et_rows.append(record)

        et_df = pd.DataFrame(et_rows)
        et_agg = et_df.groupby(["item", "condition", "offset"])[ET_MEASURES].mean().reset_index()

        merged = sub.merge(et_agg, on=["item", "condition", "offset"], how="inner")

        R(f"\n--- {subset} ---")
        R(f"  Merged (SEDD x ET): {len(merged)} rows")

        if len(merged) < 5:
            continue

        R(f"  {'SEDD Metric':>20} {'ET':>6} {'r':>8} {'p':>10} {'N':>5}")
        R(f"  {'-'*55}")
        for sm in sedd_metrics:
            for em in ["ffd", "gz", "gp", "tt"]:
                valid = merged[[sm, em]].dropna()
                if len(valid) < 5:
                    continue
                r, p = stats.pearsonr(valid[sm], valid[em])
                sig = "**" if p < 0.01 else ("*" if p < 0.05 else ("†" if p < 0.1 else ""))
                R(f"  {sm:>20} {em:>6} {r:>8.3f} {p:>10.4f} {len(valid):>5} {sig}")
            R("")

        # Per-offset at offset=0 (critical word)
        R(f"  Offset=0 only (critical/target word):")
        crit0 = merged[merged["offset"] == 0]
        if len(crit0) >= 4:
            R(f"  {'SEDD Metric':>20} {'ET':>6} {'r':>8} {'p':>10} {'N':>5}")
            R(f"  {'-'*55}")
            for sm in sedd_metrics:
                for em in ["ffd", "gz", "gp", "tt"]:
                    valid = crit0[[sm, em]].dropna()
                    if len(valid) < 4:
                        continue
                    r, p = stats.pearsonr(valid[sm], valid[em])
                    sig = "**" if p < 0.01 else ("*" if p < 0.05 else ("†" if p < 0.1 else ""))
                    R(f"  {sm:>20} {em:>6} {r:>8.3f} {p:>10.4f} {len(valid):>5} {sig}")
                R("")


def analysis_condition_effects(df, R):
    R.section("6. CONDITION EFFECTS (PAIRED TESTS PER SUBSET)")

    sedd_metrics = ["steps_taken", "final_surprisal", "final_entropy", "cumulative_kl"]

    for subset in SUBSETS:
        sub = df[df["subset"] == subset].copy()
        if sub.empty:
            continue

        conditions = sorted(sub["condition"].unique())
        if len(conditions) < 2:
            continue

        R(f"\n--- {subset} (conditions: {conditions}) ---")

        # For subsets with 2 conditions, do pairwise comparison
        # For subsets with 3+ conditions, compare each pair
        cond_pairs = []
        if len(conditions) == 2:
            cond_pairs = [(conditions[0], conditions[1])]
        else:
            cond_pairs = [(conditions[i], conditions[j])
                          for i in range(len(conditions))
                          for j in range(i+1, len(conditions))]

        for c1, c2 in cond_pairs:
            R(f"\n  {c1} vs {c2}:")
            R(f"  {'Metric':>20} {'Offset':>8} {c1+' mean':>12} {c2+' mean':>12} {'Diff':>8} {'t':>8} {'p':>10}")
            R(f"  {'-'*85}")

            for sm in sedd_metrics:
                for offset in OFFSETS:
                    v1 = sub[(sub["condition"] == c1) & (sub["offset"] == offset)][sm].dropna()
                    v2 = sub[(sub["condition"] == c2) & (sub["offset"] == offset)][sm].dropna()
                    if len(v1) < 2 or len(v2) < 2:
                        continue
                    diff = v2.mean() - v1.mean()
                    t, p = stats.ttest_ind(v2, v1)
                    sig = "**" if p < 0.01 else ("*" if p < 0.05 else ("†" if p < 0.1 else ""))
                    R(f"  {sm:>20} {offset:>+8d} {v1.mean():>12.2f} {v2.mean():>12.2f} "
                      f"{diff:>+8.2f} {t:>8.2f} {p:>10.4f} {sig}")
                R("")


def analysis_derived_trajectory_metrics(df, R):
    R.section("7. DERIVED TRAJECTORY METRICS (FROM EXISTING FRONTIER HISTORY)")

    R("  Computing post-hoc trajectory metrics from frontier_history...")
    R("  (Only for experimental subsets with full trajectory data)")

    all_trajectory_rows = []

    for subset in SUBSETS:
        subset_dir = CRITICAL_DIR / subset
        if not subset_dir.exists():
            continue
        for cond_dir in sorted(subset_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            condition = cond_dir.name
            for fpath in sorted(cond_dir.glob("item_*_pos_*.json")):
                try:
                    with open(fpath) as f:
                        d = json.load(f)
                except Exception:
                    continue

                parts = fpath.stem.split("_")
                item_id = int(parts[1])
                offset = int(parts[3])
                hist = d.get("frontier_history", [])
                cl = d["commitment_log"]
                target_tok_id = cl.get("target_token_id")

                if not hist:
                    continue

                entropies = [h.get("entropy", 0) for h in hist]
                kls = [h.get("kl_from_prev", 0) for h in hist]
                p_targets = [h.get("target_prob", h.get("p_target", 0)) or 0 for h in hist]
                argmaxes = [h.get("argmax_id") for h in hist]

                # Belief convergence: first step where p_target > 0.1
                convergence_01 = None
                convergence_05 = None
                for i, pt in enumerate(p_targets):
                    if pt > 0.1 and convergence_01 is None:
                        convergence_01 = hist[i]["step"]
                    if pt > 0.5 and convergence_05 is None:
                        convergence_05 = hist[i]["step"]

                # Entropy cliff: largest single-step entropy drop
                max_entropy_drop = 0
                entropy_cliff_step = None
                for i in range(1, len(entropies)):
                    drop = entropies[i-1] - entropies[i]
                    if drop > max_entropy_drop:
                        max_entropy_drop = drop
                        entropy_cliff_step = hist[i]["step"]

                # Max KL step
                max_kl = 0
                max_kl_step = None
                for i, kl in enumerate(kls):
                    if kl > max_kl:
                        max_kl = kl
                        max_kl_step = hist[i]["step"]

                # Argmax stability: consecutive steps with same argmax before commitment
                argmax_stability = 0
                if argmaxes:
                    last_argmax = argmaxes[-1]
                    count = 0
                    for a in reversed(argmaxes):
                        if a == last_argmax:
                            count += 1
                        else:
                            break
                    argmax_stability = count

                # Top-K overlap: mean Jaccard of top-5 between consecutive steps
                top5_jaccards = []
                for i in range(1, len(hist)):
                    tk_prev = hist[i-1].get("top_k", [])
                    tk_curr = hist[i].get("top_k", [])
                    if tk_prev and tk_curr:
                        ids_prev = set(t[0] if isinstance(t, (list, tuple)) else t.get("id", -1) for t in tk_prev[:5])
                        ids_curr = set(t[0] if isinstance(t, (list, tuple)) else t.get("id", -1) for t in tk_curr[:5])
                        union = ids_prev | ids_curr
                        if union:
                            top5_jaccards.append(len(ids_prev & ids_curr) / len(union))
                mean_jaccard = np.mean(top5_jaccards) if top5_jaccards else None

                all_trajectory_rows.append({
                    "subset": subset,
                    "item": item_id,
                    "condition": condition,
                    "offset": offset,
                    "n_steps": len(hist),
                    "convergence_01": convergence_01,
                    "convergence_05": convergence_05,
                    "entropy_cliff_step": entropy_cliff_step,
                    "max_entropy_drop": max_entropy_drop,
                    "max_kl_step": max_kl_step,
                    "max_kl": max_kl,
                    "argmax_stability": argmax_stability,
                    "mean_top5_jaccard": mean_jaccard,
                    "final_p_target": p_targets[-1] if p_targets else None,
                    "final_entropy": entropies[-1] if entropies else None,
                })

    traj_df = pd.DataFrame(all_trajectory_rows)
    if traj_df.empty:
        R("  No trajectory data computed.")
        return traj_df

    R(f"  Computed trajectory metrics for {len(traj_df)} items")
    R("")

    traj_metrics = ["convergence_01", "convergence_05", "entropy_cliff_step",
                    "max_entropy_drop", "max_kl", "argmax_stability", "mean_top5_jaccard"]

    for subset in SUBSETS:
        sub = traj_df[traj_df["subset"] == subset]
        if sub.empty:
            continue
        R(f"  --- {subset} ---")
        for m in traj_metrics:
            vals = sub[m].dropna()
            if len(vals) == 0:
                continue
            R(f"    {m:>25s}: mean={vals.mean():>8.2f}, median={vals.median():>8.2f}, "
              f"std={vals.std():>8.2f} (N={len(vals)})")
        R("")

    # Save
    out_path = OUTPUT_DIR / "trajectory_metrics.csv"
    traj_df.to_csv(out_path, index=False)
    R(f"  Saved trajectory metrics to {out_path.relative_to(REPO)}")

    return traj_df


def analysis_summary(df, coeffs, R):
    R.section("8. SUMMARY AND KEY TAKEAWAYS")

    R("1. METRIC COVERAGE")
    R(f"   Every position gets a full 1024-step budget: "
      f"{df['final_surprisal'].notna().sum()}/{len(df)} = "
      f"{df['final_surprisal'].notna().sum()/len(df)*100:.0f}% have metrics.")
    R("")

    R("2. ACCURACY")
    R(f"   Overall: {df['correct'].sum()/len(df)*100:.1f}%")
    for subset in SUBSETS:
        sub = df[df["subset"] == subset]
        if not sub.empty:
            R(f"   {subset}: {sub['correct'].sum()/len(sub)*100:.1f}%")
    R("   The model almost never commits the correct token.")
    R("   This means steps_taken reflects stochastic sampling dynamics, not prediction accuracy.")
    R("")

    R("3. CONVERSION FACTORS")
    if coeffs:
        for model_name, coeff_dict in coeffs.items():
            main_pred = [k for k in coeff_dict if k not in ("intercept", "word_position_s", "word_length_s")]
            if main_pred:
                R(f"   {model_name}: {main_pred[0]}={coeff_dict[main_pred[0]]:.4f} ms/SD")
    R("")

    R("4. WHAT WORKS BEST")
    R("   Based on the correlation analyses above, check which SEDD metric")
    R("   correlates most strongly with human reading times across subsets.")
    R("   From the Agreement-only preliminary analysis, final_surprisal dominated.")
    R("   The full analysis tests whether this holds for all subsets.")
    R("")

    R("5. NEXT STEPS")
    R("   - Run bidirectional baseline to establish upper bound")
    R("   - Implement new trajectory metrics (future token tracking, etc.)")
    R("   - Run soft-context experiment with lambda ablation")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    R = Report()

    R("Loading all critical-position results...")
    df = load_all_critical_results()
    R(f"Loaded {len(df)} results.\n")

    analysis_overview(df, R)
    analysis_per_subset_stats(df, R)
    coeffs = analysis_filler_conversion(df, R)
    analysis_spr_correlation(df, R)
    analysis_et_correlation(df, R)
    analysis_condition_effects(df, R)
    traj_df = analysis_derived_trajectory_metrics(df, R)
    analysis_summary(df, coeffs, R)

    report_path = OUTPUT_DIR / "CRITICAL_POSITION_FULL_REPORT.md"
    R.save(report_path)
    R(f"\nReport saved to {report_path.relative_to(REPO)}")

    # Also save master CSV
    master_path = OUTPUT_DIR / "critical_position_master.csv"
    df.to_csv(master_path, index=False)
    R(f"Master CSV saved to {master_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
