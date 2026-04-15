"""
Comprehensive analysis of strict-LTR SEDD results across all SAP subsets.

Performs all analyses without requiring HuggingFace tokenizer downloads:
  1. Summary statistics per subset/condition
  2. Condition effect tests at critical positions
  3. Factor decomposition (steps vs entropy vs position)
  4. Correlation with eye-tracking data
  5. Correlation with GPT-2 surprisal
  6. Trajectory shape analysis (clustering)
  7. Filler word-level analysis for Plan A
  8. Markdown report generation

Usage:
  python LTR_SAP/analysis/run_full_analysis.py
"""

import json
import os
import sys
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAP_STIMULI_DIR = REPO_ROOT / "SAP_stimuli"
LTR_SAP_DIR = REPO_ROOT / "LTR_SAP"
ET_DATA_PATH = (
    REPO_ROOT / "Huang_et_al_2024_spr_osf" / "material & exp_script"
    / "EM_analysis" / "R" / "all_wide.csv"
)
FILLER_ET_PATH = (
    REPO_ROOT / "Huang_et_al_2024_spr_osf" / "material & exp_script"
    / "EM_analysis" / "R" / "filler_wide.csv"
)
GPT2_SURP_DIR = REPO_ROOT / "sapbenchmark" / "Surprisals" / "data" / "gpt2"
SPR_DIR = REPO_ROOT / "sapbenchmark" / "analysis" / "spr"
OUTPUT_DIR = REPO_ROOT / "LTR_SAP" / "analysis" / "results" / "strict_ltr"
FIG_DIR = REPO_ROOT / "LTR_SAP" / "analysis" / "figures"

CRITICAL_POS_COLUMN = {
    "sap_items_Agreement": "disambPosition",
    "sap_items_ClassicGP": "disambPosition",
    "sap_items_AttachmentAmbiguity": "disambPosition",
    "sap_items_RelativeClause": "targetPosition",
    "sap_items_filler": None,
}

ET_MEASURES = ["ffd", "gz", "gp", "tt", "regin", "regout"]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.figsize": (10, 6), "font.size": 12,
    "axes.labelsize": 14, "axes.titlesize": 14,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

CONDITION_COLORS = {
    "AGREE": "#2196F3", "UNAGREE": "#F44336",
    "NPS_AMB": "#FF9800", "NPS_UAMB": "#4CAF50",
    "NPZ_AMB": "#FF9800", "NPZ_UAMB": "#4CAF50",
    "MVRR_AMB": "#FF9800", "MVRR_UAMB": "#4CAF50",
    "RC_Subj": "#2196F3", "RC_Obj": "#F44336",
    "AttachMulti": "#9C27B0", "AttachHigh": "#FF9800", "AttachLow": "#4CAF50",
}

# ============================================================================
# Data loading
# ============================================================================

def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_subset_name(csv_path):
    return Path(csv_path).stem.replace("sap_items_", "")


def simple_token_to_word_map(sentence, commitment_log):
    """Map token positions to word indices using committed/target token text.

    Uses a simple heuristic: tokens starting with space begin a new word.
    Falls back to position-based mapping when possible.
    """
    words = sentence.split()
    word_starts = []
    char_pos = 0
    for i, w in enumerate(words):
        word_starts.append(char_pos)
        char_pos += len(w) + 1

    token_to_word = {}
    word_idx = 0
    current_word_chars = ""

    for entry in commitment_log:
        pos = entry["position"]
        target = entry.get("target_token", "")

        if target.startswith(" ") or target.startswith("Ġ"):
            if current_word_chars:
                word_idx += 1
            current_word_chars = target.lstrip(" Ġ")
        else:
            current_word_chars += target

        token_to_word[pos] = min(word_idx, len(words) - 1)

    return token_to_word


def aggregate_to_words(commitment_log, token_to_word, total_steps=1024):
    """Aggregate token-level metrics to word level.
    
    Positions with final_surprisal=None (never reached by the denoising loop)
    are flagged with has_metrics=False and get NaN for surprisal/entropy/kl.
    """
    by_word = defaultdict(list)
    for entry in commitment_log:
        pos = entry["position"]
        wp = token_to_word.get(pos)
        if wp is not None:
            step = entry["step"]
            t_commit = 1.0 - step / total_steps
            by_word[wp].append({
                **entry,
                "t_commitment": t_commit,
                "weighted_steps": entry["steps_taken"] * (1.0 - t_commit),
            })

    result = []
    for wp in sorted(by_word.keys()):
        entries = by_word[wp]
        any_has_metrics = any(e.get("final_surprisal") is not None for e in entries)
        
        if any_has_metrics:
            surp_vals = [e["final_surprisal"] for e in entries if e.get("final_surprisal") is not None]
            ent_vals = [e["final_entropy"] for e in entries if e.get("final_entropy") is not None]
            kl_vals = [e["cumulative_kl"] for e in entries if e.get("cumulative_kl") is not None]
            surprisal = sum(surp_vals) if surp_vals else np.nan
            entropy = sum(ent_vals) if ent_vals else np.nan
            cumulative_kl = sum(kl_vals) if kl_vals else np.nan
        else:
            surprisal = np.nan
            entropy = np.nan
            cumulative_kl = np.nan

        agg = {
            "word_pos": wp,
            "n_tokens": len(entries),
            "steps_to_commit": sum(e["steps_taken"] for e in entries),
            "weighted_steps": sum(e["weighted_steps"] for e in entries),
            "surprisal": surprisal,
            "entropy": entropy,
            "cumulative_kl": cumulative_kl,
            "correct": all(e.get("correct", False) for e in entries),
            "has_metrics": any_has_metrics,
        }
        result.append(agg)
    return result


def load_all_data(total_steps=1024):
    """Load and process all strict-LTR results into a single DataFrame."""
    all_rows = []

    for csv_path in sorted(SAP_STIMULI_DIR.glob("sap_items_*.csv")):
        subset = get_subset_name(csv_path)
        stem = csv_path.stem
        crit_col = CRITICAL_POS_COLUMN.get(stem)
        df_stim = pd.read_csv(csv_path)
        cond_col = "condition" if "condition" in df_stim.columns else None

        if subset == "filler":
            output_dir = LTR_SAP_DIR / "filler"
            if not output_dir.exists():
                continue
            for jf in sorted(output_dir.glob("item_*.json")):
                out = load_json(jf)
                sentence = out["tokenization"]["sentence"]
                words = sentence.split()
                t2w = simple_token_to_word_map(sentence, out["commitment_log"])
                word_metrics = aggregate_to_words(out["commitment_log"], t2w, total_steps)

                item_match = re.search(r"item_(\d+)", jf.stem)
                item_id = int(item_match.group(1)) if item_match else None

                for wm in word_metrics:
                    wp = wm["word_pos"]
                    wm["item"] = item_id
                    wm["condition"] = None
                    wm["subset"] = "filler"
                    wm["sentence"] = sentence
                    wm["word"] = words[wp] if wp < len(words) else "?"
                    all_rows.append(wm)
        else:
            conditions = df_stim[cond_col].unique().tolist() if cond_col else [None]
            for cond in conditions:
                if cond:
                    cond_dir = LTR_SAP_DIR / subset / cond
                else:
                    cond_dir = LTR_SAP_DIR / subset
                if not cond_dir.exists():
                    continue

                for jf in sorted(cond_dir.glob("item_*.json")):
                    out = load_json(jf)
                    sentence = out["tokenization"]["sentence"]
                    words = sentence.split()

                    matching = df_stim[df_stim["Sentence"] == sentence]
                    if matching.empty:
                        continue

                    item_row = matching.iloc[0]
                    item_id = item_row.get("item", None)
                    crit_pos = int(item_row[crit_col]) if crit_col else None

                    t2w = simple_token_to_word_map(sentence, out["commitment_log"])
                    word_metrics = aggregate_to_words(out["commitment_log"], t2w, total_steps)

                    for wm in word_metrics:
                        wp = wm["word_pos"]
                        wm["item"] = item_id
                        wm["condition"] = cond
                        wm["subset"] = subset
                        wm["sentence"] = sentence
                        wm["word"] = words[wp] if wp < len(words) else "?"
                        if crit_pos is not None:
                            wm["critical_pos"] = crit_pos
                            wm["relative_pos"] = wp - (crit_pos - 1)
                        all_rows.append(wm)

    return pd.DataFrame(all_rows)


def load_token_level_data(total_steps=1024):
    """Load token-level (not word-aggregated) data for factor analysis."""
    all_rows = []

    for csv_path in sorted(SAP_STIMULI_DIR.glob("sap_items_*.csv")):
        subset = get_subset_name(csv_path)
        if subset == "filler":
            continue
        stem = csv_path.stem
        crit_col = CRITICAL_POS_COLUMN.get(stem)
        df_stim = pd.read_csv(csv_path)
        cond_col = "condition" if "condition" in df_stim.columns else None
        conditions = df_stim[cond_col].unique().tolist() if cond_col else [None]

        for cond in conditions:
            cond_dir = LTR_SAP_DIR / subset / (cond or "")
            if not cond_dir.exists():
                continue

            for jf in sorted(cond_dir.glob("item_*.json")):
                out = load_json(jf)
                sentence = out["tokenization"]["sentence"]
                matching = df_stim[df_stim["Sentence"] == sentence]
                if matching.empty:
                    continue
                item_row = matching.iloc[0]
                item_id = item_row.get("item")
                crit_pos = int(item_row[crit_col]) if crit_col else None

                for entry in out["commitment_log"]:
                    step = entry["step"]
                    t_commit = 1.0 - step / total_steps
                    has_metrics = entry.get("final_surprisal") is not None
                    row = {
                        "item": item_id, "condition": cond, "subset": subset,
                        "position": entry["position"],
                        "steps_taken": entry["steps_taken"],
                        "weighted_steps": entry["steps_taken"] * (1.0 - t_commit),
                        "t_commitment": t_commit, "step": step,
                        "final_entropy": entry.get("final_entropy"),
                        "final_surprisal": entry.get("final_surprisal"),
                        "cumulative_kl": entry.get("cumulative_kl"),
                        "committed_token": entry.get("committed_token"),
                        "target_token": entry.get("target_token"),
                        "correct": entry.get("correct"),
                        "has_metrics": has_metrics,
                    }
                    if crit_pos is not None:
                        row["critical_pos"] = crit_pos
                        row["relative_pos"] = entry["position"] - crit_pos
                    all_rows.append(row)

    return pd.DataFrame(all_rows)


# ============================================================================
# Analysis functions
# ============================================================================

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std == 0:
        return 0
    return (group1.mean() - group2.mean()) / pooled_std


def analysis_summary_stats(df, report):
    """Overall summary statistics."""
    report.append("## 1. Data Overview\n")

    for subset in ["Agreement", "ClassicGP", "RelativeClause", "AttachmentAmbiguity", "filler"]:
        sub = df[df["subset"] == subset]
        if sub.empty:
            continue
        n_items = sub["item"].nunique()
        conditions = sub["condition"].dropna().unique().tolist()
        n_words = len(sub)
        n_with_metrics = sub["has_metrics"].sum() if "has_metrics" in sub.columns else n_words
        n_none = n_words - n_with_metrics
        report.append(f"### {subset}")
        report.append(f"- Items: {n_items}, Conditions: {conditions or 'N/A'}, Word-level entries: {n_words}")
        report.append(f"- **Words with actual denoising metrics: {n_with_metrics}/{n_words}** "
                      f"({n_none} words had None — never reached by denoising loop)")

        tracked = sub[sub["has_metrics"] == True] if "has_metrics" in sub.columns else sub
        report.append(f"- Steps-to-commit (all): mean={sub['steps_to_commit'].mean():.1f}, "
                      f"sd={sub['steps_to_commit'].std():.1f}, "
                      f"range=[{sub['steps_to_commit'].min():.0f}, {sub['steps_to_commit'].max():.0f}]")
        report.append(f"- Weighted steps (all): mean={sub['weighted_steps'].mean():.1f}, sd={sub['weighted_steps'].std():.1f}")
        surp_valid = tracked["surprisal"].dropna()
        ent_valid = tracked["entropy"].dropna()
        kl_valid = tracked["cumulative_kl"].dropna()
        if len(surp_valid) > 0:
            report.append(f"- Surprisal (tracked only, n={len(surp_valid)}): mean={surp_valid.mean():.2f}, sd={surp_valid.std():.2f}")
        if len(ent_valid) > 0:
            report.append(f"- Entropy (tracked only, n={len(ent_valid)}): mean={ent_valid.mean():.2f}, sd={ent_valid.std():.2f}")
        if len(kl_valid) > 0:
            report.append(f"- Cumulative KL (tracked only, n={len(kl_valid)}): mean={kl_valid.mean():.4f}, sd={kl_valid.std():.4f}")
        if "correct" in sub.columns:
            correct_rate = sub["correct"].mean() * 100
            report.append(f"- Token prediction accuracy: {correct_rate:.1f}%")
        report.append("")

    total = df[df["subset"] != "filler"]
    total_tracked = total[total["has_metrics"] == True] if "has_metrics" in total.columns else total
    report.append(f"**Total experimental words:** {len(total)} ({len(total_tracked)} with metrics)")
    report.append(f"**Total filler words:** {len(df[df['subset'] == 'filler'])}")
    report.append("")


def analysis_condition_effects(df, report, output_dir):
    """Test condition effects at the critical position."""
    report.append("## 2. Condition Effects at Critical Position\n")

    results_all = []
    metrics = ["steps_to_commit", "weighted_steps", "surprisal", "entropy", "cumulative_kl"]

    for subset in ["Agreement", "ClassicGP", "RelativeClause", "AttachmentAmbiguity"]:
        sub = df[df["subset"] == subset]
        if "relative_pos" not in sub.columns:
            continue
        sub = sub[sub["relative_pos"] == 0]
        if sub.empty:
            continue

        conditions = sorted(sub["condition"].dropna().unique())
        if len(conditions) < 2:
            continue

        n_total = len(sub)
        n_with = sub["has_metrics"].sum() if "has_metrics" in sub.columns else n_total
        n_none = n_total - n_with
        report.append(f"### {subset} ({' vs '.join(conditions[:2])})\n")
        if n_none > 0:
            report.append(f"**WARNING: {n_none}/{n_total} items at critical position have NO denoising "
                          f"metrics (None) — the loop ran out of steps before reaching them.**\n")
        report.append(f"| Metric | {conditions[0]} | {conditions[1]} | diff | Cohen's d | t-stat | p-value |")
        report.append("|--------|-------|-------|------|-----------|--------|---------|")

        for metric in metrics:
            if metric not in sub.columns:
                continue
            g1 = sub[sub["condition"] == conditions[0]][metric].dropna()
            g2 = sub[sub["condition"] == conditions[1]][metric].dropna()
            if len(g1) < 2 or len(g2) < 2:
                continue

            t_stat, p_val = stats.ttest_ind(g1, g2)
            d = cohens_d(g1, g2)
            diff = g1.mean() - g2.mean()

            sig = ""
            if p_val < 0.001: sig = "***"
            elif p_val < 0.01: sig = "**"
            elif p_val < 0.05: sig = "*"

            report.append(f"| {metric} | {g1.mean():.2f} ({g1.std():.2f}) | "
                          f"{g2.mean():.2f} ({g2.std():.2f}) | {diff:+.2f} | {d:.3f} | {t_stat:.2f} | {p_val:.4f}{sig} |")

            results_all.append({
                "subset": subset, "metric": metric,
                "cond_a": conditions[0], "cond_b": conditions[1],
                "mean_a": g1.mean(), "mean_b": g2.mean(),
                "diff": diff, "cohens_d": d, "t_stat": t_stat, "p_value": p_val,
            })

        if len(conditions) > 2:
            report.append(f"\n*Additional conditions: {conditions[2:]}*")

        report.append("")

    if results_all:
        pd.DataFrame(results_all).to_csv(output_dir / "condition_effects.csv", index=False)

    return results_all


def analysis_effect_direction_profile(df, report, output_dir):
    """Effect direction profile across positions relative to critical."""
    report.append("## 3. Effect Direction Profile (ROI 0 to +2)\n")

    for subset in ["Agreement", "ClassicGP", "RelativeClause", "AttachmentAmbiguity"]:
        sub = df[df["subset"] == subset]
        if "relative_pos" not in sub.columns:
            continue
        conditions = sorted(sub["condition"].dropna().unique())
        if len(conditions) < 2:
            continue

        report.append(f"### {subset}\n")
        report.append(f"| ROI | Metric | {conditions[0]} mean | {conditions[1]} mean | diff | p-value |")
        report.append("|-----|--------|------|------|------|---------|")

        for roi in [-1, 0, 1, 2]:
            roi_data = sub[sub["relative_pos"] == roi]
            for metric in ["steps_to_commit", "weighted_steps", "surprisal"]:
                if metric not in roi_data.columns:
                    continue
                g1 = roi_data[roi_data["condition"] == conditions[0]][metric].dropna()
                g2 = roi_data[roi_data["condition"] == conditions[1]][metric].dropna()
                if len(g1) < 2 or len(g2) < 2:
                    continue
                _, p = stats.ttest_ind(g1, g2)
                diff = g1.mean() - g2.mean()
                sig = "*" if p < 0.05 else ""
                report.append(f"| {roi:+d} | {metric} | {g1.mean():.2f} | {g2.mean():.2f} | {diff:+.2f} | {p:.4f}{sig} |")

        report.append("")


def analysis_factor_decomposition(token_df, report, output_dir):
    """Factor decomposition: correlations of steps with entropy and position."""
    report.append("## 4. Factor Decomposition\n")
    report.append("Gate (dsigma) is constant with --steps 1024. We examine two variable factors:\n")
    report.append("- **Score sharpness**: proxied by `final_entropy`")
    report.append("- **Context quality**: proxied by `position` (linear in enforce-prefix LTR)\n")

    results_all = []
    for subset in ["Agreement", "ClassicGP", "RelativeClause", "AttachmentAmbiguity"]:
        sub = token_df[token_df["subset"] == subset].copy()
        if sub.empty:
            continue
        if "has_metrics" in sub.columns:
            sub = sub[sub["has_metrics"] == True]
        valid = sub.dropna(subset=["steps_taken", "final_entropy", "position"])
        if len(valid) < 5:
            continue

        report.append(f"### {subset} (n={len(valid)} tokens)\n")
        report.append("| Target | Factor | Spearman rho | p-value | Pearson r | p-value |")
        report.append("|--------|--------|-------------|---------|-----------|---------|")

        for target in ["steps_taken", "weighted_steps"]:
            for factor in ["final_entropy", "position"]:
                clean = valid[[target, factor]].dropna()
                if len(clean) < 5:
                    continue
                rho, p_rho = stats.spearmanr(clean[target], clean[factor])
                r, p_r = stats.pearsonr(clean[target], clean[factor])
                report.append(f"| {target} | {factor} | {rho:.3f} | {p_rho:.4f} | {r:.3f} | {p_r:.4f} |")
                results_all.append({
                    "subset": subset, "target": target, "factor": factor,
                    "spearman_rho": rho, "p_spearman": p_rho,
                    "pearson_r": r, "p_pearson": p_r, "n": len(clean),
                })

        report.append("")

        # Prediction accuracy
        accuracy = valid["correct"].mean() * 100 if "correct" in valid.columns else 0
        report.append(f"- Token prediction accuracy: {accuracy:.1f}%")

        # Condition comparison at critical position
        if "relative_pos" in valid.columns:
            crit = valid[valid["relative_pos"] == 0]
            conditions = sorted(crit["condition"].dropna().unique())
            if len(conditions) >= 2:
                report.append(f"- At critical position:")
                for factor in ["steps_taken", "weighted_steps", "final_entropy", "t_commitment"]:
                    if factor not in crit.columns:
                        continue
                    g1 = crit[crit["condition"] == conditions[0]][factor].dropna()
                    g2 = crit[crit["condition"] == conditions[1]][factor].dropna()
                    if len(g1) > 0 and len(g2) > 0:
                        report.append(f"  - {factor}: {conditions[0]}={g1.mean():.2f}, "
                                      f"{conditions[1]}={g2.mean():.2f}")
        report.append("")

    if results_all:
        pd.DataFrame(results_all).to_csv(output_dir / "factor_correlations.csv", index=False)

    # Generate scatter plots
    fig_dir = FIG_DIR / "factor_decomposition"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for subset in ["Agreement", "ClassicGP", "RelativeClause", "AttachmentAmbiguity"]:
        sub = token_df[token_df["subset"] == subset].dropna(subset=["steps_taken", "final_entropy", "position"])
        if len(sub) < 5:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for col, target in enumerate(["steps_taken", "weighted_steps"]):
            for row, factor in enumerate(["final_entropy", "position"]):
                ax = axes[row, col]
                ax.scatter(sub[factor], sub[target], alpha=0.3, s=10)
                rho, _ = stats.spearmanr(sub[target], sub[factor])
                ax.set_xlabel(factor)
                ax.set_ylabel(target)
                ax.set_title(f"rho={rho:.3f}")
        plt.suptitle(f"Factor Decomposition: {subset}")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{subset}_factors.png")
        plt.close()


def analysis_et_correlations(df, report, output_dir):
    """Correlations with eye-tracking data."""
    report.append("## 5. Correlations with Eye-Tracking Data\n")

    if not ET_DATA_PATH.exists():
        report.append("*Eye-tracking data not found.*\n")
        return

    et_df = pd.read_csv(ET_DATA_PATH)
    report.append(f"ET data loaded: {len(et_df)} rows, {et_df['item'].nunique()} items\n")

    results_all = []
    for subset in ["Agreement", "ClassicGP", "RelativeClause", "AttachmentAmbiguity"]:
        sub = df[(df["subset"] == subset)]
        if "relative_pos" not in sub.columns:
            continue
        crit = sub[sub["relative_pos"] == 0].copy()
        if crit.empty:
            continue

        conditions = sorted(crit["condition"].dropna().unique())
        report.append(f"### {subset}\n")

        found_any = False
        for cond in conditions:
            cond_sedd = crit[crit["condition"] == cond]
            cond_et = et_df[et_df["cond"] == cond]
            if cond_et.empty:
                continue

            for sedd_m in ["steps_to_commit", "weighted_steps", "surprisal", "entropy"]:
                if sedd_m not in cond_sedd.columns:
                    continue
                for et_m in ET_MEASURES:
                    merged_items = []
                    for _, srow in cond_sedd.iterrows():
                        item_id = srow["item"]
                        crit_pos = srow.get("critical_pos")
                        if crit_pos is None or pd.isna(crit_pos):
                            continue
                        et_col = f"{et_m}R{int(crit_pos)}"
                        if et_col not in cond_et.columns:
                            continue
                        et_vals = cond_et[cond_et["item"] == item_id][et_col].dropna()
                        if et_vals.empty:
                            continue
                        merged_items.append({
                            "item": item_id,
                            "sedd_val": srow[sedd_m],
                            "et_val": et_vals.mean(),
                        })

                    if len(merged_items) >= 5:
                        mdf = pd.DataFrame(merged_items)
                        rho, p = stats.spearmanr(mdf["sedd_val"], mdf["et_val"])
                        results_all.append({
                            "subset": subset, "condition": cond,
                            "sedd_metric": sedd_m, "et_metric": et_m,
                            "spearman_rho": rho, "p_value": p, "n": len(merged_items),
                        })
                        if abs(rho) > 0.3:
                            found_any = True

        if not found_any:
            report.append("*No correlations with |rho| > 0.3 found at item level.*\n")

    if results_all:
        res_df = pd.DataFrame(results_all)
        res_df.to_csv(output_dir / "et_correlations.csv", index=False)

        report.append("### Summary of notable correlations (|rho| > 0.3)\n")
        notable = res_df[res_df["spearman_rho"].abs() > 0.3]
        if notable.empty:
            report.append("*No item-level correlations with |rho| > 0.3 found.*\n")
        else:
            report.append("| Subset | Condition | SEDD Metric | ET Metric | rho | p | n |")
            report.append("|--------|-----------|-------------|-----------|-----|---|---|")
            for _, r in notable.iterrows():
                report.append(f"| {r['subset']} | {r['condition']} | {r['sedd_metric']} | "
                              f"{r['et_metric']} | {r['spearman_rho']:.3f} | {r['p_value']:.4f} | {r['n']} |")
            report.append("")

        # Aggregate: which SEDD metric has highest mean |rho| across all comparisons?
        if len(res_df) > 0:
            report.append("### Mean |rho| by SEDD metric (across all subsets/conditions/ET measures)\n")
            summary = res_df.groupby("sedd_metric")["spearman_rho"].agg(
                mean_abs_rho=lambda x: x.abs().mean(),
                mean_rho="mean",
                n_comparisons="count"
            ).reset_index()
            report.append("| SEDD Metric | Mean |rho| | Mean rho | N comparisons |")
            report.append("|-------------|------------|----------|---------------|")
            for _, r in summary.iterrows():
                report.append(f"| {r['sedd_metric']} | {r['mean_abs_rho']:.4f} | {r['mean_rho']:.4f} | {int(r['n_comparisons'])} |")
            report.append("")


def analysis_filler_word_level(df, report, output_dir):
    """Plan A: steps vs surprisal on filler + all data."""
    report.append("## 6. Steps-to-Commit vs Surprisal (Plan A)\n")

    has_m = df["has_metrics"] == True if "has_metrics" in df.columns else pd.Series(True, index=df.index)
    combined = df[has_m].dropna(subset=["steps_to_commit", "surprisal"]).copy()
    combined = combined[combined["surprisal"] > 0]
    if len(combined) < 10:
        report.append("*Insufficient data.*\n")
        return

    combined["steps_z"] = stats.zscore(combined["steps_to_commit"])
    combined["surprisal_z"] = stats.zscore(combined["surprisal"])

    r, p_r = stats.pearsonr(combined["steps_z"], combined["surprisal_z"])
    rho, p_rho = stats.spearmanr(combined["steps_z"], combined["surprisal_z"])

    report.append(f"- **All data** (n={len(combined)}):")
    report.append(f"  - Pearson r = {r:.3f} (p = {p_r:.2e})")
    report.append(f"  - Spearman rho = {rho:.3f} (p = {p_rho:.2e})")

    # Per-subset
    report.append("\n### Per-subset correlations\n")
    report.append("| Subset | n | Pearson r | Spearman rho |")
    report.append("|--------|---|-----------|--------------|")
    for subset in ["Agreement", "ClassicGP", "RelativeClause", "AttachmentAmbiguity", "filler"]:
        sub = combined[combined["subset"] == subset]
        if len(sub) < 5:
            continue
        r_sub, _ = stats.pearsonr(sub["steps_to_commit"], sub["surprisal"])
        rho_sub, _ = stats.spearmanr(sub["steps_to_commit"], sub["surprisal"])
        report.append(f"| {subset} | {len(sub)} | {r_sub:.3f} | {rho_sub:.3f} |")
    report.append("")

    # Weighted steps vs surprisal
    r_w, p_w = stats.pearsonr(combined["weighted_steps"], combined["surprisal"])
    rho_w, p_rw = stats.spearmanr(combined["weighted_steps"], combined["surprisal"])
    report.append(f"- **Weighted steps vs surprisal**: r={r_w:.3f}, rho={rho_w:.3f}\n")

    # Steps vs KL
    kl_valid = combined.dropna(subset=["cumulative_kl"])
    if len(kl_valid) > 10:
        r_kl, _ = stats.pearsonr(kl_valid["surprisal"], kl_valid["cumulative_kl"])
        report.append(f"- **Cumulative KL vs surprisal**: r={r_kl:.3f}\n")

    # Scatter plot
    fig_dir = FIG_DIR / "plan_a"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1: Steps vs surprisal
    ax = axes[0]
    for subset in combined["subset"].unique():
        sub = combined[combined["subset"] == subset]
        ax.scatter(sub["surprisal_z"], sub["steps_z"], alpha=0.3, s=8, label=subset)
    slope, intercept = np.polyfit(combined["surprisal_z"], combined["steps_z"], 1)
    x_line = np.linspace(combined["surprisal_z"].min(), combined["surprisal_z"].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=1.5)
    ax.set_xlabel("Surprisal (z)")
    ax.set_ylabel("Steps-to-commit (z)")
    ax.set_title(f"Steps vs Surprisal (r={r:.3f})")
    ax.legend(fontsize=8)

    # 2: Weighted steps vs surprisal
    ax = axes[1]
    ws_z = stats.zscore(combined["weighted_steps"])
    ax.scatter(combined["surprisal_z"], ws_z, alpha=0.3, s=8, c="#FF9800")
    ax.set_xlabel("Surprisal (z)")
    ax.set_ylabel("Weighted steps (z)")
    ax.set_title(f"Weighted Steps vs Surprisal (r={r_w:.3f})")

    # 3: KL vs surprisal
    ax = axes[2]
    if len(kl_valid) > 10:
        ax.scatter(kl_valid["surprisal"], kl_valid["cumulative_kl"], alpha=0.3, s=8, c="#4CAF50")
        ax.set_xlabel("Surprisal (bits)")
        ax.set_ylabel("Cumulative KL (bits)")
        ax.set_title(f"KL vs Surprisal (r={r_kl:.3f})")

    plt.tight_layout()
    plt.savefig(fig_dir / "steps_vs_surprisal.png")
    plt.close()
    report.append(f"*Scatter plots saved to {fig_dir}*\n")

    combined.to_csv(output_dir / "word_level_metrics.csv", index=False)


def analysis_trajectory_clustering(report, output_dir):
    """Plan B: trajectory shape clustering."""
    report.append("## 7. Trajectory Shape Typology (Plan B)\n")

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    all_features = []
    for csv_path in sorted(SAP_STIMULI_DIR.glob("sap_items_*.csv")):
        subset = get_subset_name(csv_path)
        df_stim = pd.read_csv(csv_path)
        cond_col = "condition" if "condition" in df_stim.columns else None
        conditions = df_stim[cond_col].unique().tolist() if cond_col else [None]

        for cond in conditions:
            if cond and subset != "filler":
                cond_dir = LTR_SAP_DIR / subset / cond
            elif subset == "filler":
                cond_dir = LTR_SAP_DIR / "filler"
            else:
                cond_dir = LTR_SAP_DIR / subset
            if not cond_dir.exists():
                continue

            for jf in sorted(cond_dir.glob("item_*.json")):
                out = load_json(jf)
                frontier_history = out.get("frontier_history", {})
                for entry in out["commitment_log"]:
                    pos = entry["position"]
                    hist = frontier_history.get(str(pos), [])
                    if not hist:
                        continue

                    entropies = [h.get("entropy", 0) for h in hist]
                    steps_taken = entry["steps_taken"]
                    final_entropy = entropies[-1] if entropies else 0

                    max_ent = max(entropies) if entropies else 0
                    if max_ent > 0:
                        plateau = sum(1 for e in entropies if e >= 0.9 * max_ent) / len(entropies)
                    else:
                        plateau = 0

                    if len(entropies) >= 2:
                        slope = np.polyfit(range(len(entropies)), entropies, 1)[0]
                    else:
                        slope = 0

                    kls = [h.get("cumulative_kl", 0) for h in hist]
                    cum_kl = kls[-1] if kls else 0

                    all_features.append({
                        "subset": subset, "condition": cond,
                        "steps_taken": steps_taken,
                        "plateau_duration": plateau,
                        "entropy_slope": slope,
                        "final_entropy": final_entropy,
                        "cumulative_kl": cum_kl,
                        "correct": entry.get("correct"),
                    })

            if subset == "filler":
                break

    if not all_features:
        report.append("*No trajectory data found.*\n")
        return

    feat_df = pd.DataFrame(all_features)
    report.append(f"Collected {len(feat_df)} token-level trajectory features\n")

    feature_cols = ["steps_taken", "plateau_duration", "entropy_slope", "final_entropy", "cumulative_kl"]
    X = feat_df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    feat_df["cluster"] = kmeans.fit_predict(X_scaled)

    report.append("### Cluster profiles\n")
    report.append("| Cluster | n | steps_taken | plateau | slope | entropy | cum_kl |")
    report.append("|---------|---|-------------|---------|-------|---------|--------|")
    for cl in range(n_clusters):
        mask = feat_df["cluster"] == cl
        n = mask.sum()
        means = feat_df.loc[mask, feature_cols].mean()
        report.append(f"| {cl} | {n} | {means['steps_taken']:.1f} | {means['plateau_duration']:.2f} | "
                      f"{means['entropy_slope']:.4f} | {means['final_entropy']:.2f} | {means['cumulative_kl']:.4f} |")

    report.append("\n### Cluster distribution by subset\n")
    ct = pd.crosstab(feat_df["subset"], feat_df["cluster"], normalize="index")
    report.append("| Subset | " + " | ".join([f"Cluster {c}" for c in range(n_clusters)]) + " |")
    report.append("|--------|" + "|".join(["------" for _ in range(n_clusters)]) + "|")
    for subset in ct.index:
        vals = " | ".join([f"{ct.loc[subset, c]:.2%}" for c in range(n_clusters)])
        report.append(f"| {subset} | {vals} |")
    report.append("")

    # PCA plot
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig_dir = FIG_DIR / "plan_b"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))
    for cl in range(n_clusters):
        mask = feat_df["cluster"] == cl
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.4, s=15, color=colors[cl],
                   label=f"Cluster {cl} (n={mask.sum()})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Denoising Trajectory Typology (PCA)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "trajectory_clusters_pca.png")
    plt.close()

    feat_df.to_csv(output_dir / "trajectory_features.csv", index=False)


def analysis_entropy_profiles(df, report, output_dir):
    """Entropy profiles around critical position per condition."""
    report.append("## 8. Entropy Profiles Around Critical Position\n")

    fig_dir = FIG_DIR / "plan_a"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for subset in ["Agreement", "ClassicGP", "RelativeClause", "AttachmentAmbiguity"]:
        sub = df[(df["subset"] == subset)]
        if "relative_pos" not in sub.columns:
            continue
        conditions = sorted(sub["condition"].dropna().unique())
        window = sub[(sub["relative_pos"] >= -3) & (sub["relative_pos"] <= 3)]
        if window.empty:
            continue

        report.append(f"### {subset}\n")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, metric in zip(axes, ["steps_to_commit", "weighted_steps", "entropy"]):
            if metric not in window.columns:
                continue
            for cond in conditions:
                cond_data = window[window["condition"] == cond]
                means = cond_data.groupby("relative_pos")[metric].mean()
                sems = cond_data.groupby("relative_pos")[metric].sem()
                color = CONDITION_COLORS.get(cond)
                ax.errorbar(means.index, means.values, yerr=sems.values,
                            label=cond, marker="o", color=color, capsize=3)
            ax.set_xlabel("Position relative to critical")
            ax.set_ylabel(metric)
            ax.legend(fontsize=8)
            ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        plt.suptitle(f"Position Profile: {subset}")
        plt.tight_layout()
        plt.savefig(fig_dir / f"position_profile_{subset}.png")
        plt.close()

        # Report key differences at ROI 0
        for metric in ["steps_to_commit", "weighted_steps", "entropy"]:
            roi0 = window[window["relative_pos"] == 0]
            if len(conditions) >= 2:
                g1 = roi0[roi0["condition"] == conditions[0]][metric].dropna()
                g2 = roi0[roi0["condition"] == conditions[1]][metric].dropna()
                if len(g1) > 0 and len(g2) > 0:
                    diff = g1.mean() - g2.mean()
                    report.append(f"- {metric} at ROI=0: {conditions[0]}={g1.mean():.2f}, "
                                  f"{conditions[1]}={g2.mean():.2f} (diff={diff:+.2f})")
        report.append("")


def analysis_gpt2_comparison(df, report, output_dir):
    """Compare SEDD metrics with GPT-2 surprisal."""
    report.append("## 9. SEDD vs GPT-2 Surprisal\n")

    if not GPT2_SURP_DIR.exists():
        report.append("*GPT-2 surprisal data not found.*\n")
        return

    results = []
    for subset in ["Agreement", "ClassicGP", "RelativeClause", "AttachmentAmbiguity", "filler"]:
        gpt2_fname = f"items_{subset}.gpt2.csv.scaled"
        gpt2_path = GPT2_SURP_DIR / gpt2_fname
        if not gpt2_path.exists():
            continue

        gpt2 = pd.read_csv(gpt2_path)
        sedd_sub = df[df["subset"] == subset].copy()

        if "Sentence" in gpt2.columns and "word_pos" in gpt2.columns:
            sedd_sub["word_pos_0"] = sedd_sub["word_pos"]
            merged = pd.merge(
                sedd_sub, gpt2[["Sentence", "word_pos", "sum_surprisal"]].rename(
                    columns={"sum_surprisal": "gpt2_surprisal", "word_pos": "word_pos_0"}),
                left_on=["sentence", "word_pos_0"],
                right_on=["Sentence", "word_pos_0"],
                how="inner",
            )

            if len(merged) > 10:
                for sedd_m in ["steps_to_commit", "weighted_steps", "surprisal", "entropy"]:
                    if sedd_m not in merged.columns:
                        continue
                    valid = merged.dropna(subset=[sedd_m, "gpt2_surprisal"])
                    if len(valid) < 5:
                        continue
                    rho, p = stats.spearmanr(valid[sedd_m], valid["gpt2_surprisal"])
                    r, p_r = stats.pearsonr(valid[sedd_m], valid["gpt2_surprisal"])
                    results.append({
                        "subset": subset, "sedd_metric": sedd_m,
                        "spearman_rho": rho, "p_spearman": p,
                        "pearson_r": r, "p_pearson": p_r, "n": len(valid),
                    })

    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_dir / "sedd_vs_gpt2.csv", index=False)

        report.append("| Subset | SEDD Metric | Spearman rho | Pearson r | n |")
        report.append("|--------|-------------|-------------|-----------|---|")
        for _, r in res_df.iterrows():
            report.append(f"| {r['subset']} | {r['sedd_metric']} | {r['spearman_rho']:.3f} | {r['pearson_r']:.3f} | {int(r['n'])} |")
        report.append("")

        # Summary
        summary = res_df.groupby("sedd_metric").agg(
            mean_rho=("spearman_rho", "mean"),
            mean_r=("pearson_r", "mean"),
        ).reset_index()
        report.append("### Average correlation with GPT-2 surprisal\n")
        report.append("| SEDD Metric | Mean Spearman | Mean Pearson |")
        report.append("|-------------|---------------|--------------|")
        for _, r in summary.iterrows():
            report.append(f"| {r['sedd_metric']} | {r['mean_rho']:.3f} | {r['mean_r']:.3f} |")
        report.append("")


def analysis_filler_regression_prep(df, report, output_dir):
    """Prepare filler data for the R regression script and report statistics."""
    report.append("## 10. Filler Analysis for Conversion Model\n")

    filler = df[df["subset"] == "filler"].copy()
    if filler.empty:
        report.append("*No filler data available.*\n")
        return

    n_with = filler["has_metrics"].sum() if "has_metrics" in filler.columns else len(filler)
    report.append(f"- Filler word-level entries: {len(filler)}")
    report.append(f"- Words with denoising metrics: {n_with}/{len(filler)}")
    report.append(f"- Filler items: {filler['item'].nunique()}")
    report.append(f"- Steps-to-commit: mean={filler['steps_to_commit'].mean():.1f}, sd={filler['steps_to_commit'].std():.1f}")
    surp_valid = filler["surprisal"].dropna()
    ent_valid = filler["entropy"].dropna()
    if len(surp_valid) > 0:
        report.append(f"- Surprisal (tracked only, n={len(surp_valid)}): mean={surp_valid.mean():.2f}, sd={surp_valid.std():.2f}")
    if len(ent_valid) > 0:
        report.append(f"- Entropy (tracked only, n={len(ent_valid)}): mean={ent_valid.mean():.2f}, sd={ent_valid.std():.2f}")
    report.append("")

    # Correlation between filler steps and surprisal
    valid = filler.dropna(subset=["steps_to_commit", "surprisal"])
    valid = valid[valid["surprisal"] > 0]
    if len(valid) > 5:
        r, p = stats.pearsonr(valid["steps_to_commit"], valid["surprisal"])
        rho, p_rho = stats.spearmanr(valid["steps_to_commit"], valid["surprisal"])
        report.append(f"- **Filler steps vs surprisal**: r={r:.3f}, rho={rho:.3f}")

    # Save filler metrics for R script
    filler.to_csv(output_dir / "filler_word_metrics.csv", index=False)
    report.append(f"- Saved filler_word_metrics.csv for R regression model\n")

    # Also compute filler stats specifically for tracked-only items
    filler_tracked = filler[filler["has_metrics"] == True] if "has_metrics" in filler.columns else filler
    filler_surp = filler_tracked["surprisal"].dropna()
    filler_ent = filler_tracked["entropy"].dropna()
    if len(filler_surp) > 0:
        report.append(f"- **Tracked filler words** (n={len(filler_surp)}): "
                      f"surprisal mean={filler_surp.mean():.2f}, entropy mean={filler_ent.mean():.2f}")
    n_untracked = len(filler) - len(filler_tracked)
    report.append(f"- Filler words with None metrics (never reached by loop): {n_untracked}")
    report.append("")


def analysis_plan_c_spr(df, report, output_dir):
    """Plan C: Merge SEDD metrics with SPR reading times and compute correlations."""
    report.append("## 11. Plan C: SEDD Metrics vs SPR Reading Times\n")

    spr_name_map = {
        "filler": "Fillers.csv",
        "Agreement": "AgreementSet.csv",
        "ClassicGP": "ClassicGardenPathSet.csv",
        "RelativeClause": "RelativeClauseSet.csv",
        "AttachmentAmbiguity": "AttachmentSet.csv",
    }

    all_merged = []
    for subset, fname in spr_name_map.items():
        spr_path = SPR_DIR / fname
        if not spr_path.exists():
            report.append(f"- SPR data not found for {subset}: {spr_path}")
            continue

        spr = pd.read_csv(spr_path)
        sedd_sub = df[df["subset"] == subset].copy()
        if sedd_sub.empty:
            continue

        sedd_sub["WordPosition"] = sedd_sub["word_pos"] + 1

        sedd_unique = sedd_sub.drop_duplicates(subset=["sentence", "WordPosition"])
        merge_cols = ["sentence", "WordPosition", "steps_to_commit", "weighted_steps",
                      "surprisal", "entropy", "cumulative_kl", "word", "subset",
                      "item", "condition", "has_metrics"]
        merge_cols = [c for c in merge_cols if c in sedd_unique.columns]

        merged = pd.merge(
            spr,
            sedd_unique[merge_cols],
            left_on=["Sentence", "WordPosition"],
            right_on=["sentence", "WordPosition"],
            how="inner",
        )
        if not merged.empty:
            all_merged.append(merged)

    if not all_merged:
        report.append("*No SPR data could be merged.*\n")
        return

    spr_merged = pd.concat(all_merged, ignore_index=True)
    spr_merged.to_csv(output_dir / "sedd_spr_merged.csv", index=False)
    report.append(f"Total SPR-SEDD merged rows: {len(spr_merged)}\n")

    sedd_metrics = ["steps_to_commit", "weighted_steps"]
    tracked_only = spr_merged[spr_merged["has_metrics"] == True] if "has_metrics" in spr_merged.columns else spr_merged
    sedd_metrics_full = ["steps_to_commit", "weighted_steps", "surprisal", "entropy", "cumulative_kl"]

    # Identify the RT column (merge may produce duplicates)
    rt_col = "RT"
    if rt_col not in spr_merged.columns:
        rt_candidates = [c for c in spr_merged.columns if c.startswith("RT") and "Answering" not in c and "across" not in c]
        rt_col = rt_candidates[0] if rt_candidates else None
    if rt_col is None:
        report.append("*No RT column found in SPR-SEDD merged data.*\n")
        return spr_merged

    # Ensure 1-D numeric
    if isinstance(spr_merged[rt_col], pd.DataFrame):
        spr_merged[rt_col] = spr_merged[rt_col].iloc[:, 0]
    spr_merged[rt_col] = pd.to_numeric(spr_merged[rt_col], errors="coerce")

    report.append("### Correlation of SEDD metrics with SPR RT\n")
    report.append("**All positions (steps_to_commit, weighted_steps available for all):**\n")
    report.append("| Subset | Metric | Pearson r | Spearman rho | n |")
    report.append("|--------|--------|-----------|-------------|---|")

    for subset in ["filler", "Agreement", "ClassicGP", "RelativeClause", "AttachmentAmbiguity"]:
        sub = spr_merged[spr_merged["subset"] == subset]
        if sub.empty:
            continue
        for metric in sedd_metrics:
            if metric not in sub.columns:
                continue
            valid = sub.dropna(subset=[metric, rt_col])
            if len(valid) < 5:
                continue
            r, p_r = stats.pearsonr(valid[metric].values, valid[rt_col].values)
            rho, p_rho = stats.spearmanr(valid[metric].values, valid[rt_col].values)
            sig = "*" if p_r < 0.05 else ""
            report.append(f"| {subset} | {metric} | {r:.3f}{sig} | {rho:.3f} | {len(valid)} |")

    report.append("")
    report.append("**Tracked-only positions (where surprisal/entropy are NOT None):**\n")
    report.append("| Subset | Metric | Pearson r | Spearman rho | n |")
    report.append("|--------|--------|-----------|-------------|---|")

    for subset in ["filler", "Agreement", "ClassicGP", "RelativeClause", "AttachmentAmbiguity"]:
        sub = tracked_only[tracked_only["subset"] == subset]
        if sub.empty:
            continue
        for metric in sedd_metrics_full:
            if metric not in sub.columns:
                continue
            valid = sub.dropna(subset=[metric, rt_col])
            if len(valid) < 5:
                continue
            r, p_r = stats.pearsonr(valid[metric].values, valid[rt_col].values)
            rho, p_rho = stats.spearmanr(valid[metric].values, valid[rt_col].values)
            sig = "*" if p_r < 0.05 else ""
            report.append(f"| {subset} | {metric} | {r:.3f}{sig} | {rho:.3f} | {len(valid)} |")
    report.append("")

    return spr_merged


def analysis_plan_c_et(df, report, output_dir):
    """Plan C: Merge SEDD with eye-tracking data and compute correlations."""
    report.append("## 12. Plan C: SEDD Metrics vs Eye-Tracking Measures\n")

    if not ET_DATA_PATH.exists():
        report.append("*Eye-tracking data (all_wide.csv) not found.*\n")
        return

    et_raw = pd.read_csv(ET_DATA_PATH)
    report.append(f"Eye-tracking raw data: {len(et_raw)} rows, {et_raw['item'].nunique()} items\n")

    # Pivot each ET measure from wide (ffdR1, ffdR2, ...) to long (subj, item, cond, region, ffd)
    # Then join all measures on (subj, item, cond, region)
    measure_longs = {}
    for measure in ET_MEASURES:
        measure_cols = [c for c in et_raw.columns if c.startswith(measure + "R")]
        if not measure_cols:
            continue
        rows = []
        for col in measure_cols:
            region = int(col.replace(measure + "R", ""))
            sub = et_raw[["subj", "item", "cond", col]].copy()
            sub = sub.rename(columns={col: measure})
            sub["region"] = region
            rows.append(sub)
        measure_longs[measure] = pd.concat(rows, ignore_index=True)

    if not measure_longs:
        report.append("*Could not pivot ET data.*\n")
        return

    measures_iter = iter(measure_longs.values())
    et_long = next(measures_iter)
    for part in measures_iter:
        et_long = pd.merge(et_long, part, on=["subj", "item", "cond", "region"], how="outer")

    sedd_exp = df[df["subset"] != "filler"].copy()
    sedd_exp["region"] = sedd_exp["word_pos"] + 1

    sedd_unique = sedd_exp.drop_duplicates(subset=["item", "condition", "region"])
    merge_cols = ["item", "condition", "region", "subset", "sentence",
                  "steps_to_commit", "weighted_steps", "surprisal", "entropy",
                  "cumulative_kl", "word", "n_tokens", "has_metrics"]
    merge_cols = [c for c in merge_cols if c in sedd_unique.columns]

    et_merged = pd.merge(
        et_long,
        sedd_unique[merge_cols],
        left_on=["item", "cond", "region"],
        right_on=["item", "condition", "region"],
        how="inner",
    )

    if et_merged.empty:
        report.append("*No ET data could be merged with SEDD (check item/condition alignment).*\n")
        return

    et_merged.to_csv(output_dir / "sedd_et_merged.csv", index=False)
    report.append(f"ET-SEDD merged rows: {len(et_merged)}\n")

    report.append("### Correlation of SEDD metrics with ET measures (all positions)\n")
    report.append("| Measure | steps_to_commit r | weighted_steps r | n |")
    report.append("|---------|-------------------|-----------------|---|")

    et_avail = [m for m in ET_MEASURES if m in et_merged.columns]
    for measure in et_avail:
        row_parts = [f"| {measure}"]
        for metric in ["steps_to_commit", "weighted_steps"]:
            valid = et_merged.dropna(subset=[metric, measure])
            if len(valid) >= 5:
                r, p = stats.pearsonr(valid[metric], valid[measure])
                sig = "*" if p < 0.05 else ""
                row_parts.append(f" {r:.3f}{sig}")
            else:
                row_parts.append(" —")
        n_valid = len(et_merged.dropna(subset=["steps_to_commit", et_avail[0]])) if et_avail else 0
        row_parts.append(f" {n_valid}")
        report.append(" |".join(row_parts) + " |")
    report.append("")

    tracked = et_merged[et_merged["has_metrics"] == True] if "has_metrics" in et_merged.columns else et_merged
    if len(tracked) > 10:
        report.append("### Tracked-only positions (surprisal/entropy available)\n")
        report.append(f"n = {len(tracked)} merged rows\n")
        report.append("| Measure | surprisal r | entropy r | cumulative_kl r |")
        report.append("|---------|------------|-----------|----------------|")
        for measure in et_avail:
            row_parts = [f"| {measure}"]
            for metric in ["surprisal", "entropy", "cumulative_kl"]:
                valid = tracked.dropna(subset=[metric, measure])
                if len(valid) >= 5:
                    r, p = stats.pearsonr(valid[metric], valid[measure])
                    sig = "*" if p < 0.05 else ""
                    row_parts.append(f" {r:.3f}{sig}")
                else:
                    row_parts.append(" —")
            report.append(" |".join(row_parts) + " |")
        report.append("")

    return et_merged


def analysis_plan_c_spillover(df, spr_merged, report, output_dir):
    """Plan C: Spillover analysis — does SEDD at position i predict RT at i+k?"""
    report.append("## 13. Plan C: Spillover Analysis\n")

    if spr_merged is None or spr_merged.empty:
        report.append("*No SPR-SEDD merged data for spillover.*\n")
        return

    predictors = ["steps_to_commit", "weighted_steps"]
    predictors = [p for p in predictors if p in spr_merged.columns]

    # Identify the RT column — the SPR merge might create duplicates
    rt_col = "RT"
    if rt_col not in spr_merged.columns:
        rt_candidates = [c for c in spr_merged.columns if c.startswith("RT") and "Answering" not in c and "across" not in c]
        if rt_candidates:
            rt_col = rt_candidates[0]
        else:
            report.append("*No RT column found in SPR-SEDD merged data.*\n")
            return

    # Ensure RT is 1-D numeric
    spr_work = spr_merged.copy()
    if isinstance(spr_work[rt_col], pd.DataFrame):
        spr_work[rt_col] = spr_work[rt_col].iloc[:, 0]
    spr_work[rt_col] = pd.to_numeric(spr_work[rt_col], errors="coerce")

    all_corrs = []
    for pred in predictors:
        for lag in range(0, 4):
            if lag == 0:
                valid = spr_work.dropna(subset=[pred, rt_col])
            else:
                spr_shifted = spr_work.copy()
                spr_shifted["RT_lagged"] = spr_shifted.groupby(["Sentence"])[rt_col].shift(-lag)
                valid = spr_shifted.dropna(subset=[pred, "RT_lagged"])

            if len(valid) < 10:
                continue
            outcome = valid["RT_lagged"] if lag > 0 else valid[rt_col]
            r, p = stats.pearsonr(valid[pred].values, outcome.values)
            all_corrs.append({"predictor": pred, "lag": lag, "r": r, "p": p, "n": len(valid)})

    if not all_corrs:
        report.append("*Insufficient data for spillover analysis.*\n")
        return

    corr_df = pd.DataFrame(all_corrs)
    corr_df.to_csv(output_dir / "spr_spillover_correlations.csv", index=False)

    report.append("### SPR spillover: SEDD metric at position i vs RT at i+k\n")
    report.append("| Predictor | Lag | r | p | n |")
    report.append("|-----------|-----|---|---|---|")
    for _, row in corr_df.iterrows():
        sig = "*" if row["p"] < 0.05 else ""
        report.append(f"| {row['predictor']} | i+{int(row['lag'])} | {row['r']:.3f}{sig} | {row['p']:.4f} | {int(row['n'])} |")
    report.append("")

    # Plot spillover profiles
    fig_dir = FIG_DIR / "plan_c"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for pred in predictors:
        sub = corr_df[corr_df["predictor"] == pred]
        ax.plot(sub["lag"], sub["r"], marker="o", label=pred, linewidth=2)
        for _, row in sub.iterrows():
            if row["p"] < 0.05:
                ax.annotate("*", (row["lag"], row["r"]), textcoords="offset points",
                            xytext=(0, 5), ha="center", fontsize=14, fontweight="bold")
    ax.set_xlabel("Spillover lag (words)")
    ax.set_ylabel("Pearson r with RT")
    ax.set_title("SPR Spillover: SEDD steps-to-commit vs Reading Time")
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"i+{k}" for k in range(4)])
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "spr_spillover_profile.png")
    plt.close()
    report.append("*Spillover figure saved to figures/plan_c/spr_spillover_profile.png*\n")


def analysis_plan_c_condition_effects(df, spr_merged, report, output_dir):
    """Plan C: Compare condition effects in SEDD metrics vs reading times."""
    report.append("## 14. Plan C: Condition Effects — SEDD vs Reading Time\n")

    if spr_merged is None or spr_merged.empty:
        report.append("*No SPR-SEDD merged data.*\n")
        return

    contrasts = {
        "Agreement": ("UNAGREE", "AGREE"),
        "RelativeClause": ("RC_Obj", "RC_Subj"),
    }

    type_col = "Type" if "Type" in spr_merged.columns else "condition"

    # Identify the RT column
    rt_col = "RT"
    if rt_col not in spr_merged.columns:
        rt_candidates = [c for c in spr_merged.columns if c.startswith("RT") and "Answering" not in c and "across" not in c]
        rt_col = rt_candidates[0] if rt_candidates else None
    if rt_col is None:
        report.append("*No RT column found.*\n")
        return
    if isinstance(spr_merged[rt_col], pd.DataFrame):
        spr_merged[rt_col] = spr_merged[rt_col].iloc[:, 0]
    spr_merged[rt_col] = pd.to_numeric(spr_merged[rt_col], errors="coerce")

    results = []
    for subset, (cond_a, cond_b) in contrasts.items():
        sub = spr_merged[spr_merged["subset"] == subset]
        if sub.empty:
            continue

        g_a = sub[sub[type_col] == cond_a]
        g_b = sub[sub[type_col] == cond_b]
        if g_a.empty or g_b.empty:
            continue

        for metric in [rt_col, "steps_to_commit", "weighted_steps"]:
            if metric not in g_a.columns:
                continue
            vals_a = g_a[metric].dropna()
            vals_b = g_b[metric].dropna()
            if len(vals_a) < 3 or len(vals_b) < 3:
                continue
            diff = vals_a.mean() - vals_b.mean()
            pooled_std = np.sqrt(((len(vals_a) - 1) * vals_a.std()**2 + (len(vals_b) - 1) * vals_b.std()**2)
                                 / (len(vals_a) + len(vals_b) - 2))
            d = diff / pooled_std if pooled_std > 0 else 0
            t_stat, p_val = stats.ttest_ind(vals_a, vals_b)
            results.append({
                "subset": subset, "contrast": f"{cond_a} - {cond_b}",
                "metric": metric, "mean_a": vals_a.mean(), "mean_b": vals_b.mean(),
                "diff": diff, "cohens_d": d, "t": t_stat, "p": p_val,
            })

    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(output_dir / "plan_c_condition_effects.csv", index=False)

        report.append("### Do SEDD metrics show the same direction as RT for condition contrasts?\n")
        report.append("| Subset | Contrast | Metric | Mean_A | Mean_B | diff | d | p |")
        report.append("|--------|----------|--------|--------|--------|------|---|---|")
        for _, r in res_df.iterrows():
            sig = "*" if r["p"] < 0.05 else ""
            report.append(f"| {r['subset']} | {r['contrast']} | {r['metric']} | "
                          f"{r['mean_a']:.1f} | {r['mean_b']:.1f} | {r['diff']:+.1f} | "
                          f"{r['cohens_d']:.2f} | {r['p']:.4f}{sig} |")
        report.append("")

        report.append("### Interpretation\n")
        for subset, (cond_a, cond_b) in contrasts.items():
            sub_res = res_df[res_df["subset"] == subset]
            rt_row = sub_res[sub_res["metric"] == rt_col]
            steps_row = sub_res[sub_res["metric"] == "steps_to_commit"]
            if not rt_row.empty and not steps_row.empty:
                rt_dir = "+" if rt_row.iloc[0]["diff"] > 0 else "-"
                steps_dir = "+" if steps_row.iloc[0]["diff"] > 0 else "-"
                same = rt_dir == steps_dir
                report.append(f"- **{subset}** ({cond_a} vs {cond_b}): "
                              f"RT diff = {rt_row.iloc[0]['diff']:+.1f}ms ({rt_dir}), "
                              f"steps diff = {steps_row.iloc[0]['diff']:+.1f} ({steps_dir}) "
                              f"→ {'Same' if same else 'OPPOSITE'} direction")
        report.append("")


# ============================================================================
# Main
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = []
    report.append("# Strict-LTR SEDD Analysis: Full Results\n")
    report.append(f"*Generated from all SAP subsets including fillers*\n")
    report.append("---\n")

    print("Loading all strict-LTR data...")
    df = load_all_data()
    print(f"  Loaded {len(df)} word-level entries across {df['subset'].nunique()} subsets")

    print("Loading token-level data for factor analysis...")
    token_df = load_token_level_data()
    print(f"  Loaded {len(token_df)} token-level entries")

    print("\n1. Summary statistics...")
    analysis_summary_stats(df, report)

    print("2. Condition effects...")
    cond_effects = analysis_condition_effects(df, report, OUTPUT_DIR)

    print("3. Effect direction profile...")
    analysis_effect_direction_profile(df, report, OUTPUT_DIR)

    print("4. Factor decomposition...")
    analysis_factor_decomposition(token_df, report, OUTPUT_DIR)

    print("5. Eye-tracking correlations...")
    analysis_et_correlations(df, report, OUTPUT_DIR)

    print("6. Steps vs surprisal (Plan A)...")
    analysis_filler_word_level(df, report, OUTPUT_DIR)

    print("7. Trajectory clustering (Plan B)...")
    analysis_trajectory_clustering(report, OUTPUT_DIR)

    print("8. Entropy profiles...")
    analysis_entropy_profiles(df, report, OUTPUT_DIR)

    print("9. GPT-2 comparison...")
    analysis_gpt2_comparison(df, report, OUTPUT_DIR)

    print("10. Filler regression prep...")
    analysis_filler_regression_prep(df, report, OUTPUT_DIR)

    print("11. Plan C: SPR correlations...")
    spr_merged = analysis_plan_c_spr(df, report, OUTPUT_DIR)

    print("12. Plan C: ET correlations...")
    analysis_plan_c_et(df, report, OUTPUT_DIR)

    print("13. Plan C: Spillover analysis...")
    analysis_plan_c_spillover(df, spr_merged, report, OUTPUT_DIR)

    print("14. Plan C: Condition effects vs RT...")
    analysis_plan_c_condition_effects(df, spr_merged, report, OUTPUT_DIR)

    # ====================================================================
    # Key takeaways
    # ====================================================================
    report.append("---\n")
    report.append("## Key Takeaways\n")

    # Summarize condition effects
    if cond_effects:
        sig_effects = [e for e in cond_effects if e["p_value"] < 0.05]
        total = len(cond_effects)
        report.append(f"### Condition differentiation")
        report.append(f"- {len(sig_effects)}/{total} metric-subset combinations show significant "
                      f"condition effects at the critical position (p < 0.05)")
        for e in sig_effects:
            report.append(f"  - **{e['subset']}** / {e['metric']}: {e['cond_a']} vs {e['cond_b']}, "
                          f"diff={e['diff']:+.2f}, d={e['cohens_d']:.2f}, p={e['p_value']:.4f}")
        report.append("")

    # Steps vs surprisal relationship (tracked-only)
    has_m_mask = df["has_metrics"] == True if "has_metrics" in df.columns else pd.Series(True, index=df.index)
    combined = df[has_m_mask].dropna(subset=["steps_to_commit", "surprisal"])
    combined = combined[combined["surprisal"] > 0]
    if len(combined) > 10:
        r, _ = stats.pearsonr(combined["steps_to_commit"], combined["surprisal"])
        r_w, _ = stats.pearsonr(combined["weighted_steps"], combined["surprisal"])
        report.append("### Steps-to-commit vs surprisal")
        report.append(f"- Overall Pearson correlation: r = {r:.3f}")
        report.append(f"- Weighted steps vs surprisal: r = {r_w:.3f}")
        if abs(r) > 0.3:
            report.append(f"- Steps and surprisal are moderately correlated, suggesting shared signal")
        report.append(f"- The difference (residual) between steps and surprisal may capture additional processing difficulty")
        report.append("")

    # Positions with actual metrics
    exp_df = df[df["subset"] != "filler"]
    if "has_metrics" in exp_df.columns:
        n_tracked = exp_df["has_metrics"].sum()
        n_total = len(exp_df)
        pct = n_tracked / n_total * 100 if n_total > 0 else 0
        report.append("### Denoising coverage")
        report.append(f"- Only **{n_tracked}/{n_total} ({pct:.0f}%)** experimental word positions were "
                      f"actually reached by the denoising loop")
        report.append(f"- The remaining {n_total - n_tracked} positions have **no metrics at all** "
                      f"(None) — the 1024-step budget was exhausted on earlier positions")
        report.append("")

    # Prediction accuracy
    if "correct" in df.columns:
        acc = exp_df["correct"].mean() * 100
        report.append("### Token prediction accuracy")
        report.append(f"- Experimental items: {acc:.1f}% of committed tokens match the target")
        report.append(f"- This means `steps_to_commit` reflects denoising effort, not just target token difficulty")
        report.append("")

    report.append("## Remaining Issues\n")
    report.append("1. **Most positions have NO denoising metrics**: The strict-LTR loop exhausts its 1024-step ")
    report.append("   budget on the first ~3 tokens. Later positions are committed in the post-loop final denoiser ")
    report.append("   with `final_surprisal=None, final_entropy=None`. Only `steps_to_commit` (=1) is recorded.\n")
    report.append("2. **Position confound in strict-LTR**: `steps_taken` is strongly correlated with position ")
    report.append("   (later positions commit faster). `weighted_steps` partially addresses this, but the ")
    report.append("   critical-position experiment (running each position from t=0) provides a cleaner comparison.\n")
    report.append("3. **Low token prediction accuracy**: Most committed tokens don't match targets. The trajectory ")
    report.append("   and denoising effort measures still reflect processing difficulty, but the relationship to ")
    report.append("   specific target tokens is indirect.\n")
    report.append("4. **Metric-sampler mismatch**: Logged surprisal/entropy are computed from the **normalized raw score**, ")
    report.append("   but the actual token commitment uses a different distribution (`staggered_score * transp_transition` ")
    report.append("   then `sample_categorical`). These can diverge, so surprisal does not directly describe the decision ")
    report.append("   that produced the committed token.\n")
    report.append("5. **ClassicGP conditions**: Only UAMB conditions are present in the stimuli CSV. The AMB ")
    report.append("   conditions would be needed for a full ambiguity contrast.\n")
    report.append("6. **Eye-tracking correlations**: Item-level correlations may be weak due to small n per condition. ")
    report.append("   Participant-level analysis with mixed-effects models (via the R script) would be more powerful.\n")

    report.append("## Next Steps\n")
    report.append("1. **Critical-position experiment** (highest priority): Run each target position from full noise ")
    report.append("   (step 0) with correct prefix. This gives every position the full step budget and avoids the ")
    report.append("   coverage/positional confound that makes most strict-LTR metrics meaningless.\n")
    report.append("2. **Fix the metric-sampler mismatch**: Either log surprisal/entropy from the same `probs` ")
    report.append("   distribution used by `sample_categorical`, or add `sampler_surprisal` alongside the current ")
    report.append("   `raw_score_surprisal`. This makes metrics directly interpretable.\n")
    report.append("3. **Run soft-context and Monte Carlo experiments** to test whether continuous context ")
    report.append("   representations improve predictive power.\n")
    report.append("4. **Cross-experiment comparison**: After both strict-LTR and critical-position results are available, ")
    report.append("   run `LTR_SAP_comparison/compare_experiments.py`.\n")

    # Write report
    report_text = "\n".join(report)
    report_path = OUTPUT_DIR / "STRICT_LTR_ANALYSIS_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"\n{'='*70}")
    print(f"Analysis complete!")
    print(f"  Report: {report_path}")
    print(f"  CSVs:   {OUTPUT_DIR}")
    print(f"  Figures: {FIG_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
