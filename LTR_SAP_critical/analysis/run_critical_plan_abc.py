"""
Run adapted Plan A/B/C analyses on critical-position experiment results.

Critical-position results differ from strict-LTR in structure:
  - Each JSON file targets one specific position (not a full sentence sweep)
  - No position confound: each target got the full noise schedule
  - commitment_log is a single entry (not a list)
  - future_scores data available for look-ahead analysis

This script adapts the same analysis ideas but reads from LTR_SAP_critical/.

Pipeline:
  1. Build word-level metrics table from critical-position outputs
  2. Plan A adaptations: steps vs surprisal, entropy analysis
  3. Plan B adaptation: trajectory clustering at critical positions
  4. Plan C adaptation: alignment with human data, regression

Prerequisite: filler critical-position results must be collected.

Usage:
  python LTR_SAP_critical/analysis/run_critical_plan_abc.py
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
    get_sap_files, get_subset_name, get_critical_pos_col, load_sap_csv,
    load_critical_outputs_by_offset, load_gpt2_surprisals,
    load_eye_tracking, load_spr_data,
    ET_MEASURES, setup_matplotlib, condition_palette,
    LTR_SAP_CRITICAL_DIR,
)


def build_critical_metrics_table():
    """Build a unified table from all critical-position outputs.

    Returns DataFrame with columns:
      subset, condition, item, offset, word_position, word,
      steps_taken, t_commitment, final_entropy, final_surprisal,
      cumulative_kl, correct
    """
    all_rows = []

    for csv_path in get_sap_files():
        subset = get_subset_name(csv_path)
        df_stim = load_sap_csv(csv_path)
        crit_col = get_critical_pos_col(csv_path)
        cond_col = "condition" if "condition" in df_stim.columns else None

        if crit_col is None:
            # Filler: load per-position results
            filler_dir = LTR_SAP_CRITICAL_DIR / subset
            if not filler_dir.exists():
                continue
            for json_file in sorted(filler_dir.glob("item_*_wpos_*.json")):
                import json
                with open(json_file) as f:
                    output = json.load(f)
                entry = output.get("commitment_log", {})
                if not entry:
                    continue
                tok = output.get("tokenization", {})
                all_rows.append({
                    "subset": subset,
                    "condition": None,
                    "item": None,
                    "offset": None,
                    "word_position": entry.get("word_position"),
                    "word": entry.get("word"),
                    "steps_taken": entry.get("steps_taken"),
                    "t_commitment": entry.get("t_commitment"),
                    "final_entropy": entry.get("final_entropy"),
                    "final_surprisal": entry.get("final_surprisal"),
                    "cumulative_kl": entry.get("cumulative_kl"),
                    "correct": entry.get("correct"),
                    "sentence": tok.get("sentence"),
                })
            continue

        conditions = df_stim[cond_col].unique() if cond_col else [None]
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
                        "subset": subset,
                        "condition": condition,
                        "item": item_id,
                        "offset": offset,
                        "word_position": entry.get("word_position"),
                        "word": entry.get("word"),
                        "steps_taken": entry.get("steps_taken"),
                        "t_commitment": entry.get("t_commitment"),
                        "final_entropy": entry.get("final_entropy"),
                        "final_surprisal": entry.get("final_surprisal"),
                        "cumulative_kl": entry.get("cumulative_kl"),
                        "correct": entry.get("correct"),
                        "sentence": sentence,
                    })

    return pd.DataFrame(all_rows)


def plan_a_analysis(df, output_dir):
    """Plan A adaptations for critical-position data."""
    plt = setup_matplotlib()
    palette = condition_palette()

    print(f"\n{'='*60}")
    print(f"Plan A: Steps vs Surprisal (Critical-Position)")
    print(f"{'='*60}")

    valid = df.dropna(subset=["steps_taken", "final_surprisal"])
    if len(valid) < 10:
        print("  Insufficient data for scatter analysis")
        return

    valid["steps_z"] = stats.zscore(valid["steps_taken"])
    valid["surprisal_z"] = stats.zscore(valid["final_surprisal"])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(valid["surprisal_z"], valid["steps_z"],
               alpha=0.3, s=10, c="#2196F3", edgecolors="none")

    r, p = stats.pearsonr(valid["surprisal_z"], valid["steps_z"])
    rho, p_rho = stats.spearmanr(valid["surprisal_z"], valid["steps_z"])

    slope, intercept = np.polyfit(valid["surprisal_z"], valid["steps_z"], 1)
    x_line = np.linspace(valid["surprisal_z"].min(), valid["surprisal_z"].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=1.5)

    ax.set_xlabel("Surprisal (z-scored)")
    ax.set_ylabel("Steps-to-commit (z-scored)")
    ax.set_title(f"Steps vs Surprisal (Critical-Position)\nr={r:.3f}, rho={rho:.3f}")
    fig.savefig(output_dir / "scatter_steps_vs_surprisal_critical.png")
    plt.close(fig)
    print(f"  Saved scatter (r={r:.3f}, rho={rho:.3f})")

    # Cumulative KL vs surprisal
    kl_valid = valid.dropna(subset=["cumulative_kl"])
    if len(kl_valid) > 10:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(kl_valid["final_surprisal"], kl_valid["cumulative_kl"],
                   alpha=0.3, s=10, c="#FF9800", edgecolors="none")
        r_kl, _ = stats.pearsonr(kl_valid["final_surprisal"], kl_valid["cumulative_kl"])
        ax.set_xlabel("Surprisal (bits)")
        ax.set_ylabel("Cumulative KL (bits)")
        ax.set_title(f"Cumulative KL vs Surprisal (r={r_kl:.3f})")
        fig.savefig(output_dir / "cumkl_vs_surprisal_critical.png")
        plt.close(fig)

    valid.to_csv(output_dir / "critical_word_metrics.csv", index=False)


def plan_b_analysis(df, output_dir):
    """Plan B adaptations: trajectory clustering on critical-position data."""
    plt = setup_matplotlib()

    print(f"\n{'='*60}")
    print(f"Plan B: Trajectory Features (Critical-Position)")
    print(f"{'='*60}")

    # For critical-position, each output has a full trajectory
    # We can extract features from the frontier_history stored in the JSONs
    # For now, use the summary metrics as features
    feature_cols = ["steps_taken", "final_entropy", "cumulative_kl"]
    valid = df.dropna(subset=feature_cols)

    if len(valid) < 20:
        print("  Insufficient data for clustering")
        return

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    X = valid[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = min(4, len(valid) // 5)
    if n_clusters < 2:
        print("  Too few data points for meaningful clustering")
        return

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    valid = valid.copy()
    valid["cluster"] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    valid["pca1"] = X_pca[:, 0]
    valid["pca2"] = X_pca[:, 1]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))
    for cl in range(n_clusters):
        mask = valid["cluster"] == cl
        ax.scatter(valid.loc[mask, "pca1"], valid.loc[mask, "pca2"],
                   alpha=0.5, s=20, color=colors[cl],
                   label=f"Cluster {cl} (n={mask.sum()})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Trajectory Typology (Critical-Position)")
    ax.legend()
    fig.savefig(output_dir / "trajectory_clusters_critical.png")
    plt.close(fig)
    print(f"  Saved trajectory_clusters_critical.png ({n_clusters} clusters)")

    valid.to_csv(output_dir / "critical_trajectory_features.csv", index=False)


def plan_c_analysis(df, output_dir):
    """Plan C adaptations: align with human data."""
    print(f"\n{'='*60}")
    print(f"Plan C: Human Data Alignment (Critical-Position)")
    print(f"{'='*60}")

    # Merge with eye-tracking at item level for offset=0
    experimental = df[(df["subset"] != "filler") & (df["offset"] == 0)].copy()

    try:
        et_data = load_eye_tracking()
    except FileNotFoundError:
        print("  Eye-tracking data not found")
        et_data = None

    if et_data is not None and not experimental.empty:
        results = []
        for subset in experimental["subset"].unique():
            sub = experimental[experimental["subset"] == subset]
            conditions = sorted(sub["condition"].dropna().unique())
            if len(conditions) < 2:
                continue

            for sedd_m in ["steps_taken", "final_surprisal", "final_entropy"]:
                for et_m in ET_MEASURES:
                    merged_items = []
                    for _, sedd_row in sub.iterrows():
                        item_id = sedd_row["item"]
                        cond = sedd_row["condition"]
                        wpos = sedd_row.get("word_position")
                        if wpos is None or item_id is None:
                            continue
                        et_col = f"{et_m}R{wpos}"
                        cond_et = et_data[(et_data["cond"] == cond) & (et_data["item"] == item_id)]
                        if et_col not in cond_et.columns:
                            continue
                        et_vals = cond_et[et_col].dropna()
                        if et_vals.empty:
                            continue
                        merged_items.append({
                            "item": item_id,
                            "condition": cond,
                            "sedd_val": sedd_row[sedd_m],
                            "et_val": et_vals.mean(),
                        })

                    if len(merged_items) >= 5:
                        merged_df = pd.DataFrame(merged_items)
                        rho, p_val = stats.spearmanr(merged_df["sedd_val"], merged_df["et_val"])
                        results.append({
                            "subset": subset, "sedd_metric": sedd_m, "et_metric": et_m,
                            "spearman_rho": rho, "p_value": p_val, "n": len(merged_items),
                        })
                        if abs(rho) > 0.3:
                            print(f"    {subset} | {sedd_m} vs {et_m}: "
                                  f"rho={rho:.3f}, p={p_val:.4f}")

        if results:
            pd.DataFrame(results).to_csv(
                output_dir / "critical_et_correlations.csv", index=False
            )
            print(f"  Saved critical_et_correlations.csv ({len(results)} rows)")

    # Save filler data for conversion model
    filler = df[df["subset"] == "filler"].copy()
    if not filler.empty:
        filler.to_csv(output_dir / "critical_filler_metrics.csv", index=False)
        print(f"  Saved critical_filler_metrics.csv ({len(filler)} filler entries)")


def main():
    parser = argparse.ArgumentParser(description="Plan A/B/C on critical-position results")
    parser.add_argument("--output_dir", type=str, default="LTR_SAP_critical/analysis/results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building critical-position metrics table...")
    df = build_critical_metrics_table()

    if df.empty:
        print("No critical-position outputs found. Run batch_runner_critical_position.py first.")
        return

    print(f"  Total entries: {len(df)}")
    print(f"  Subsets: {df['subset'].unique().tolist()}")
    print(f"  Offsets: {sorted(df['offset'].dropna().unique().tolist())}")

    plan_a_analysis(df, output_dir)
    plan_b_analysis(df, output_dir)
    plan_c_analysis(df, output_dir)

    print(f"\nAll Plan A/B/C results saved to {output_dir}")


if __name__ == "__main__":
    main()
