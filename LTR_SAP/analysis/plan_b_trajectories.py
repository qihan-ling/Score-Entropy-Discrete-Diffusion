"""
Plan B: Trajectory Shape as a Processing Typology.

Extract trajectory features from SEDD enforce-prefix outputs, run clustering,
and produce typology/taxonomy figures.

Features per word:
  - steps_taken: total denoising steps to commit
  - plateau_duration: fraction of steps where entropy is within 10% of max
  - entropy_slope: linear fit slope of entropy over steps
  - final_entropy: entropy at last step before commitment
  - cumulative_kl: total KL divergence accumulated

Produces:
  1. 2D PCA/t-SNE projection of feature vectors, colored by cluster
  2. Taxonomy figure: representative example from each cluster
  3. Trajectory overlay panel per SAP subset at the critical word
  4. Proportion stacked bar chart: cluster fractions by syntactic role

Usage:
  python LTR_SAP/analysis/plan_b_trajectories.py --output_dir LTR_SAP/analysis/figures/plan_b
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from utils import (
    get_sap_files, get_subset_name, get_critical_pos_col, load_sap_csv,
    load_all_outputs, extract_all_trajectories,
    setup_matplotlib, condition_palette, LTR_SAP_DIR,
)


def extract_trajectory_features(sedd_output):
    """Extract feature vector for each tracked position.

    Returns list of dicts with: position, word, steps_taken, plateau_duration,
    entropy_slope, final_entropy, cumulative_kl
    """
    commitment_log = sedd_output.get("commitment_log", [])
    frontier_history = sedd_output.get("frontier_history", {})
    sentence = sedd_output.get("tokenization", {}).get("sentence", "")
    words = sentence.split()

    features = []
    for entry in commitment_log:
        pos = entry["position"]
        pos_str = str(pos)
        hist = frontier_history.get(pos_str, [])

        if not hist:
            continue

        entropies = [h.get("entropy", 0) for h in hist]
        kls = [h.get("cumulative_kl", 0) for h in hist]

        steps_taken = entry["steps_taken"]
        final_entropy = entropies[-1] if entropies else 0
        cumulative_kl = kls[-1] if kls else 0

        # Plateau duration: fraction of steps with entropy >= 90% of max
        max_ent = max(entropies) if entropies else 0
        if max_ent > 0:
            plateau_count = sum(1 for e in entropies if e >= 0.9 * max_ent)
            plateau_duration = plateau_count / len(entropies)
        else:
            plateau_duration = 0.0

        # Entropy slope: linear regression of entropy over step index
        if len(entropies) >= 2:
            x = np.arange(len(entropies), dtype=float)
            y = np.array(entropies, dtype=float)
            slope = np.polyfit(x, y, 1)[0]
        else:
            slope = 0.0

        word_idx = pos - 1  # subtract <|endoftext|>
        word = words[word_idx] if 0 <= word_idx < len(words) else "?"

        features.append({
            "position": pos,
            "word_pos": word_idx,
            "word": word,
            "steps_taken": steps_taken,
            "plateau_duration": plateau_duration,
            "entropy_slope": slope,
            "final_entropy": final_entropy,
            "cumulative_kl": cumulative_kl,
            "committed_token": entry.get("committed_token", ""),
            "target_token": entry.get("target_token", ""),
            "correct": entry.get("correct", None),
            "sentence": sentence,
        })

    return features


def main():
    parser = argparse.ArgumentParser(description="Plan B: Trajectory typology analysis")
    parser.add_argument("--output_dir", type=str, default="LTR_SAP/analysis/figures/plan_b")
    parser.add_argument("--n_clusters", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plt = setup_matplotlib()
    palette = condition_palette()

    # --- Collect trajectory features across all subsets ---
    all_features = []
    for csv_path in get_sap_files():
        subset = get_subset_name(csv_path)
        df_stim = load_sap_csv(csv_path)
        cond_col = "condition" if "condition" in df_stim.columns else None

        conditions = df_stim[cond_col].unique().tolist() if cond_col else [None]
        for cond in conditions:
            outputs = load_all_outputs(subset, cond)
            for out in outputs:
                feats = extract_trajectory_features(out)
                for f in feats:
                    f["subset"] = subset
                    f["condition"] = cond
                all_features.extend(feats)

    if not all_features:
        print("No trajectory features found. Run batch_runner.py first.")
        return

    feat_df = pd.DataFrame(all_features)
    print(f"Collected {len(feat_df)} word-level trajectory features")

    # --- Clustering ---
    feature_cols = ["steps_taken", "plateau_duration", "entropy_slope",
                    "final_entropy", "cumulative_kl"]
    X = feat_df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    feat_df["cluster"] = kmeans.fit_predict(X_scaled)

    # --- PCA projection ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    feat_df["pca1"] = X_pca[:, 0]
    feat_df["pca2"] = X_pca[:, 1]

    # Figure 1: PCA scatter colored by cluster
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set2(np.linspace(0, 1, args.n_clusters))
    for cl in range(args.n_clusters):
        mask = feat_df["cluster"] == cl
        ax.scatter(
            feat_df.loc[mask, "pca1"], feat_df.loc[mask, "pca2"],
            alpha=0.4, s=15, color=colors[cl], label=f"Cluster {cl} (n={mask.sum()})",
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Denoising Trajectory Typology (PCA)")
    ax.legend()
    fig.savefig(os.path.join(args.output_dir, "trajectory_clusters_pca.png"))
    plt.close(fig)
    print("  Saved trajectory_clusters_pca.png")

    # Figure 2: Cluster profile bar chart
    cluster_means = feat_df.groupby("cluster")[feature_cols].mean()
    fig, axes = plt.subplots(1, len(feature_cols), figsize=(4 * len(feature_cols), 5))
    for i, col in enumerate(feature_cols):
        axes[i].bar(range(args.n_clusters), cluster_means[col], color=colors[:args.n_clusters])
        axes[i].set_xlabel("Cluster")
        axes[i].set_title(col)
    fig.suptitle("Cluster Feature Profiles")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "cluster_profiles.png"))
    plt.close(fig)
    print("  Saved cluster_profiles.png")

    # Figure 3: Proportion by subset
    if len(feat_df["subset"].unique()) > 1:
        prop_table = pd.crosstab(feat_df["subset"], feat_df["cluster"], normalize="index")
        fig, ax = plt.subplots(figsize=(10, 6))
        prop_table.plot.bar(stacked=True, ax=ax, color=colors[:args.n_clusters])
        ax.set_ylabel("Proportion")
        ax.set_title("Cluster distribution by SAP subset")
        ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1))
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "cluster_by_subset.png"))
        plt.close(fig)
        print("  Saved cluster_by_subset.png")

    # Figure 4: Trajectory overlay for critical words per subset
    for csv_path in get_sap_files():
        subset = get_subset_name(csv_path)
        crit_col = get_critical_pos_col(csv_path)
        if crit_col is None:
            continue

        df_stim = load_sap_csv(csv_path)
        cond_col = "condition" if "condition" in df_stim.columns else None
        conditions = sorted(df_stim[cond_col].unique().tolist()) if cond_col else [None]

        fig, ax = plt.subplots(figsize=(12, 6))
        has_data = False

        for cond in conditions:
            outputs = load_all_outputs(subset, cond)
            for out in outputs:
                sentence = out["tokenization"]["sentence"]
                match = df_stim[df_stim["Sentence"] == sentence]
                if match.empty:
                    continue
                crit_pos = int(match.iloc[0][crit_col])

                # Find the token position for the critical word
                for entry in out["commitment_log"]:
                    if entry["position"] - 1 == crit_pos - 1:
                        pos_str = str(entry["position"])
                        hist = out.get("frontier_history", {}).get(pos_str, [])
                        if hist:
                            target_probs = [h.get("target_prob", 0) for h in hist]
                            steps = range(len(target_probs))
                            color = palette.get(cond, None)
                            ax.plot(steps, target_probs, alpha=0.3, color=color, linewidth=0.8)
                            has_data = True
                        break

        if has_data:
            # Add legend entries
            for cond in conditions:
                color = palette.get(cond, None)
                ax.plot([], [], color=color, label=cond, linewidth=2)
            ax.set_xlabel("Denoising step")
            ax.set_ylabel("P(target)")
            ax.set_title(f"{subset}: P(target) trajectory at critical word")
            ax.legend()
            fig.savefig(os.path.join(args.output_dir, f"trajectory_overlay_{subset}.png"))
            print(f"  Saved trajectory_overlay_{subset}.png")
        plt.close(fig)

    # Save features CSV
    out_csv = os.path.join(args.output_dir, "trajectory_features.csv")
    feat_df.to_csv(out_csv, index=False)
    print(f"  Saved trajectory_features.csv ({len(feat_df)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
