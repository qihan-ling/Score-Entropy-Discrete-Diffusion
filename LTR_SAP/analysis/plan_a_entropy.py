"""
Plan A: Entropy trajectory visualizations centered on disambPosition/targetPosition.

For each SAP subset, extracts entropy trajectories for positions in the
[target-3, target+3] window. Produces:
  1. Entropy heatmaps (x=denoising step, y=relative position) per condition
  2. Line plots of mean final-step entropy per position, one line per condition

Usage:
  python LTR_SAP/analysis/plan_a_entropy.py --output_dir LTR_SAP/analysis/figures/plan_a
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from utils import (
    get_sap_files, get_subset_name, get_critical_pos_col, load_sap_csv,
    load_all_outputs, extract_critical_region_metrics, extract_all_trajectories,
    setup_matplotlib, condition_palette, LTR_SAP_DIR,
)


def get_condition_items(csv_path):
    """Group items by condition from a SAP CSV.

    Returns dict: condition -> list of (item, critical_pos, sentence)
    """
    df = load_sap_csv(csv_path)
    subset = get_subset_name(csv_path)
    crit_col = get_critical_pos_col(csv_path)
    cond_col = "condition"

    if crit_col is None or cond_col not in df.columns:
        return {}

    groups = {}
    for _, row in df.iterrows():
        cond = row[cond_col]
        item = row.get("item", None)
        crit_pos = int(row[crit_col])
        sentence = row["Sentence"]
        groups.setdefault(cond, []).append((item, crit_pos, sentence))
    return groups


def main():
    parser = argparse.ArgumentParser(description="Plan A: Entropy trajectory visualizations")
    parser.add_argument("--output_dir", type=str, default="LTR_SAP/analysis/figures/plan_a")
    parser.add_argument("--window", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plt = setup_matplotlib()
    palette = condition_palette()

    for csv_path in get_sap_files():
        subset = get_subset_name(csv_path)
        crit_col = get_critical_pos_col(csv_path)

        if crit_col is None:
            print(f"Skipping {subset} (no critical position column)")
            continue

        print(f"\nProcessing {subset}...")
        cond_items = get_condition_items(csv_path)
        if not cond_items:
            print(f"  No conditions found")
            continue

        conditions = sorted(cond_items.keys())

        # --- Figure 1: Line plot of mean entropy at commitment per relative position ---
        fig, ax = plt.subplots(figsize=(10, 6))
        has_data = False

        for cond in conditions:
            items = cond_items[cond]
            outputs = load_all_outputs(subset, cond)

            if not outputs:
                print(f"  No outputs for {subset}/{cond}")
                continue

            region_dfs = []
            for out in outputs:
                item_info = next(
                    (it for it in items if it[2] == out["tokenization"]["sentence"]),
                    None,
                )
                if item_info is None:
                    continue
                _, crit_pos, _ = item_info
                rdf = extract_critical_region_metrics(out, crit_pos, window=args.window)
                if not rdf.empty:
                    region_dfs.append(rdf)

            if not region_dfs:
                continue

            combined = pd.concat(region_dfs, ignore_index=True)
            mean_by_pos = combined.groupby("relative_pos").agg(
                entropy_mean=("entropy", "mean"),
                entropy_se=("entropy", lambda x: x.std() / np.sqrt(len(x)) if len(x) > 1 else 0),
                steps_mean=("steps_to_commit", "mean"),
            ).reset_index()

            color = palette.get(cond, None)
            ax.errorbar(
                mean_by_pos["relative_pos"], mean_by_pos["entropy_mean"],
                yerr=mean_by_pos["entropy_se"],
                label=cond, marker="o", capsize=3, color=color,
            )
            has_data = True

        if has_data:
            ax.set_xlabel("Position relative to critical word")
            ax.set_ylabel("Mean entropy at commitment (bits)")
            ax.set_title(f"{subset}: Entropy around critical position")
            ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
            ax.legend()
            ax.set_xticks(range(-args.window, args.window + 1))
            fig.savefig(os.path.join(args.output_dir, f"entropy_lineplot_{subset}.png"))
            print(f"  Saved entropy_lineplot_{subset}.png")
        plt.close(fig)

        # --- Figure 2: Heatmaps per condition ---
        for cond in conditions:
            outputs = load_all_outputs(subset, cond)
            if not outputs:
                continue

            items = cond_items[cond]
            all_trajectories = []

            for out in outputs:
                item_info = next(
                    (it for it in items if it[2] == out["tokenization"]["sentence"]),
                    None,
                )
                if item_info is None:
                    continue
                _, crit_pos, _ = item_info

                # Map token positions to relative word positions
                sentence = out["tokenization"]["sentence"]
                words = sentence.split()
                crit_0 = crit_pos - 1

                trajectories = extract_all_trajectories(out, metric="entropy")

                # For each relative position, collect the entropy trajectory
                for rel_pos in range(-args.window, args.window + 1):
                    word_idx = crit_0 + rel_pos
                    if word_idx < 0 or word_idx >= len(words):
                        continue
                    # Find token positions for this word (approximate: use commitment_log)
                    for entry in out["commitment_log"]:
                        tok_pos = entry["position"]
                        tok_word_pos = tok_pos - 1  # subtract <|endoftext|>
                        # Rough mapping: assume one token per word for simplicity
                        if tok_word_pos == word_idx:
                            traj = trajectories.get(tok_pos, [])
                            if traj:
                                all_trajectories.append({
                                    "relative_pos": rel_pos,
                                    "trajectory": traj,
                                })
                            break

            if not all_trajectories:
                continue

            # Build heatmap: average entropy at each (relative_pos, step_fraction)
            n_step_bins = 50
            heatmap = np.full((2 * args.window + 1, n_step_bins), np.nan)
            counts = np.zeros_like(heatmap)

            for entry in all_trajectories:
                rel = entry["relative_pos"] + args.window
                traj = entry["trajectory"]
                n_steps = len(traj)
                if n_steps == 0:
                    continue
                for bin_idx in range(n_step_bins):
                    step_idx = int(bin_idx / n_step_bins * n_steps)
                    step_idx = min(step_idx, n_steps - 1)
                    val = traj[step_idx]
                    if np.isnan(heatmap[rel, bin_idx]):
                        heatmap[rel, bin_idx] = val
                    else:
                        heatmap[rel, bin_idx] += val
                    counts[rel, bin_idx] += 1

            with np.errstate(divide="ignore", invalid="ignore"):
                heatmap = np.where(counts > 0, heatmap / counts, np.nan)

            fig, ax = plt.subplots(figsize=(12, 5))
            im = ax.imshow(
                heatmap, aspect="auto", origin="lower",
                extent=[0, 100, -args.window - 0.5, args.window + 0.5],
                cmap="YlOrRd",
            )
            ax.set_xlabel("Denoising progress (%)")
            ax.set_ylabel("Position relative to critical word")
            ax.set_title(f"{subset} / {cond}: Entropy trajectory heatmap")
            ax.set_yticks(range(-args.window, args.window + 1))
            ax.axhline(y=0, color="white", linestyle="--", alpha=0.7, linewidth=1)
            plt.colorbar(im, label="Entropy (bits)")
            fig.savefig(os.path.join(args.output_dir, f"entropy_heatmap_{subset}_{cond}.png"))
            plt.close(fig)
            print(f"  Saved entropy_heatmap_{subset}_{cond}.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
