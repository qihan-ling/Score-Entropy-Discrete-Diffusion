"""
Post-hoc computation of derived trajectory metrics from existing results.

Metrics computed:
  A. Future token persistence (requires --track_future_tokens data)
  B. Derived trajectory metrics from frontier_history:
     - Belief convergence step (p_target > threshold)
     - Entropy cliff step (largest single-step entropy drop)
     - Max KL step
     - Argmax stability (consecutive same-argmax steps before commitment)
     - Top-K overlap stability (Jaccard of top-5 between consecutive steps)
  C. Prefix score probing summary (requires --track_prefix_scores data)

Usage:
  python LTR_SAP_critical/analysis/compute_derived_metrics.py [--results_dir LTR_SAP_critical]
"""

import json
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent


def compute_trajectory_metrics(frontier_history):
    """Compute derived metrics from a single result's frontier_history."""
    if not frontier_history:
        return {}

    entropies = [h.get("entropy", 0) for h in frontier_history]
    kls = [h.get("kl_from_prev", 0) for h in frontier_history]
    p_targets = [h.get("target_prob", h.get("p_target", 0)) or 0 for h in frontier_history]

    # Extract argmax from top_k if available
    argmaxes = []
    for h in frontier_history:
        top_k = h.get("top_k", [])
        if top_k:
            first = top_k[0]
            if isinstance(first, dict):
                argmaxes.append(first.get("id", -1))
            elif isinstance(first, (list, tuple)):
                argmaxes.append(first[0])
            else:
                argmaxes.append(-1)
        else:
            argmaxes.append(h.get("argmax_id", -1))

    metrics = {}

    # Belief convergence: first step where p_target exceeds threshold
    for thresh in [0.01, 0.05, 0.1, 0.5]:
        for i, pt in enumerate(p_targets):
            if pt > thresh:
                metrics[f"convergence_{thresh}"] = frontier_history[i]["step"]
                break

    # Entropy cliff: step with largest single-step entropy drop
    max_drop = 0
    for i in range(1, len(entropies)):
        drop = entropies[i-1] - entropies[i]
        if drop > max_drop:
            max_drop = drop
            metrics["entropy_cliff_step"] = frontier_history[i]["step"]
    metrics["max_entropy_drop"] = max_drop

    # Max KL step
    max_kl = 0
    for i, kl in enumerate(kls):
        if kl > max_kl:
            max_kl = kl
            metrics["max_kl_step"] = frontier_history[i]["step"]
    metrics["max_kl"] = max_kl

    # Argmax stability: consecutive same-argmax from the end
    if argmaxes:
        last = argmaxes[-1]
        count = 0
        for a in reversed(argmaxes):
            if a == last:
                count += 1
            else:
                break
        metrics["argmax_stability"] = count

    # Top-K overlap: mean Jaccard of top-5 between consecutive steps
    jaccards = []
    for i in range(1, len(frontier_history)):
        tk_prev = frontier_history[i-1].get("top_k", [])
        tk_curr = frontier_history[i].get("top_k", [])
        if tk_prev and tk_curr:
            def extract_ids(tk):
                ids = set()
                for t in tk[:5]:
                    if isinstance(t, dict):
                        ids.add(t.get("id", -1))
                    elif isinstance(t, (list, tuple)):
                        ids.add(t[0])
                return ids
            ids_prev = extract_ids(tk_prev)
            ids_curr = extract_ids(tk_curr)
            union = ids_prev | ids_curr
            if union:
                jaccards.append(len(ids_prev & ids_curr) / len(union))
    if jaccards:
        metrics["mean_top5_jaccard"] = float(np.mean(jaccards))
        metrics["min_top5_jaccard"] = float(np.min(jaccards))

    # P(target) trajectory summary
    if p_targets:
        metrics["mean_p_target"] = float(np.mean(p_targets))
        metrics["max_p_target"] = float(np.max(p_targets))
        metrics["final_p_target"] = p_targets[-1]

    return metrics


def compute_future_token_persistence(future_tokens_log):
    """Analyze future token tracking data for persistence patterns.

    Returns per-position stats on how often the same token reappears across steps.
    """
    if not future_tokens_log:
        return {}

    # Build position -> list of (step, token_id) appearances
    pos_history = {}
    for entry in future_tokens_log:
        step = entry["step"]
        for tok in entry["tokens"]:
            pos = tok["position"]
            tid = tok["token_id"]
            pos_history.setdefault(pos, []).append((step, tid))

    results = {}
    for pos, appearances in pos_history.items():
        total_appearances = len(appearances)
        token_ids = [a[1] for a in appearances]
        unique_tokens = len(set(token_ids))

        # Max consecutive same-token streak
        max_streak = 1
        streak = 1
        for i in range(1, len(token_ids)):
            if token_ids[i] == token_ids[i-1]:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 1

        # Most common token and its frequency
        from collections import Counter
        counts = Counter(token_ids)
        most_common_id, most_common_count = counts.most_common(1)[0]

        results[pos] = {
            "total_appearances": total_appearances,
            "unique_tokens": unique_tokens,
            "max_streak": max_streak,
            "most_common_token_id": most_common_id,
            "most_common_freq": most_common_count / total_appearances,
            "steps_range": (appearances[0][0], appearances[-1][0]),
        }

    return results


def compute_prefix_score_summary(prefix_scores_log):
    """Summarize prefix score probing data.

    Returns per-position stats on how p(gt) evolves over the denoising trajectory.
    """
    if not prefix_scores_log:
        return {}

    pos_p_gts = {}
    for entry in prefix_scores_log:
        for pos_str, info in entry["positions"].items():
            pos = int(pos_str)
            pos_p_gts.setdefault(pos, []).append((entry["step"], info["p_gt"]))

    results = {}
    for pos, history in pos_p_gts.items():
        p_gts = [h[1] for h in history]
        results[pos] = {
            "mean_p_gt": float(np.mean(p_gts)),
            "min_p_gt": float(np.min(p_gts)),
            "max_p_gt": float(np.max(p_gts)),
            "final_p_gt": p_gts[-1],
            "p_gt_drop": p_gts[0] - p_gts[-1] if len(p_gts) > 1 else 0,
        }

    return results


def process_all_results(results_dir):
    """Process all JSON results and compute derived metrics."""
    results_dir = Path(results_dir)
    all_rows = []

    for root, dirs, files in os.walk(results_dir):
        for fname in sorted(files):
            if not fname.endswith(".json"):
                continue
            fpath = Path(root) / fname

            try:
                with open(fpath) as f:
                    d = json.load(f)
            except Exception:
                continue

            hist = d.get("frontier_history", [])
            cl = d.get("commitment_log", {})

            # Determine subset/item/condition from path
            rel = fpath.relative_to(results_dir)
            parts = rel.parts

            traj_metrics = compute_trajectory_metrics(hist)
            ft_persistence = compute_future_token_persistence(d.get("future_tokens_log", []))
            prefix_summary = compute_prefix_score_summary(d.get("prefix_scores_log", []))

            row = {
                "file": str(rel),
                "steps_taken": cl.get("steps_taken"),
                "correct": cl.get("correct"),
                **traj_metrics,
            }

            if ft_persistence:
                row["n_future_positions_tracked"] = len(ft_persistence)
                avg_appearances = np.mean([v["total_appearances"] for v in ft_persistence.values()])
                avg_unique = np.mean([v["unique_tokens"] for v in ft_persistence.values()])
                avg_streak = np.mean([v["max_streak"] for v in ft_persistence.values()])
                row["avg_future_appearances"] = avg_appearances
                row["avg_future_unique_tokens"] = avg_unique
                row["avg_future_max_streak"] = avg_streak

            if prefix_summary:
                mean_prefix_p_gt = np.mean([v["mean_p_gt"] for v in prefix_summary.values()])
                min_prefix_p_gt = np.min([v["min_p_gt"] for v in prefix_summary.values()])
                row["mean_prefix_p_gt"] = mean_prefix_p_gt
                row["min_prefix_p_gt"] = min_prefix_p_gt

            all_rows.append(row)

    return pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compute derived trajectory metrics from critical-position results"
    )
    parser.add_argument(
        "--results_dir", type=str,
        default=str(REPO / "LTR_SAP_critical"),
        help="Root directory of critical-position results",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(REPO / "LTR_SAP_critical" / "analysis" / "results" / "derived_metrics_all.csv"),
    )
    args = parser.parse_args()

    print(f"Processing results from {args.results_dir}...")
    df = process_all_results(args.results_dir)
    print(f"Processed {len(df)} result files.")

    if df.empty:
        print("No results found.")
        return

    print(f"\nDerived metric columns: {[c for c in df.columns if c not in ('file', 'steps_taken', 'correct')]}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")

    # Summary stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print("\n--- Summary statistics ---")
    for col in numeric_cols:
        vals = df[col].dropna()
        if len(vals) > 0:
            print(f"  {col:>30s}: mean={vals.mean():>8.3f}, std={vals.std():>8.3f}, N={len(vals)}")


if __name__ == "__main__":
    main()
