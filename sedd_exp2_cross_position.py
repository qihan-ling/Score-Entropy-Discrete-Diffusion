"""
Experiment 2: Cross-Position Probability Tracking

Goal: Evaluate how prior words prime the prediction of the target word.
Track P(target_token) in the logits at ALL prior token positions during
the denoising trajectory.

Method:
- For each stimulus, run full diffusion (t=1.0 → t≈0) with left-to-right
  projection up to the target position.
- At each timestep, for every position p < target_pos, record:
    P(target_token_id) from logits at position p
- This creates a 2D map: position × timestep → P(target)

Key insight: If position p "primes" the target, we expect P(target) at p
to increase over the diffusion trajectory, especially once p's own token
converges to the ground truth.
"""

import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sedd_experiment_utils import (
    SEDDModelWrapper, StimulusLoader, compute_surprisal, create_output_dir
)
from sampling import get_predictor, Denoiser
from model import utils as mutils


def run_cross_position_tracking(model, stimuli, num_steps=256, save_every=4):
    """
    For each stimulus, run diffusion and track P(target_token) at every
    prior position across the denoising trajectory.

    Returns:
        detail_df: One row per (stimulus, position, timestep) combination.
        summary_df: One row per (stimulus, position), aggregated over timesteps.
    """
    detail_rows = []
    summary_rows = []

    predictor = get_predictor('analytic')(model.graph, model.noise)
    denoiser = Denoiser(model.graph, model.noise)
    score_fn = mutils.get_score_fn(model.model, train=False, sampling=True)

    eps = 1e-5
    timesteps = torch.linspace(1, eps, num_steps + 1, device=model.device)
    dt = (1 - eps) / num_steps

    for stim in tqdm(stimuli, desc="Stimuli"):
        tokens = model.tokenize(stim['sentence'])
        batch_size, seq_len = tokens.shape
        target_token_id = stim['target_token_id']
        target_pos = stim['target_token_pos']

        if target_pos < 1 or target_pos >= seq_len:
            continue

        tracking = {p: [] for p in range(target_pos)}

        x = model.graph.sample_limit(batch_size, 1024).to(model.device)

        def project(x):
            """Fix all positions before target_pos to ground truth."""
            x[:, :target_pos] = tokens[:, :target_pos]
            return x

        with torch.no_grad():
            for i in range(num_steps):
                t = timesteps[i] * torch.ones(
                    x.shape[0], 1, device=model.device)
                curr_sigma = model.noise(t)[0]

                x = project(x)

                if i % save_every == 0:
                    logits = model.forward_no_diagonal_masking(tokens, curr_sigma)
                    probs = model.logits_to_probs(logits)

                    t_val = t[0, 0].item()

                    for p in range(target_pos):
                        target_prob = model.get_target_prob(
                            probs, target_token_id, p)
                        if target_prob is None:
                            continue

                        current_token = x[0, p].item()
                        ground_truth = tokens[0, p].item()

                        tracking[p].append({
                            'step': i,
                            'timestep': t_val,
                            'target_prob': target_prob,
                            'target_surprisal': compute_surprisal(target_prob),
                            'position_token_correct': current_token == ground_truth,
                        })

                x = predictor.update_fn(score_fn, x, t, dt)

            # Final step
            x = project(x)
            t = timesteps[-1] * torch.ones(
                x.shape[0], 1, device=model.device)
            final_sigma = model.noise(t)[0]

            logits = model.forward_no_diagonal_masking(tokens, final_sigma)
            probs = model.logits_to_probs(logits)

            for p in range(target_pos):
                target_prob = model.get_target_prob(
                    probs, target_token_id, p)
                if target_prob is None:
                    continue
                tracking[p].append({
                    'step': num_steps,
                    'timestep': eps,
                    'target_prob': target_prob,
                    'target_surprisal': compute_surprisal(target_prob),
                    'position_token_correct': True,
                })

        base_info = {
            'item': stim['item'],
            'condition': stim['condition'],
            'base_condition': stim.get('base_condition', stim['condition']),
            'sentence': stim['sentence'],
            'target_word': stim['target_word'],
            'target_token_pos': target_pos,
            'disamb_position': stim['disamb_position'],
            'ambiguous': stim.get('ambiguous', None),
        }

        for p in range(target_pos):
            word_at_p = model.tokenizer.decode([tokens[0, p].item()])
            distance = target_pos - p

            for point in tracking[p]:
                detail_rows.append({
                    **base_info,
                    'context_position': p,
                    'context_word': word_at_p,
                    'distance_to_target': distance,
                    **point,
                })

            if tracking[p]:
                prob_series = [pt['target_prob'] for pt in tracking[p]]
                surp_series = [pt['target_surprisal'] for pt in tracking[p]]

                summary_rows.append({
                    **base_info,
                    'context_position': p,
                    'context_word': word_at_p,
                    'distance_to_target': distance,
                    'target_prob_initial': prob_series[0],
                    'target_prob_final': prob_series[-1],
                    'target_prob_max': max(prob_series),
                    'target_prob_mean': np.mean(prob_series),
                    'target_surprisal_initial': surp_series[0],
                    'target_surprisal_final': surp_series[-1],
                    'target_surprisal_min': min(surp_series),
                    'prob_increase': prob_series[-1] - prob_series[0],
                    'surprisal_decrease': surp_series[0] - surp_series[-1],
                })

    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def plot_cross_position(detail_df, summary_df, stimuli, model, output_dir):
    """Create visualizations of cross-position tracking."""

    # 1. Heatmap: averaged P(target) across positions and timesteps
    if len(detail_df) == 0:
        print("  No data to plot.")
        return

    # Aggregate across all items
    pivot = detail_df.groupby(
        ['distance_to_target', 'timestep']
    )['target_prob'].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: average P(target) at final timestep by distance
    ax = axes[0]
    final_data = summary_df.groupby('distance_to_target').agg({
        'target_prob_final': ['mean', 'std'],
        'target_surprisal_final': ['mean', 'std'],
    }).reset_index()
    final_data.columns = [
        'distance', 'prob_mean', 'prob_std', 'surp_mean', 'surp_std']
    final_data = final_data.sort_values('distance')

    ax.errorbar(final_data['distance'], final_data['prob_mean'],
                yerr=final_data['prob_std'], fmt='o-', capsize=3)
    ax.set_xlabel('Distance to target (tokens)', fontsize=12)
    ax.set_ylabel('P(target token) at final timestep', fontsize=12)
    ax.set_title('How prior positions predict the target', fontsize=13)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    # Right: surprisal decrease by distance
    ax = axes[1]
    decrease_data = summary_df.groupby('distance_to_target').agg({
        'surprisal_decrease': ['mean', 'std'],
    }).reset_index()
    decrease_data.columns = ['distance', 'decrease_mean', 'decrease_std']
    decrease_data = decrease_data.sort_values('distance')

    ax.bar(decrease_data['distance'], decrease_data['decrease_mean'],
           yerr=decrease_data['decrease_std'], capsize=3, alpha=0.7)
    ax.set_xlabel('Distance to target (tokens)', fontsize=12)
    ax.set_ylabel('Surprisal decrease during denoising (nats)', fontsize=12)
    ax.set_title('Priming effect by distance', fontsize=13)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = f'{output_dir}/exp2_cross_position_overview.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot: {plot_path}")
    plt.close()

    # 2. Heatmap for a few example items
    unique_items = detail_df['item'].unique()[:4]
    if len(unique_items) > 0:
        fig, axes = plt.subplots(
            1, len(unique_items),
            figsize=(5 * len(unique_items), 5))
        if len(unique_items) == 1:
            axes = [axes]

        for ax, item in zip(axes, unique_items):
            item_data = detail_df[detail_df['item'] == item]
            cond = item_data['condition'].iloc[0]

            item_pivot = item_data.pivot_table(
                index='context_position',
                columns='timestep',
                values='target_prob',
                aggfunc='mean')

            if item_pivot.empty:
                continue

            item_pivot = item_pivot.sort_index(ascending=True)
            item_pivot = item_pivot[sorted(
                item_pivot.columns, reverse=True)]

            im = ax.imshow(
                item_pivot.values,
                aspect='auto',
                cmap='YlOrRd',
                interpolation='nearest')

            y_labels = []
            for pos in item_pivot.index:
                word_rows = item_data[item_data['context_position'] == pos]
                if len(word_rows) > 0:
                    w = word_rows.iloc[0]['context_word'].strip()[:8]
                    y_labels.append(f'{pos}:{w}')
                else:
                    y_labels.append(str(pos))

            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels, fontsize=8)
            ax.set_xlabel('Timestep (noise → clean)', fontsize=10)
            ax.set_ylabel('Context position', fontsize=10)
            ax.set_title(f'Item {item} ({cond})\n'
                         f'Target: "{item_data.iloc[0]["target_word"]}"',
                         fontsize=10)
            plt.colorbar(im, ax=ax, shrink=0.8)

        plt.tight_layout()
        heatmap_path = f'{output_dir}/exp2_heatmaps.png'
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot: {heatmap_path}")
        plt.close()

    # 3. Condition comparison
    if 'condition' in summary_df.columns and summary_df['condition'].nunique() > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        for cond, cond_df in summary_df.groupby('condition'):
            grouped = cond_df.groupby(
                'distance_to_target')['target_prob_final'].mean()
            grouped = grouped.sort_index(ascending=False)
            ax.plot(grouped.index, grouped.values, 'o-', label=cond,
                    markersize=5)

        ax.set_xlabel('Distance to target (tokens)', fontsize=12)
        ax.set_ylabel('P(target) at final timestep', fontsize=12)
        ax.set_title('Cross-position priming by condition', fontsize=13)
        ax.invert_xaxis()
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        cond_path = f'{output_dir}/exp2_by_condition.png'
        plt.savefig(cond_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot: {cond_path}")
        plt.close()

    # 4. Ambiguous vs unambiguous priming comparison
    has_ambig = ('ambiguous' in summary_df.columns and
                 summary_df['ambiguous'].notna().any())
    if has_ambig:
        amb_labels = {0: 'Unambiguous', 1: 'Ambiguous'}
        colors = {0: '#2196F3', 1: '#F44336'}

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        for amb_val in sorted(summary_df['ambiguous'].dropna().unique()):
            sub = summary_df[summary_df['ambiguous'] == amb_val]
            grouped = sub.groupby(
                'distance_to_target')['target_prob_final'].agg(
                ['mean', 'sem'])
            grouped = grouped.sort_index(ascending=False)
            label = amb_labels.get(int(amb_val), str(amb_val))
            ax.errorbar(grouped.index, grouped['mean'],
                        yerr=grouped['sem'], fmt='o-',
                        label=label, color=colors.get(int(amb_val)),
                        capsize=3, markersize=5)
        ax.set_xlabel('Distance to target (tokens)', fontsize=12)
        ax.set_ylabel('P(target) at final timestep', fontsize=12)
        ax.set_title('Priming: Ambiguous vs Unambiguous', fontsize=13)
        ax.invert_xaxis()
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        for amb_val in sorted(summary_df['ambiguous'].dropna().unique()):
            sub = summary_df[summary_df['ambiguous'] == amb_val]
            grouped = sub.groupby(
                'distance_to_target')['surprisal_decrease'].agg(
                ['mean', 'sem'])
            grouped = grouped.sort_index(ascending=False)
            label = amb_labels.get(int(amb_val), str(amb_val))
            ax.errorbar(grouped.index, grouped['mean'],
                        yerr=grouped['sem'], fmt='o-',
                        label=label, color=colors.get(int(amb_val)),
                        capsize=3, markersize=5)
        ax.set_xlabel('Distance to target (tokens)', fontsize=12)
        ax.set_ylabel('Surprisal decrease during denoising', fontsize=12)
        ax.set_title('Priming strength: Amb vs Unamb', fontsize=13)
        ax.invert_xaxis()
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        amb_path = f'{output_dir}/exp2_ambiguity_effect.png'
        plt.savefig(amb_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot: {amb_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Exp 2: Cross-position probability tracking')
    parser.add_argument('--input', type=str,
                        default='SAP_stimuli copy/sap_items_ClassicGP.csv',
                        help='Input CSV')
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-steps', type=int, default=256,
                        help='Diffusion steps (lower = faster but coarser)')
    parser.add_argument('--save-every', type=int, default=4,
                        help='Record every N steps')
    args = parser.parse_args()

    output_dir = create_output_dir(args.output_dir)

    print("=" * 60)
    print(" Experiment 2: Cross-Position Probability Tracking")
    print("=" * 60)
    print(f"  Input: {args.input}")
    print(f"  Steps: {args.num_steps}, save every {args.save_every}")

    print("\nLoading model...")
    model = SEDDModelWrapper(device=args.device)
    print(f"  Model loaded on {model.device}")

    print("\nLoading stimuli...")
    loader = StimulusLoader(args.input, model.tokenizer)
    stimuli = loader.get_stimuli()
    print(f"  Loaded {len(stimuli)} stimuli from {loader.name}")

    if len(stimuli) == 0:
        print("ERROR: No stimuli loaded.")
        return 1

    ex = stimuli[0]
    print(f"\n  Example:")
    print(f"    Sentence: {ex['sentence']}")
    print(f"    Target: '{ex['target_word']}' at token pos {ex['target_token_pos']}")
    print(f"    Tracking {ex['target_token_pos']} prior positions")

    print("\nRunning cross-position tracking...")
    detail_df, summary_df = run_cross_position_tracking(
        model, stimuli,
        num_steps=args.num_steps,
        save_every=args.save_every)

    detail_path = f'{output_dir}/exp2_cross_position_detail_{loader.name}.csv'
    summary_path = f'{output_dir}/exp2_cross_position_summary_{loader.name}.csv'
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Saved detail: {detail_path} ({len(detail_df)} rows)")
    print(f"  Saved summary: {summary_path} ({len(summary_df)} rows)")

    # Summary stats
    print("\n" + "=" * 60)
    print("  Summary by distance to target:")
    print("=" * 60)
    if len(summary_df) > 0:
        dist_summary = summary_df.groupby('distance_to_target').agg({
            'target_prob_final': ['mean', 'std'],
            'target_surprisal_final': ['mean', 'std'],
            'prob_increase': 'mean',
        }).round(4)
        print(dist_summary.to_string())

    print("\nCreating plots...")
    plot_cross_position(detail_df, summary_df, stimuli, model, output_dir)

    print("\nDone.")
    return 0


if __name__ == '__main__':
    exit(main())
