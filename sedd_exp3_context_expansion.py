"""
Experiment 3: Context Window Expansion

Goal: Evaluate how incremental accumulation of information impacts the
denoising of the target token.

Method:
- For each stimulus, run N separate evaluations, each with a different
  amount of unmasked left context:
    Context 1: [w1, MASK, MASK, ..., MASK, target, MASK, ...]
    Context 2: [w1, w2, MASK, ..., MASK, target, MASK, ...]
    Context 3: [w1, w2, w3, MASK, ..., MASK, target, MASK, ...]
    ...
    Context K: [w1, w2, ..., w_{K}, MASK, ..., target, MASK, ...]
  where K goes up to target_pos-1.

- At each context level, run full diffusion and measure how well the
  model recovers the target token.

Key insight: This simulates incremental human reading - as more words
are processed, does the target become more predictable? At which point
in the sentence does the critical context appear?

Note: "MASK" positions are initialized as noise (absorbing state) and
are NOT denoised - only the target position is denoised. The context
words are projected (fixed) to ground truth.
"""

import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sedd_experiment_utils import (
    SEDDModelWrapper, StimulusLoader, compute_surprisal, create_output_dir,
    GPT2ModelWrapper, compute_gpt2_baseline, NATS_TO_BITS
)
from sampling import get_predictor, Denoiser
from model import utils as mutils


def run_context_expansion(model, stimuli, num_steps=256, save_every=8,
                          max_spillover=3):
    """
    For each stimulus, incrementally expand the unmasked context window
    and measure how it affects denoising of the target token. Also tracks
    spillover P(target) at positions n+1, n+2, ..., n+max_spillover.

    Returns:
        detail_df: One row per (stimulus, context_size, timestep).
        summary_df: One row per (stimulus, context_size), aggregated.
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

        valid_sp_offsets = [off for off in range(1, max_spillover + 1)
                           if target_pos + off < seq_len]

        for ctx_size in range(target_pos + 1):
            trajectory = []

            x = model.graph.sample_limit(batch_size, seq_len).to(model.device)

            def project(x, ctx_sz=ctx_size, tgt_pos=target_pos):
                if ctx_sz > 0:
                    x[:, :ctx_sz] = tokens[:, :ctx_sz]
                x[:, target_pos] = x[:, target_pos]
                return x

            with torch.no_grad():
                for i in range(num_steps):
                    t = timesteps[i] * torch.ones(
                        x.shape[0], 1, device=model.device)
                    curr_sigma = model.noise(t)[0]

                    x = project(x)

                    if i % save_every == 0:
                        current_token = x[0, target_pos].item()
                        is_correct = (current_token == target_token_id)

                        logits = model.forward_no_diagonal_masking(
                            x, curr_sigma)
                        probs = model.logits_to_probs(logits)

                        target_prob_at_prev = None
                        if target_pos > 0:
                            target_prob_at_prev = model.get_target_prob(
                                probs, target_token_id, target_pos - 1)

                        target_prob_at_self = model.get_target_prob(
                            probs, target_token_id, target_pos)

                        probs_at_pos = probs[0, target_pos]
                        log_p = torch.log(probs_at_pos + 1e-10)
                        entropy = -(probs_at_pos * log_p).sum().item()

                        sorted_idx = torch.argsort(
                            probs_at_pos, descending=True)
                        rank = (sorted_idx == target_token_id
                                ).nonzero(as_tuple=True)[0].item() + 1

                        point = {
                            'step': i,
                            'timestep': t[0, 0].item(),
                            'current_token': current_token,
                            'is_correct': is_correct,
                            'target_prob_at_self': target_prob_at_self,
                            'target_prob_at_prev': target_prob_at_prev,
                            'target_surprisal': compute_surprisal(
                                target_prob_at_self) * NATS_TO_BITS if target_prob_at_self else None,
                            'entropy': entropy,
                            'target_rank': rank,
                        }

                        for off in valid_sp_offsets:
                            sp_prob = model.get_target_prob(
                                probs, target_token_id, target_pos + off)
                            point[f'target_prob_at_n_plus_{off}'] = sp_prob

                        trajectory.append(point)

                    x = predictor.update_fn(score_fn, x, t, dt)

                # Final denoising
                x = project(x)
                t = timesteps[-1] * torch.ones(
                    x.shape[0], 1, device=model.device)
                x = denoiser.update_fn(score_fn, x, t)

                final_token = x[0, target_pos].item()
                final_correct = (final_token == target_token_id)

                final_sigma = model.noise(t)[0]
                logits = model.forward_no_diagonal_masking(x, final_sigma)
                probs = model.logits_to_probs(logits)

                final_prob = model.get_target_prob(
                    probs, target_token_id, target_pos)
                final_prob_prev = model.get_target_prob(
                    probs, target_token_id, target_pos - 1) if target_pos > 0 else None

                final_point = {
                    'step': num_steps,
                    'timestep': eps,
                    'current_token': final_token,
                    'is_correct': final_correct,
                    'target_prob_at_self': final_prob,
                    'target_prob_at_prev': final_prob_prev,
                    'target_surprisal': compute_surprisal(
                        final_prob) * NATS_TO_BITS if final_prob else None,
                    'entropy': 0.0,
                    'target_rank': 1 if final_correct else None,
                }

                for off in valid_sp_offsets:
                    sp_prob = model.get_target_prob(
                        probs, target_token_id, target_pos + off)
                    final_point[f'target_prob_at_n_plus_{off}'] = sp_prob

                trajectory.append(final_point)

            context_words = model.tokenizer.decode(
                tokens[0, :ctx_size].tolist()) if ctx_size > 0 else "<none>"

            base_info = {
                'item': stim['item'],
                'condition': stim['condition'],
                'base_condition': stim.get('base_condition', stim['condition']),
                'sentence': stim['sentence'],
                'target_word': stim['target_word'],
                'target_token_pos': target_pos,
                'disamb_position': stim['disamb_position'],
                'ambiguous': stim.get('ambiguous', None),
                'context_size': ctx_size,
                'context_words': context_words,
                'context_fraction': ctx_size / target_pos if target_pos > 0 else 0,
            }

            for pt in trajectory:
                detail_rows.append({**base_info, **pt})

            prob_series = [p['target_prob_at_self'] for p in trajectory
                           if p['target_prob_at_self'] is not None]
            surp_series = [p['target_surprisal'] for p in trajectory
                           if p['target_surprisal'] is not None]
            correct_series = [p['is_correct'] for p in trajectory]
            rank_series = [p['target_rank'] for p in trajectory
                           if p['target_rank'] is not None]

            summary_entry = {
                **base_info,
                'final_token': final_token,
                'final_correct': final_correct,
                'final_prob': prob_series[-1] if prob_series else None,
                'final_surprisal': surp_series[-1] if surp_series else None,
                'final_rank': rank_series[-1] if rank_series else None,
                'mean_prob': np.mean(prob_series) if prob_series else None,
                'mean_surprisal': np.mean(surp_series) if surp_series else None,
                'correctness_ratio': np.mean(correct_series),
                'convergence_step': next(
                    (i for i, c in enumerate(correct_series)
                     if c and all(correct_series[i:])), len(correct_series)
                ) / len(correct_series) if correct_series else 1.0,
            }

            for off in valid_sp_offsets:
                col = f'target_prob_at_n_plus_{off}'
                sp_series = [p[col] for p in trajectory
                             if p.get(col) is not None]
                summary_entry[f'final_prob_n_plus_{off}'] = (
                    sp_series[-1] if sp_series else None)
                summary_entry[f'mean_prob_n_plus_{off}'] = (
                    np.mean(sp_series) if sp_series else None)

            summary_rows.append(summary_entry)

    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def _build_context_heatmap(subset_df, prob_column='target_prob_at_self'):
    """
    Build a heatmap of P(target) vs (context_size, timestep).

    Returns the averaged heatmap, contribution counts, context_size list,
    and sorted timestep list.
    """
    valid = subset_df.dropna(subset=[prob_column])
    if len(valid) == 0:
        return None, None, None, None

    ctx_sizes = sorted(valid['context_size'].unique())
    timesteps = sorted(valid['timestep'].unique(), reverse=True)

    n_ctx = len(ctx_sizes)
    n_ts = len(timesteps)

    heatmap = np.zeros((n_ctx, n_ts))
    counts = np.zeros((n_ctx, n_ts), dtype=int)

    ctx_to_row = {c: i for i, c in enumerate(ctx_sizes)}
    ts_to_col = {t: i for i, t in enumerate(timesteps)}

    for _, row in valid.iterrows():
        r = ctx_to_row.get(row['context_size'])
        c = ts_to_col.get(row['timestep'])
        if r is None or c is None:
            continue
        heatmap[r, c] += row[prob_column]
        counts[r, c] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        avg_heatmap = np.where(counts > 0, heatmap / counts, 0.0)

    return avg_heatmap, counts, ctx_sizes, timesteps


def _render_context_heatmap_grid(slices_data, output_path, ncols=None,
                                  gpt2_values=None):
    """
    Render a grid of context-expansion heatmaps.

    slices_data: list of (label, heatmap, counts, ctx_sizes, timesteps)
    gpt2_values: optional list of GPT-2 P(target) values, one per slice.
                 If provided, each non-None value is appended as an extra
                 column separated by a white line.
    """
    n = len(slices_data)
    if n == 0:
        return
    if ncols is None:
        ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 5 * nrows),
                             squeeze=False)

    vmax_global = 0
    for _, hm, _, _, _ in slices_data:
        if hm is not None:
            vmax_global = max(vmax_global, hm.max())
    if gpt2_values:
        for v in gpt2_values:
            if v is not None:
                vmax_global = max(vmax_global, v)
    if vmax_global == 0:
        vmax_global = 1e-6

    for idx, (label, hm, ct, ctx_sizes, ts) in enumerate(slices_data):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        gpt2_val = (gpt2_values[idx]
                    if gpt2_values and idx < len(gpt2_values) else None)

        if hm is None:
            ax.set_title(f'{label}\n(no data)')
            ax.axis('off')
            continue

        if gpt2_val is not None:
            gpt2_col = np.full((hm.shape[0], 1), gpt2_val)
            hm_ext = np.hstack([hm, gpt2_col])
        else:
            hm_ext = hm

        im = ax.imshow(hm_ext, aspect='auto', cmap='YlOrRd',
                        interpolation='nearest',
                        vmin=0, vmax=vmax_global)

        if gpt2_val is not None:
            ax.axvline(x=len(ts) - 0.5, color='white', linewidth=2)

        ax.set_yticks(range(len(ctx_sizes)))
        ax.set_yticklabels([str(s) for s in ctx_sizes], fontsize=8)

        n_ts = len(ts)
        tick_step = max(1, n_ts // 8)
        x_tick_idx = list(range(0, n_ts, tick_step))
        x_labels = [f'{ts[i]:.2f}' for i in x_tick_idx]

        if gpt2_val is not None:
            x_tick_idx.append(n_ts)
            x_labels.append('GPT-2')

        ax.set_xticks(x_tick_idx)
        ax.set_xticklabels(x_labels, fontsize=7, rotation=45)
        ax.set_xlabel('Timestep (noise → clean)', fontsize=10)
        ax.set_ylabel('Context words revealed', fontsize=10)

        total_sents = ct.max() if ct.max() > 0 else 0
        min_sents = ct[ct > 0].min() if (ct > 0).any() else 0
        ax.set_title(f'{label}\n(N={total_sents}, min={min_sents})',
                      fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8, label='Mean P(target)')

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot: {output_path}")
    plt.close()


def plot_context_expansion(detail_df, summary_df, output_dir, gpt2_df=None):
    """Create heatmap visualizations of context expansion results."""
    if len(detail_df) == 0:
        print("  No data to plot.")
        return

    has_ambig = ('ambiguous' in detail_df.columns and
                 detail_df['ambiguous'].notna().any())
    has_base = 'base_condition' in detail_df.columns

    def _gpt2_target_prob(gpt2_subset):
        if gpt2_subset is None or len(gpt2_subset) == 0:
            return None
        at_target = gpt2_subset[gpt2_subset['distance_to_target'] == 1]
        if len(at_target) == 0:
            return None
        return 2 ** (-at_target['target_surprisal_bits'].mean())

    # --- Figure 1: All / Ambiguous / Unambiguous heatmaps ---
    slices = [('All sentences', detail_df)]
    if has_ambig:
        slices.append(('Ambiguous', detail_df[detail_df['ambiguous'] == 1]))
        slices.append(('Unambiguous', detail_df[detail_df['ambiguous'] == 0]))

    heatmaps = []
    for label, sub in slices:
        hm, ct, ctx, ts = _build_context_heatmap(sub)
        heatmaps.append((label, hm, ct, ctx, ts))

    gpt2_values = None
    if gpt2_df is not None:
        gpt2_values = [_gpt2_target_prob(gpt2_df)]
        if has_ambig:
            gpt2_values.append(_gpt2_target_prob(
                gpt2_df[gpt2_df['ambiguous'] == 1]))
            gpt2_values.append(_gpt2_target_prob(
                gpt2_df[gpt2_df['ambiguous'] == 0]))

    _render_context_heatmap_grid(
        heatmaps, f'{output_dir}/exp3_context_expansion.png',
        gpt2_values=gpt2_values)

    # Print contribution counts
    for label, hm, ct, ctx, ts in heatmaps:
        if ct is None:
            continue
        print(f"\n  Sentence contributions ({label}):")
        for i, cs in enumerate(ctx):
            min_ct = ct[i].min()
            max_ct = ct[i].max()
            print(f"    ctx_size={cs}: {min_ct}–{max_ct} sentences")

    # --- Figure 2: By condition heatmaps ---
    if has_base:
        base_conds = sorted(detail_df['base_condition'].unique())
        cond_heatmaps = []
        cond_gpt2_values = [] if gpt2_df is not None else None
        for bc in base_conds:
            sub = detail_df[detail_df['base_condition'] == bc]
            hm, ct, ctx, ts = _build_context_heatmap(sub)
            cond_heatmaps.append((bc, hm, ct, ctx, ts))
            if gpt2_df is not None:
                cond_gpt2_values.append(_gpt2_target_prob(
                    gpt2_df[gpt2_df['base_condition'] == bc]))

        _render_context_heatmap_grid(
            cond_heatmaps, f'{output_dir}/exp3_by_condition.png',
            gpt2_values=cond_gpt2_values)


def plot_spillover_exp3(summary_df, output_dir, max_spillover=3):
    """Create separate spillover visualizations for exp3."""
    sp_cols = [f'final_prob_n_plus_{off}' for off in range(1, max_spillover + 1)
               if f'final_prob_n_plus_{off}' in summary_df.columns]
    if not sp_cols:
        print("  No spillover data to plot.")
        return

    has_ambig = ('ambiguous' in summary_df.columns and
                 summary_df['ambiguous'].notna().any())
    amb_labels = {0: 'Unambiguous', 1: 'Ambiguous'}
    colors = {0: '#2196F3', 1: '#F44336'}

    # --- Figure 1: Spillover vs context size ---
    fig, axes = plt.subplots(1, len(sp_cols), figsize=(5 * len(sp_cols), 5),
                             squeeze=False)
    axes = axes[0]

    for ax, col in zip(axes, sp_cols):
        offset = int(col.split('_')[-1])
        sub = summary_df.dropna(subset=[col])
        if len(sub) == 0:
            ax.set_title(f'n+{offset}\n(no data)')
            continue

        grouped = sub.groupby('context_size')[col].agg(['mean', 'sem'])
        ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['sem'],
                    fmt='o-', capsize=3, markersize=4, color='#4CAF50')
        ax.set_xlabel('Context words revealed', fontsize=11)
        ax.set_ylabel(f'P(target) at n+{offset}', fontsize=11)
        ax.set_title(f'Spillover at n+{offset}', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Spillover P(target) vs context amount', fontsize=14, y=1.02)
    plt.tight_layout()
    path = f'{output_dir}/exp3_spillover.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot: {path}")
    plt.close()

    # --- Figure 2: Spillover ambiguity comparison ---
    if has_ambig:
        fig, axes = plt.subplots(1, len(sp_cols),
                                 figsize=(5 * len(sp_cols), 5),
                                 squeeze=False)
        axes = axes[0]

        for ax, col in zip(axes, sp_cols):
            offset = int(col.split('_')[-1])
            sub = summary_df.dropna(subset=[col])
            if len(sub) == 0:
                ax.set_title(f'n+{offset}\n(no data)')
                continue

            for amb_val in sorted(sub['ambiguous'].dropna().unique()):
                amb_sub = sub[sub['ambiguous'] == amb_val]
                grouped = amb_sub.groupby('context_fraction')[col].agg(
                    ['mean', 'sem'])
                label = amb_labels.get(int(amb_val), str(amb_val))
                ax.errorbar(grouped.index, grouped['mean'],
                            yerr=grouped['sem'], fmt='o-',
                            label=label, color=colors.get(int(amb_val)),
                            capsize=3, markersize=4)
            ax.set_xlabel('Context fraction', fontsize=11)
            ax.set_ylabel(f'P(target) at n+{offset}', fontsize=11)
            ax.set_title(f'Spillover at n+{offset}', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle('Spillover: Ambiguous vs Unambiguous', fontsize=14,
                      y=1.02)
        plt.tight_layout()
        path = f'{output_dir}/exp3_spillover_ambiguity.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot: {path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Exp 3: Context window expansion')
    parser.add_argument('--input', type=str,
                        default='SAP_stimuli copy/sap_items_ClassicGP.csv',
                        help='Input CSV')
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-steps', type=int, default=256,
                        help='Diffusion steps per context level')
    parser.add_argument('--save-every', type=int, default=8,
                        help='Record every N steps')
    parser.add_argument('--max-items', type=int, default=None,
                        help='Limit items (for testing)')
    parser.add_argument('--gpt2-device', type=str, default=None,
                        help='Device for GPT-2 (default: same as --device)')
    parser.add_argument('--no-gpt2', action='store_true',
                        help='Skip GPT-2 baseline')
    args = parser.parse_args()

    output_dir = create_output_dir(args.output_dir)

    print("=" * 60)
    print(" Experiment 3: Context Window Expansion")
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

    if args.max_items and len(stimuli) > args.max_items:
        stimuli = stimuli[:args.max_items]
        print(f"  Limited to {len(stimuli)} items for testing")

    if len(stimuli) == 0:
        print("ERROR: No stimuli loaded.")
        return 1

    ex = stimuli[0]
    print(f"\n  Example:")
    print(f"    Sentence: {ex['sentence']}")
    print(f"    Target: '{ex['target_word']}' at token pos {ex['target_token_pos']}")
    print(f"    Will test {ex['target_token_pos'] + 1} context levels")
    total_runs = sum(s['target_token_pos'] + 1 for s in stimuli)
    print(f"\n  Total diffusion runs: {total_runs}")
    print(f"  (each with {args.num_steps} steps)")

    print("\nRunning context expansion...")
    detail_df, summary_df = run_context_expansion(
        model, stimuli,
        num_steps=args.num_steps,
        save_every=args.save_every)

    detail_path = f'{output_dir}/exp3_context_expansion_detail_{loader.name}.csv'
    summary_path = f'{output_dir}/exp3_context_expansion_summary_{loader.name}.csv'
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Saved detail: {detail_path} ({len(detail_df)} rows)")
    print(f"  Saved summary: {summary_path} ({len(summary_df)} rows)")

    print("\n" + "=" * 60)
    print("  Summary by context size:")
    print("=" * 60)
    if len(summary_df) > 0:
        ctx_summary = summary_df.groupby('context_size').agg({
            'final_surprisal': ['mean', 'std'],
            'final_correct': 'mean',
            'final_rank': 'median',
        }).round(4)
        print(ctx_summary.to_string())

    gpt2_df = None
    if not args.no_gpt2:
        print("\nLoading GPT-2...")
        gpt2_device = args.gpt2_device or args.device
        gpt2 = GPT2ModelWrapper(device=gpt2_device)
        gpt2_df = compute_gpt2_baseline(stimuli, gpt2)
        print(f"  GPT-2 baseline: {len(gpt2_df)} rows")

    print("\nCreating plots...")
    plot_context_expansion(detail_df, summary_df, output_dir, gpt2_df=gpt2_df)
    plot_spillover_exp3(summary_df, output_dir)

    print("\nDone.")
    return 0


if __name__ == '__main__':
    exit(main())
