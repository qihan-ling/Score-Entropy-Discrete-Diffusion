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
    SEDDModelWrapper, StimulusLoader, compute_surprisal, create_output_dir,
    GPT2ModelWrapper, compute_gpt2_baseline, NATS_TO_BITS
)
from sampling import get_predictor, Denoiser
from model import utils as mutils


def run_cross_position_tracking(model, stimuli, num_steps=256, save_every=4,
                                max_spillover=3):
    """
    For each stimulus, run diffusion and track P(target_token) at every
    prior position across the denoising trajectory. Also tracks spillover
    at positions n+1, n+2, ..., n+max_spillover after the target.

    Returns:
        detail_df: One row per (stimulus, context position, timestep).
        summary_df: One row per (stimulus, context position), aggregated.
        spillover_detail_df: One row per (stimulus, spillover offset, timestep).
        spillover_summary_df: One row per (stimulus, spillover offset), aggregated.
    """
    detail_rows = []
    summary_rows = []
    spillover_detail_rows = []
    spillover_summary_rows = []

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

        valid_sp_offsets = [off for off in range(1, max_spillover + 1)
                           if target_pos + off < seq_len]
        sp_tracking = {off: [] for off in valid_sp_offsets}

        x = model.graph.sample_limit(batch_size, seq_len).to(model.device)

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

                    for offset in valid_sp_offsets:
                        sp_pos = target_pos + offset
                        sp_prob = model.get_target_prob(
                            probs, target_token_id, sp_pos)
                        if sp_prob is None:
                            continue
                        sp_tracking[offset].append({
                            'step': i,
                            'timestep': t_val,
                            'target_prob': sp_prob,
                            'target_surprisal': compute_surprisal(sp_prob),
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

            for offset in valid_sp_offsets:
                sp_pos = target_pos + offset
                sp_prob = model.get_target_prob(
                    probs, target_token_id, sp_pos)
                if sp_prob is None:
                    continue
                sp_tracking[offset].append({
                    'step': num_steps,
                    'timestep': eps,
                    'target_prob': sp_prob,
                    'target_surprisal': compute_surprisal(sp_prob),
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

        for offset in valid_sp_offsets:
            sp_pos = target_pos + offset
            word_at_sp = model.tokenizer.decode([tokens[0, sp_pos].item()])

            for point in sp_tracking[offset]:
                spillover_detail_rows.append({
                    **base_info,
                    'spillover_offset': offset,
                    'spillover_position': sp_pos,
                    'spillover_word': word_at_sp,
                    **point,
                })

            if sp_tracking[offset]:
                prob_series = [pt['target_prob'] for pt in sp_tracking[offset]]
                surp_series = [pt['target_surprisal']
                               for pt in sp_tracking[offset]]

                spillover_summary_rows.append({
                    **base_info,
                    'spillover_offset': offset,
                    'spillover_position': sp_pos,
                    'spillover_word': word_at_sp,
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

    return (pd.DataFrame(detail_rows), pd.DataFrame(summary_rows),
            pd.DataFrame(spillover_detail_rows),
            pd.DataFrame(spillover_summary_rows))


def _build_relative_heatmap(subset_df, max_distance=5):
    """
    Build an averaged heatmap of P(target) using relative positions.

    Rows: relative position (n-1, n-2, ..., n-5) where n = target position.
    Columns: timestep values (noise → clean).

    For sentences where target_pos < max_distance, missing positions are
    filled with 0. Returns the heatmap matrix, the count of non-zero
    contributions per cell, and the sorted timestep values.
    """
    capped = subset_df[subset_df['distance_to_target'] <= max_distance].copy()
    if len(capped) == 0:
        return None, None, None

    timesteps_sorted = sorted(capped['timestep'].unique(), reverse=True)

    # Identify all sentence keys (item × condition × ambiguous)
    sent_keys = capped.groupby(
        ['item', 'condition', 'ambiguous']).ngroups
    all_sentences = capped.groupby(
        ['item', 'condition', 'ambiguous']).first().index

    distances = list(range(1, max_distance + 1))

    heatmap = np.zeros((max_distance, len(timesteps_sorted)))
    counts = np.zeros((max_distance, len(timesteps_sorted)), dtype=int)

    ts_to_col = {t: i for i, t in enumerate(timesteps_sorted)}

    for _, row in capped.iterrows():
        dist = int(row['distance_to_target'])
        ts = row['timestep']
        if dist < 1 or dist > max_distance:
            continue
        r = dist - 1
        c = ts_to_col.get(ts)
        if c is None:
            continue
        heatmap[r, c] += row['target_prob']
        counts[r, c] += 1

    # Average (avoid divide by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_heatmap = np.where(counts > 0, heatmap / counts, 0.0)

    return avg_heatmap, counts, timesteps_sorted


def _build_gpt2_column(gpt2_df, subset_filter=None, max_distance=5):
    """
    Build a GPT-2 P(target) column vector (one value per distance 1..max_distance).

    Returns an array of shape (max_distance, 1) or None.
    """
    if gpt2_df is None or len(gpt2_df) == 0:
        return None

    sub = gpt2_df.copy()
    if subset_filter is not None:
        sub = subset_filter(sub)
    sub = sub[sub['distance_to_target'].between(1, max_distance)]
    if len(sub) == 0:
        return None

    sub = sub.copy()
    sub['target_prob'] = 2.0 ** (-sub['target_surprisal_bits'])

    col = np.zeros((max_distance, 1))
    for d in range(1, max_distance + 1):
        d_sub = sub[sub['distance_to_target'] == d]
        if len(d_sub) > 0:
            col[d - 1, 0] = d_sub['target_prob'].mean()
    return col


def _render_heatmap_grid(slices_with_data, y_labels, output_path,
                         max_distance=5, ncols=None):
    """
    Render a grid of heatmaps from pre-built data.

    slices_with_data: list of (label, heatmap, counts, timesteps, gpt2_col)
        gpt2_col: optional (max_distance, 1) array for a GPT-2 column
    """
    n = len(slices_with_data)
    if n == 0:
        return
    if ncols is None:
        ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 5 * nrows),
                             squeeze=False)

    vmax_global = 0
    for entry in slices_with_data:
        hm = entry[1]
        if hm is not None:
            vmax_global = max(vmax_global, hm.max())
        gpt2_col = entry[4] if len(entry) > 4 else None
        if gpt2_col is not None:
            vmax_global = max(vmax_global, gpt2_col.max())
    if vmax_global == 0:
        vmax_global = 1e-6

    for idx, entry in enumerate(slices_with_data):
        label, hm, ct, ts = entry[0], entry[1], entry[2], entry[3]
        gpt2_col = entry[4] if len(entry) > 4 else None
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        if hm is None:
            ax.set_title(f'{label}\n(no data)')
            ax.axis('off')
            continue

        if gpt2_col is not None:
            gap = np.full((max_distance, 1), np.nan)
            combined = np.hstack([hm, gap, gpt2_col])
        else:
            combined = hm

        masked = np.ma.array(combined, mask=np.isnan(combined))
        cmap = plt.cm.YlOrRd.copy()
        cmap.set_bad(color='white')

        im = ax.imshow(masked, aspect='auto', cmap=cmap,
                        interpolation='nearest',
                        vmin=0, vmax=vmax_global)

        ax.set_yticks(range(max_distance))
        ax.set_yticklabels(y_labels, fontsize=10)

        n_cols_hm = hm.shape[1]
        n_ts = len(ts)
        tick_step = max(1, n_ts // 8)
        x_tick_idx = list(range(0, n_ts, tick_step))
        x_labels_list = [f'{ts[i]:.2f}' for i in x_tick_idx]

        if gpt2_col is not None:
            gpt2_x = n_cols_hm + 1
            x_tick_idx.append(gpt2_x)
            x_labels_list.append('GPT-2')

        ax.set_xticks(x_tick_idx)
        ax.set_xticklabels(x_labels_list, fontsize=7, rotation=45)
        ax.set_xlabel('Timestep (noise → clean)', fontsize=10)
        ax.set_ylabel('Relative context position', fontsize=10)

        total_sents = ct.max() if ct.max() > 0 else 0
        min_sents = ct[ct > 0].min() if (ct > 0).any() else 0
        ax.set_title(f'{label}\n(N={total_sents}, min contrib={min_sents})',
                      fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8, label='Mean P(target)')

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot: {output_path}")
    plt.close()


def _plot_averaged_heatmaps(detail_df, output_dir, max_distance=5,
                            gpt2_df=None):
    """
    Create averaged heatmaps with two figures:
    1. exp2_heatmaps.png: All | Ambiguous | Unambiguous
    2. exp2_heatmaps_by_condition.png: one panel per base condition
    """
    has_ambig = ('ambiguous' in detail_df.columns and
                 detail_df['ambiguous'].notna().any())
    has_base = 'base_condition' in detail_df.columns

    y_labels = [f'n−{d}' for d in range(1, max_distance + 1)]

    # --- Figure 1: All / Ambiguous / Unambiguous ---
    slice_specs = [('All sentences', None)]
    if has_ambig:
        slice_specs.append(('Ambiguous', lambda df: df[df['ambiguous'] == 1]))
        slice_specs.append(
            ('Unambiguous', lambda df: df[df['ambiguous'] == 0]))

    heatmaps = []
    for label, filt in slice_specs:
        sub = filt(detail_df) if filt else detail_df
        hm, ct, ts = _build_relative_heatmap(sub, max_distance)
        gpt2_col = _build_gpt2_column(gpt2_df, filt, max_distance)
        heatmaps.append((label, hm, ct, ts, gpt2_col))

    _render_heatmap_grid(heatmaps, y_labels,
                         f'{output_dir}/exp2_heatmaps.png',
                         max_distance=max_distance)

    for entry in heatmaps:
        label, hm, ct, ts = entry[0], entry[1], entry[2], entry[3]
        if ct is None:
            continue
        print(f"\n  Sentence contributions per cell ({label}):")
        for d in range(max_distance):
            unique_counts = np.unique(ct[d])
            print(f"    n−{d+1}: {unique_counts}")

    # --- Figure 2: By condition ---
    if has_base:
        base_conds = sorted(detail_df['base_condition'].unique())
        cond_heatmaps = []
        for bc in base_conds:
            sub = detail_df[detail_df['base_condition'] == bc]
            hm, ct, ts = _build_relative_heatmap(sub, max_distance)
            bc_filter = (lambda df, _bc=bc:
                         df[df['base_condition'] == _bc])
            gpt2_col = _build_gpt2_column(gpt2_df, bc_filter, max_distance)
            cond_heatmaps.append((bc, hm, ct, ts, gpt2_col))

        _render_heatmap_grid(cond_heatmaps, y_labels,
                             f'{output_dir}/exp2_heatmaps_by_condition.png',
                             max_distance=max_distance)


def plot_cross_position(detail_df, summary_df, stimuli, model, output_dir,
                        gpt2_df=None):
    """Create visualizations of cross-position tracking."""

    if len(detail_df) == 0:
        print("  No data to plot.")
        return

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
                yerr=final_data['prob_std'], fmt='o-', capsize=3,
                label='SEDD')
    if gpt2_df is not None:
        gpt2_prior = gpt2_df[gpt2_df['distance_to_target'] > 0]
        gpt2_prob = gpt2_prior.copy()
        gpt2_prob['target_prob'] = 2.0 ** (-gpt2_prob['target_surprisal_bits'])
        gpt2_agg = gpt2_prob.groupby('distance_to_target')['target_prob'].agg(
            ['mean', 'std']).reset_index()
        gpt2_agg = gpt2_agg.sort_values('distance_to_target')
        ax.errorbar(gpt2_agg['distance_to_target'], gpt2_agg['mean'],
                    yerr=gpt2_agg['std'], fmt='s--', capsize=3,
                    label='GPT-2', color='#9C27B0')
    ax.set_xlabel('Distance to target (tokens)', fontsize=12)
    ax.set_ylabel('P(target token) at final timestep', fontsize=12)
    ax.set_title('How prior positions predict the target', fontsize=13)
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: surprisal decrease by distance (in bits)
    ax = axes[1]
    decrease_data = summary_df.groupby('distance_to_target').agg({
        'surprisal_decrease': ['mean', 'std'],
    }).reset_index()
    decrease_data.columns = ['distance', 'decrease_mean', 'decrease_std']
    decrease_data = decrease_data.sort_values('distance')

    ax.bar(decrease_data['distance'],
           decrease_data['decrease_mean'] * NATS_TO_BITS,
           yerr=decrease_data['decrease_std'] * NATS_TO_BITS,
           capsize=3, alpha=0.7)
    ax.set_xlabel('Distance to target (tokens)', fontsize=12)
    ax.set_ylabel('Surprisal decrease during denoising (bits)', fontsize=12)
    ax.set_title('Priming effect by distance', fontsize=13)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = f'{output_dir}/exp2_cross_position_overview.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot: {plot_path}")
    plt.close()

    # 2. Averaged heatmaps with relative positions (n-1 through n-5)
    _plot_averaged_heatmaps(detail_df, output_dir, gpt2_df=gpt2_df)

    # 3. Condition comparison
    if 'condition' in summary_df.columns and summary_df['condition'].nunique() > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        for cond, cond_df in summary_df.groupby('condition'):
            grouped = cond_df.groupby(
                'distance_to_target')['target_prob_final'].mean()
            grouped = grouped.sort_index(ascending=False)
            ax.plot(grouped.index, grouped.values, 'o-', label=cond,
                    markersize=5)

        if gpt2_df is not None:
            gpt2_prior = gpt2_df[gpt2_df['distance_to_target'] > 0]
            gpt2_prob = gpt2_prior.copy()
            gpt2_prob['target_prob'] = 2.0 ** (
                -gpt2_prob['target_surprisal_bits'])
            for cond, cond_sub in gpt2_prob.groupby('condition'):
                grouped = cond_sub.groupby(
                    'distance_to_target')['target_prob'].mean()
                grouped = grouped.sort_index(ascending=False)
                ax.plot(grouped.index, grouped.values, 's--',
                        label=f'{cond} (GPT-2)', markersize=4, alpha=0.7)

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
                        label=f'{label} (SEDD)',
                        color=colors.get(int(amb_val)),
                        capsize=3, markersize=5)
        if gpt2_df is not None:
            gpt2_prior = gpt2_df[gpt2_df['distance_to_target'] > 0]
            gpt2_prob = gpt2_prior.copy()
            gpt2_prob['target_prob'] = 2.0 ** (
                -gpt2_prob['target_surprisal_bits'])
            gpt2_has_amb = ('ambiguous' in gpt2_prob.columns and
                            gpt2_prob['ambiguous'].notna().any())
            if gpt2_has_amb:
                for amb_val in sorted(
                        gpt2_prob['ambiguous'].dropna().unique()):
                    sub = gpt2_prob[gpt2_prob['ambiguous'] == amb_val]
                    grouped = sub.groupby(
                        'distance_to_target')['target_prob'].agg(
                        ['mean', 'sem'])
                    grouped = grouped.sort_index(ascending=False)
                    label = amb_labels.get(int(amb_val), str(amb_val))
                    ax.errorbar(grouped.index, grouped['mean'],
                                yerr=grouped['sem'], fmt='s--',
                                label=f'{label} (GPT-2)',
                                color=colors.get(int(amb_val)),
                                capsize=3, markersize=4, alpha=0.7)
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
            ax.errorbar(grouped.index,
                        grouped['mean'] * NATS_TO_BITS,
                        yerr=grouped['sem'] * NATS_TO_BITS,
                        fmt='o-',
                        label=label, color=colors.get(int(amb_val)),
                        capsize=3, markersize=5)
        ax.set_xlabel('Distance to target (tokens)', fontsize=12)
        ax.set_ylabel('Surprisal decrease during denoising (bits)',
                       fontsize=12)
        ax.set_title('Priming strength: Amb vs Unamb', fontsize=13)
        ax.invert_xaxis()
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        amb_path = f'{output_dir}/exp2_ambiguity_effect.png'
        plt.savefig(amb_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot: {amb_path}")
        plt.close()


def plot_spillover_exp2(sp_detail_df, sp_summary_df, output_dir,
                        gpt2_df=None):
    """Create separate spillover visualizations for exp2."""
    if len(sp_summary_df) == 0:
        print("  No spillover data to plot.")
        return

    has_ambig = ('ambiguous' in sp_summary_df.columns and
                 sp_summary_df['ambiguous'].notna().any())
    amb_labels = {0: 'Unambiguous', 1: 'Ambiguous'}
    colors = {0: '#2196F3', 1: '#F44336'}

    gpt2_sp = None
    if gpt2_df is not None:
        gpt2_sp = gpt2_df[gpt2_df['distance_to_target'] < 0].copy()
        gpt2_sp['spillover_offset'] = -gpt2_sp['distance_to_target']
        gpt2_sp['target_prob'] = 2.0 ** (-gpt2_sp['target_surprisal_bits'])

    # --- Figure 1: Spillover overview ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for offset in sorted(sp_detail_df['spillover_offset'].unique()):
        sub = sp_detail_df[sp_detail_df['spillover_offset'] == offset]
        grouped = sub.groupby('timestep')['target_prob'].agg(['mean', 'sem'])
        grouped = grouped.sort_index(ascending=False)
        ax.plot(grouped.index, grouped['mean'], 'o-',
                label=f'n+{offset} (SEDD)', markersize=3)
        ax.fill_between(grouped.index,
                        grouped['mean'] - grouped['sem'],
                        grouped['mean'] + grouped['sem'], alpha=0.15)
    if gpt2_sp is not None and len(gpt2_sp) > 0:
        for offset in sorted(gpt2_sp['spillover_offset'].unique()):
            sub = gpt2_sp[gpt2_sp['spillover_offset'] == offset]
            mean_val = sub['target_prob'].mean()
            ax.axhline(y=mean_val, linestyle='--', alpha=0.6,
                        label=f'n+{int(offset)} (GPT-2)')
    ax.set_xlabel('Timestep (noise → clean)', fontsize=12)
    ax.set_ylabel('P(target token)', fontsize=12)
    ax.set_title('Spillover: P(target) at post-target positions', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    final_by_offset = sp_summary_df.groupby('spillover_offset').agg({
        'target_prob_final': ['mean', 'sem'],
    }).reset_index()
    final_by_offset.columns = ['offset', 'prob_mean', 'prob_sem']
    x_pos = np.arange(len(final_by_offset))
    x_labels = [f'n+{int(o)}' for o in final_by_offset['offset']]
    bar_width = 0.35
    ax.bar(x_pos - bar_width / 2, final_by_offset['prob_mean'],
           bar_width, yerr=final_by_offset['prob_sem'], capsize=5,
           alpha=0.7, color='#4CAF50', label='SEDD')
    if gpt2_sp is not None and len(gpt2_sp) > 0:
        gpt2_final = gpt2_sp.groupby('spillover_offset')['target_prob'].agg(
            ['mean', 'sem']).reset_index()
        gpt2_offsets = set(gpt2_final['spillover_offset'])
        gpt2_means = []
        gpt2_sems = []
        for o in final_by_offset['offset']:
            row = gpt2_final[gpt2_final['spillover_offset'] == o]
            if len(row) > 0:
                gpt2_means.append(row['mean'].values[0])
                gpt2_sems.append(row['sem'].values[0])
            else:
                gpt2_means.append(0)
                gpt2_sems.append(0)
        ax.bar(x_pos + bar_width / 2, gpt2_means, bar_width,
               yerr=gpt2_sems, capsize=5, alpha=0.7, color='#9C27B0',
               label='GPT-2')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Post-target position', fontsize=12)
    ax.set_ylabel('P(target) at final timestep', fontsize=12)
    ax.set_title('Spillover strength by distance', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = f'{output_dir}/exp2_spillover.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot: {path}")
    plt.close()

    # --- Figure 2: Spillover ambiguity comparison ---
    if has_ambig:
        offsets = sorted(sp_summary_df['spillover_offset'].unique())
        n_off = len(offsets)
        fig, axes = plt.subplots(1, n_off, figsize=(5 * n_off, 5),
                                 squeeze=False)
        axes = axes[0]

        for ax, offset in zip(axes, offsets):
            sub_detail = sp_detail_df[
                sp_detail_df['spillover_offset'] == offset]
            for amb_val in sorted(
                    sub_detail['ambiguous'].dropna().unique()):
                amb_sub = sub_detail[sub_detail['ambiguous'] == amb_val]
                grouped = amb_sub.groupby('timestep')['target_prob'].agg(
                    ['mean', 'sem'])
                grouped = grouped.sort_index(ascending=False)
                label = amb_labels.get(int(amb_val), str(amb_val))
                ax.plot(grouped.index, grouped['mean'], 'o-',
                        label=f'{label} (SEDD)',
                        color=colors.get(int(amb_val)),
                        markersize=3)
                ax.fill_between(grouped.index,
                                grouped['mean'] - grouped['sem'],
                                grouped['mean'] + grouped['sem'],
                                alpha=0.15, color=colors.get(int(amb_val)))
            if gpt2_sp is not None and len(gpt2_sp) > 0:
                gpt2_off = gpt2_sp[gpt2_sp['spillover_offset'] == offset]
                gpt2_has_amb = ('ambiguous' in gpt2_off.columns and
                                gpt2_off['ambiguous'].notna().any())
                if gpt2_has_amb:
                    for amb_val in sorted(
                            gpt2_off['ambiguous'].dropna().unique()):
                        amb_sub = gpt2_off[gpt2_off['ambiguous'] == amb_val]
                        mean_val = amb_sub['target_prob'].mean()
                        label = amb_labels.get(int(amb_val), str(amb_val))
                        ax.axhline(y=mean_val, linestyle='--', alpha=0.6,
                                    color=colors.get(int(amb_val)),
                                    label=f'{label} (GPT-2)')
            ax.set_xlabel('Timestep (noise → clean)', fontsize=10)
            ax.set_ylabel('P(target token)', fontsize=10)
            ax.set_title(f'Position n+{offset}', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle('Spillover: Ambiguous vs Unambiguous', fontsize=14, y=1.02)
        plt.tight_layout()
        path = f'{output_dir}/exp2_spillover_ambiguity.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot: {path}")
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
    parser.add_argument('--gpt2-device', type=str, default='cpu',
                        help='Device for GPT-2 baseline')
    parser.add_argument('--no-gpt2', action='store_true',
                        help='Skip GPT-2 baseline computation')
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
    print(f"    Tracking {ex['target_token_pos']} prior positions + spillover")

    print("\nRunning cross-position tracking...")
    detail_df, summary_df, sp_detail_df, sp_summary_df = \
        run_cross_position_tracking(
            model, stimuli,
            num_steps=args.num_steps,
            save_every=args.save_every)

    detail_path = f'{output_dir}/exp2_cross_position_detail_{loader.name}.csv'
    summary_path = f'{output_dir}/exp2_cross_position_summary_{loader.name}.csv'
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Saved detail: {detail_path} ({len(detail_df)} rows)")
    print(f"  Saved summary: {summary_path} ({len(summary_df)} rows)")

    sp_detail_path = f'{output_dir}/exp2_spillover_detail_{loader.name}.csv'
    sp_summary_path = f'{output_dir}/exp2_spillover_summary_{loader.name}.csv'
    sp_detail_df.to_csv(sp_detail_path, index=False)
    sp_summary_df.to_csv(sp_summary_path, index=False)
    print(f"  Saved spillover detail: {sp_detail_path} ({len(sp_detail_df)} rows)")
    print(f"  Saved spillover summary: {sp_summary_path} ({len(sp_summary_df)} rows)")

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

    if len(sp_summary_df) > 0:
        print("\n  Spillover summary:")
        sp_dist = sp_summary_df.groupby('spillover_offset').agg({
            'target_prob_final': ['mean', 'std'],
            'prob_increase': 'mean',
        }).round(6)
        print(sp_dist.to_string())

    gpt2_df = None
    if not args.no_gpt2:
        print("\nLoading GPT-2...")
        gpt2 = GPT2ModelWrapper(device=args.gpt2_device)
        gpt2_df = compute_gpt2_baseline(stimuli, gpt2)

    print("\nCreating plots...")
    plot_cross_position(detail_df, summary_df, stimuli, model, output_dir,
                        gpt2_df=gpt2_df)
    plot_spillover_exp2(sp_detail_df, sp_summary_df, output_dir,
                        gpt2_df=gpt2_df)

    print("\nDone.")
    return 0


if __name__ == '__main__':
    exit(main())
