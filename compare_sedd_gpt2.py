"""
Compare SEDD-medium vs GPT-2 Surprisal
Calculates word-level surprisal from both models and tests correlation
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from load_model import load_model
from model import utils as mutils
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')


class SurprisalExtractor:
    """Base class for surprisal extraction."""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    def extract_surprisals(self, sentences):
        """Extract surprisal for each word in each sentence."""
        raise NotImplementedError


class GPT2SurprisalExtractor(SurprisalExtractor):
    """Extract surprisal from GPT-2 (standard autoregressive LM)."""
    
    def __init__(self, model_name='gpt2', device='cuda'):
        super().__init__(device)
        print(f"\nLoading GPT-2: {model_name}")
        
        try:
            self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            print("✓ GPT-2 loaded")
        except Exception as e:
            if 'cuda' in str(self.device).lower() and ('CUDA' in str(e) or 'cuda' in str(e)):
                print(f"\n⚠️  CUDA error loading GPT-2: {e}")
                print("⚠️  Falling back to CPU...")
                self.device = torch.device('cpu')
                self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
                self.model.eval()
                print("✓ GPT-2 loaded on CPU")
            else:
                raise
    
    def extract_surprisals(self, sentences):
        """
        Extract GPT-2 surprisal using standard method.
        
        For each word, compute: -log P(word | previous_context)
        """
        all_results = []
        
        with torch.no_grad():
            for sent_idx, sentence in enumerate(tqdm(sentences, desc="GPT-2")):
                try:
                    tokens = self.tokenizer.encode(sentence, return_tensors='pt').to(self.device)
                    
                    if tokens.shape[1] > 1024:
                        tokens = tokens[:, :1024]
                    
                    # Forward pass
                    outputs = self.model(tokens)
                    logits = outputs.logits  # [1, seq_len, vocab_size]
                except Exception as e:
                    print(f"\n⚠️  Error on sentence {sent_idx}: {e}")
                    print("⚠️  Skipping this sentence...")
                    continue
                
                # Compute surprisal for each token
                for pos in range(1, tokens.shape[1]):  # Start at 1 (need context)
                    # Logits at position pos-1 predict token at position pos
                    logits_at_prev = logits[0, pos-1, :]
                    log_probs = F.log_softmax(logits_at_prev, dim=-1)
                    
                    target_token = tokens[0, pos].item()
                    surprisal = -log_probs[target_token].item()
                    
                    # Get word string
                    word = self.tokenizer.decode([target_token])
                    
                    all_results.append({
                        'sentence_idx': sent_idx,
                        'sentence': sentence,
                        'word_position': pos,
                        'word': word,
                        'token_id': target_token,
                        'surprisal_gpt2': surprisal,
                    })
        
        return pd.DataFrame(all_results)


class SEDDSurprisalExtractor(SurprisalExtractor):
    """Extract surprisal from SEDD using different methods."""
    
    def __init__(self, model_name='louaaron/sedd-medium', device='cuda', 
                 method='single_pass'):
        """
        Args:
            method: How to extract surprisal
                - 'single_pass': One forward pass at t≈0 (fastest, most comparable to GPT-2)
                - 'trajectory_mean': Average over diffusion trajectory (slow, full processing)
                - 'trajectory_weighted': Time-weighted average
                - 'final_only': Only final denoising step
        """
        super().__init__(device)
        self.method = method
        
        print(f"\nLoading SEDD: {model_name}")
        print(f"Surprisal method: {method}")
        self.model, self.graph, self.noise_fn = load_model(model_name, self.device)
        self.model.eval()
        
        # Create score function (proper way to call SEDD)
        self.score_fn = mutils.get_score_fn(self.model, train=False, sampling=True)
        
        print("✓ SEDD loaded")
    
    def extract_surprisals(self, sentences):
        """Extract SEDD surprisal using specified method."""
        
        if self.method == 'single_pass':
            return self._extract_single_pass(sentences)
        elif self.method == 'trajectory_mean':
            return self._extract_trajectory(sentences, weighted=False)
        elif self.method == 'trajectory_weighted':
            return self._extract_trajectory(sentences, weighted=True)
        elif self.method == 'final_only':
            return self._extract_final_only(sentences)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _extract_single_pass(self, sentences):
        """
        RECOMMENDED: Single forward pass (comparable to GPT-2).
        
        For each sentence:
        1. Encode full sentence
        2. Run SEDD at t≈0 (sigma≈0)
        3. Extract surprisal for each position
        
        This is most comparable to GPT-2's methodology.
        """
        all_results = []
        
        # Small t and sigma (near-deterministic, like t≈0)
        # Create proper tensor format for SEDD
        
        with torch.no_grad():
            for sent_idx, sentence in enumerate(tqdm(sentences, desc=f"SEDD ({self.method})")):
                try:
                    tokens = self.tokenizer.encode(sentence, return_tensors='pt').to(self.device)
                    
                    if tokens.shape[1] > 1024:
                        tokens = tokens[:, :1024]
                    
                    # Create proper sigma tensor for this batch
                    eps = 1e-5
                    t = torch.tensor([[eps]], device=self.device)  # [1, 1]
                    sigma = self.noise_fn(t)[0]  # Get sigma from noise schedule
                    
                    # Single forward pass using score_fn
                    scores = self.score_fn(tokens, sigma)  # [1, seq_len, vocab_size]
                    
                    # Convert score to probabilities (diffusion-specific!)
                    # SEDD score != logits, need proper conversion
                    stag_score = self.graph.staggered_score(scores, sigma)
                    probs = stag_score * self.graph.transp_transition(tokens, sigma)
                    
                    # Handle absorbing state (if present)
                    if self.graph.absorb:
                        probs = probs[..., :-1]  # Remove absorbing state dimension
                        
                except Exception as e:
                    print(f"\n⚠️  Error on sentence {sent_idx}: {e}")
                    print(f"    Sentence: {sentence[:100]}...")
                    print("⚠️  Skipping this sentence...")
                    continue
                
                # Compute surprisal for each position
                for pos in range(tokens.shape[1]):
                    # Get probability of target token
                    target_token = tokens[0, pos].item()
                    target_prob = probs[0, pos, target_token].item()
                    
                    # Clip to avoid log(0)
                    target_prob = max(target_prob, 1e-10)
                    
                    # Surprisal = -log(probability)
                    surprisal = -np.log(target_prob)
                    
                    word = self.tokenizer.decode([target_token])
                    
                    all_results.append({
                        'sentence_idx': sent_idx,
                        'sentence': sentence,
                        'word_position': pos,
                        'word': word,
                        'token_id': target_token,
                        'surprisal_sedd': surprisal,
                    })
        
        return pd.DataFrame(all_results)
    
    def _extract_final_only(self, sentences):
        """
        Extract using only final denoising step.
        
        Similar to single_pass but uses the denoising procedure.
        """
        # For simplicity, this is essentially the same as single_pass
        return self._extract_single_pass(sentences)
    
    def _extract_trajectory(self, sentences, weighted=False, num_steps=256):
        """
        Extract using full diffusion trajectory (SLOW but captures full processing).
        
        For each word:
        1. Run diffusion from t=1.0 to t=0.0
        2. Compute surprisal at each timestep
        3. Average (or weighted average) over trajectory
        
        NOTE: This is VERY slow (num_steps forward passes per sentence)
        """
        all_results = []
        
        eps = 1e-5
        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        
        with torch.no_grad():
            for sent_idx, sentence in enumerate(tqdm(sentences, desc=f"SEDD ({self.method})")):
                try:
                    tokens = self.tokenizer.encode(sentence, return_tensors='pt').to(self.device)
                    
                    if tokens.shape[1] > 1024:
                        tokens = tokens[:, :1024]
                    
                    seq_len = tokens.shape[1]
                    
                    # Storage for trajectory
                    surprisals_over_time = {pos: [] for pos in range(seq_len)}
                    weights_over_time = []
                    
                    # Run diffusion
                    for i, t_val in enumerate(timesteps):
                        t = t_val * torch.ones(1, 1, device=self.device)
                        sigma = self.noise_fn(t)[0]
                        
                        # Forward pass at this timestep using score_fn
                        scores = self.score_fn(tokens, sigma)
                        
                        # Convert score to probabilities (diffusion-specific!)
                        stag_score = self.graph.staggered_score(scores, sigma)
                        probs = stag_score * self.graph.transp_transition(tokens, sigma)
                        
                        # Handle absorbing state
                        if self.graph.absorb:
                            probs = probs[..., :-1]
                    
                        # Compute surprisal for each position
                        for pos in range(seq_len):
                            target_token = tokens[0, pos].item()
                            target_prob = probs[0, pos, target_token].item()
                            target_prob = max(target_prob, 1e-10)  # Clip to avoid log(0)
                            surprisal = -np.log(target_prob)
                            surprisals_over_time[pos].append(surprisal)
                    
                        weights_over_time.append(t_val.item())
                    
                    # Aggregate surprisals
                except Exception as e:
                    print(f"\n⚠️  Error on sentence {sent_idx}: {e}")
                    print(f"    Sentence: {sentence[:100]}...")
                    print("⚠️  Skipping this sentence...")
                    continue
                
                for pos in range(seq_len):
                    surprisals = surprisals_over_time[pos]
                    
                    if weighted:
                        # Time-weighted (emphasize early processing)
                        surprisal_agg = np.average(surprisals, weights=weights_over_time)
                    else:
                        # Simple mean
                        surprisal_agg = np.mean(surprisals)
                    
                    word = self.tokenizer.decode([tokens[0, pos]])
                    
                    all_results.append({
                        'sentence_idx': sent_idx,
                        'sentence': sentence,
                        'word_position': pos,
                        'word': word,
                        'token_id': tokens[0, pos].item(),
                        'surprisal_sedd': surprisal_agg,
                    })
        
        return pd.DataFrame(all_results)


def compare_models(gpt2_results, sedd_results, output_prefix='comparison'):
    """
    Compare GPT-2 and SEDD surprisals.
    
    Args:
        gpt2_results: DataFrame with 'surprisal_gpt2'
        sedd_results: DataFrame with 'surprisal_sedd'
        output_prefix: Prefix for output files
    """
    # Merge results
    merged = pd.merge(
        gpt2_results,
        sedd_results[['sentence_idx', 'word_position', 'surprisal_sedd']],
        on=['sentence_idx', 'word_position'],
        how='inner'
    )
    
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70 + "\n")
    
    # Basic statistics
    print("Summary Statistics:")
    print(f"  Number of words: {len(merged)}")
    print(f"\n  GPT-2 Surprisal:")
    print(f"    Mean: {merged['surprisal_gpt2'].mean():.3f}")
    print(f"    Std:  {merged['surprisal_gpt2'].std():.3f}")
    print(f"    Range: [{merged['surprisal_gpt2'].min():.3f}, {merged['surprisal_gpt2'].max():.3f}]")
    print(f"\n  SEDD Surprisal:")
    print(f"    Mean: {merged['surprisal_sedd'].mean():.3f}")
    print(f"    Std:  {merged['surprisal_sedd'].std():.3f}")
    print(f"    Range: [{merged['surprisal_sedd'].min():.3f}, {merged['surprisal_sedd'].max():.3f}]")
    
    # Correlations
    print(f"\n" + "="*70)
    print("CORRELATIONS")
    print("="*70)
    
    # Pearson correlation
    r_pearson, p_pearson = pearsonr(merged['surprisal_gpt2'], merged['surprisal_sedd'])
    print(f"\n  Pearson correlation:")
    print(f"    r = {r_pearson:.4f}")
    print(f"    p = {p_pearson:.2e}")
    print(f"    Interpretation: {'Strong' if abs(r_pearson) > 0.7 else 'Moderate' if abs(r_pearson) > 0.4 else 'Weak'}")
    
    # Spearman correlation (rank-based, more robust)
    r_spearman, p_spearman = spearmanr(merged['surprisal_gpt2'], merged['surprisal_sedd'])
    print(f"\n  Spearman correlation (rank-based):")
    print(f"    ρ = {r_spearman:.4f}")
    print(f"    p = {p_spearman:.2e}")
    
    # Save merged data
    merged.to_csv(f'{output_prefix}_merged.csv', index=False)
    print(f"\n✓ Saved merged data: {output_prefix}_merged.csv")
    
    # Create visualizations
    _create_visualizations(merged, output_prefix)
    
    # Analysis by word characteristics
    _analyze_by_characteristics(merged, output_prefix)
    
    return merged


def _create_visualizations(merged, output_prefix):
    """Create comparison plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Scatter plot
    ax = axes[0, 0]
    ax.scatter(merged['surprisal_gpt2'], merged['surprisal_sedd'], 
               alpha=0.5, s=20)
    
    # Add regression line
    from scipy.stats import linregress
    slope, intercept, r, p, se = linregress(merged['surprisal_gpt2'], 
                                             merged['surprisal_sedd'])
    x_line = np.array([merged['surprisal_gpt2'].min(), 
                       merged['surprisal_gpt2'].max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'r={r:.3f}')
    
    ax.set_xlabel('GPT-2 Surprisal', fontsize=12)
    ax.set_ylabel('SEDD Surprisal', fontsize=12)
    ax.set_title('GPT-2 vs SEDD Surprisal', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Distribution comparison
    ax = axes[0, 1]
    ax.hist(merged['surprisal_gpt2'], bins=50, alpha=0.5, label='GPT-2', 
            density=True, color='blue')
    ax.hist(merged['surprisal_sedd'], bins=50, alpha=0.5, label='SEDD', 
            density=True, color='red')
    ax.set_xlabel('Surprisal', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution Comparison', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Difference analysis
    ax = axes[1, 0]
    merged['difference'] = merged['surprisal_sedd'] - merged['surprisal_gpt2']
    ax.hist(merged['difference'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Difference (SEDD - GPT-2)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Surprisal Differences', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    mean_diff = merged['difference'].mean()
    ax.text(0.05, 0.95, f'Mean diff: {mean_diff:.3f}', 
            transform=ax.transAxes, fontsize=10, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Bland-Altman plot
    ax = axes[1, 1]
    mean_surprisal = (merged['surprisal_gpt2'] + merged['surprisal_sedd']) / 2
    diff_surprisal = merged['surprisal_sedd'] - merged['surprisal_gpt2']
    
    ax.scatter(mean_surprisal, diff_surprisal, alpha=0.5, s=20)
    ax.axhline(diff_surprisal.mean(), color='red', linestyle='-', 
               linewidth=2, label=f'Mean: {diff_surprisal.mean():.3f}')
    ax.axhline(diff_surprisal.mean() + 1.96*diff_surprisal.std(), 
               color='red', linestyle='--', linewidth=1)
    ax.axhline(diff_surprisal.mean() - 1.96*diff_surprisal.std(), 
               color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Mean Surprisal (GPT-2 + SEDD) / 2', fontsize=12)
    ax.set_ylabel('Difference (SEDD - GPT-2)', fontsize=12)
    ax.set_title('Bland-Altman Plot', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_plots.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved plots: {output_prefix}_plots.png")
    plt.close()


def _analyze_by_characteristics(merged, output_prefix):
    """Analyze correlation by word characteristics."""
    
    print("\n" + "="*70)
    print("ANALYSIS BY WORD CHARACTERISTICS")
    print("="*70)
    
    # By position in sentence
    merged['position_bin'] = pd.cut(merged['word_position'], 
                                     bins=[0, 5, 10, 20, 1000],
                                     labels=['0-5', '6-10', '11-20', '20+'])
    
    print("\nCorrelation by word position:")
    for bin_name, group in merged.groupby('position_bin'):
        if len(group) > 10:
            r, p = pearsonr(group['surprisal_gpt2'], group['surprisal_sedd'])
            print(f"  Position {bin_name}: r={r:.3f}, n={len(group)}")
    
    # By surprisal level (GPT-2)
    merged['surprisal_bin'] = pd.cut(merged['surprisal_gpt2'],
                                      bins=[0, 2, 5, 10, 100],
                                      labels=['Low (0-2)', 'Med (2-5)', 
                                             'High (5-10)', 'VHigh (10+)'])
    
    print("\nCorrelation by surprisal level:")
    for bin_name, group in merged.groupby('surprisal_bin'):
        if len(group) > 10:
            r, p = pearsonr(group['surprisal_gpt2'], group['surprisal_sedd'])
            print(f"  {bin_name}: r={r:.3f}, n={len(group)}")
    
    # Words where models disagree most
    merged['abs_diff'] = (merged['surprisal_sedd'] - 
                          merged['surprisal_gpt2']).abs()
    
    print("\nTop 10 words where models disagree most:")
    top_disagreements = merged.nlargest(10, 'abs_diff')
    print(top_disagreements[['word', 'surprisal_gpt2', 'surprisal_sedd', 
                              'abs_diff']].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description='Compare SEDD and GPT-2 surprisal on natural sentences'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV with sentences (column: "sentence" or "Sentence")')
    parser.add_argument('--output-prefix', type=str, default='sedd_gpt2_comparison',
                       help='Prefix for output files')
    parser.add_argument('--sedd-method', type=str, default='single_pass',
                       choices=['single_pass', 'trajectory_mean', 
                               'trajectory_weighted', 'final_only'],
                       help='Method for SEDD surprisal extraction')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu')
    parser.add_argument('--max-sentences', type=int, default=None,
                       help='Limit number of sentences (for testing)')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Force CPU mode (use if CUDA errors occur)')
    
    args = parser.parse_args()
    
    # Override device if force-cpu
    if args.force_cpu:
        args.device = 'cpu'
        print("\n⚠️  Forcing CPU mode (--force-cpu flag set)")
    
    print("\n" + "="*70)
    print(" SEDD vs GPT-2 SURPRISAL COMPARISON ".center(70))
    print("="*70)
    print(f"\nInput: {args.input}")
    print(f"SEDD method: {args.sedd_method}")
    print(f"Device: {args.device}")
    
    # Load sentences
    df = pd.read_csv(args.input)
    
    # Find sentence column
    sent_col = None
    for col in ['sentence', 'Sentence', 'text', 'Text']:
        if col in df.columns:
            sent_col = col
            break
    
    if sent_col is None:
        raise ValueError("No sentence column found. Need 'sentence' or 'Sentence'")
    
    sentences = df[sent_col].tolist()
    
    if args.max_sentences:
        sentences = sentences[:args.max_sentences]
    
    print(f"\nProcessing {len(sentences)} sentences")
    
    # Extract GPT-2 surprisals
    print("\n" + "="*70)
    print("EXTRACTING GPT-2 SURPRISAL")
    print("="*70)
    gpt2_extractor = GPT2SurprisalExtractor(device=args.device)
    gpt2_results = gpt2_extractor.extract_surprisals(sentences)
    print(f"✓ Extracted {len(gpt2_results)} word surprisals from GPT-2")
    
    # Extract SEDD surprisals
    print("\n" + "="*70)
    print("EXTRACTING SEDD SURPRISAL")
    print("="*70)
    sedd_extractor = SEDDSurprisalExtractor(
        device=args.device,
        method=args.sedd_method
    )
    sedd_results = sedd_extractor.extract_surprisals(sentences)
    print(f"✓ Extracted {len(sedd_results)} word surprisals from SEDD")
    
    # Compare
    print("\n" + "="*70)
    print("COMPARING MODELS")
    print("="*70)
    merged = compare_models(gpt2_results, sedd_results, args.output_prefix)
    
    print("\n" + "="*70)
    print("✓ COMPARISON COMPLETE")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  - {args.output_prefix}_merged.csv")
    print(f"  - {args.output_prefix}_plots.png")
    print("\n")


if __name__ == '__main__':
    main()

