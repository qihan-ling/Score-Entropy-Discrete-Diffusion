"""
SEDD Trajectory Extractor - Left-to-Right Incremental Processing
Complete implementation with progressive projection
"""

from transformers import GPT2TokenizerFast
from model import utils as mutils
from sampling import get_predictor, Denoiser
import sampling
from load_model import load_model
import torch
import pandas as pd
import argparse
from tqdm import tqdm
import sys
import os


class SEDDTrajectoryExtractor:
    """Extract diffusion trajectories using left-to-right incremental processing."""

    def __init__(self, model_name="louaaron/sedd-medium", device='cuda',
                 num_steps=1024):
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu')
        self.num_steps = num_steps

        print(f"\n{'='*60}")
        print(f"Loading SEDD: {model_name}")
        print(f"Device: {self.device}")
        print(f"Steps: {num_steps}")
        print(f"{'='*60}\n")

        self.model, self.graph, self.noise = load_model(
            model_name, self.device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

        print("✓ Model loaded successfully")

    def extract_trajectories(self, sentence, save_every=10, chunk_size=5):
        """
        Extract trajectories using progressive left-to-right projection.

        For each word, we:
        1. Fix all words to its left (context)
        2. Let current word evolve from noise
        3. Track how it converges

        Args:
            sentence: Input sentence string
            save_every: Save trajectory every N steps
            chunk_size: Process this many words per diffusion run (tradeoff speed/accuracy)

        Returns:
            word_metrics: List of dicts with metrics per word
        """

        # Tokenize
        tokens = self.tokenizer.encode(
            sentence, return_tensors='pt').to(self.device)
        batch_size, seq_len = tokens.shape

        max_len = 1024
        if seq_len > max_len:
            tokens = tokens[:, :max_len]
            seq_len = max_len

        print(f"\nProcessing: '{sentence}'")
        print(f"  Tokens: {seq_len}")
        print(f"  Chunks: {(seq_len + chunk_size - 1) // chunk_size}")

        all_word_metrics = []

        # Process sentence in chunks for efficiency
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk_positions = list(range(chunk_start, chunk_end))

            chunk_words = [self.tokenizer.decode(
                [tokens[0, p]]) for p in chunk_positions]
            print(f"  Chunk [{chunk_start}:{chunk_end}]: {chunk_words}")

            # Run diffusion for this chunk
            chunk_metrics = self._process_chunk(
                tokens, chunk_start, chunk_end,
                batch_size, max_len, save_every
            )

            all_word_metrics.extend(chunk_metrics)

        return all_word_metrics

    def _process_chunk(self, tokens, chunk_start, chunk_end, batch_size, max_len, save_every):
        """Process one chunk of words with left-to-right projection."""

        chunk_positions = list(range(chunk_start, chunk_end))

        # Define projection function for this chunk
        def left_to_right_projection(x):
            """
            Fix all positions BEFORE chunk_start (left context).
            Let chunk positions evolve naturally.
            """
            if chunk_start > 0:
                # Project (fix) everything to the left
                x[:, :chunk_start] = tokens[:, :chunk_start]
            # Positions chunk_start:chunk_end are FREE (not projected)
            # Positions after chunk_end are also FREE (future context)
            return x

        # Storage for trajectories
        trajectories = {pos: [] for pos in chunk_positions}

        # Setup sampling components
        predictor = get_predictor('analytic')(self.graph, self.noise)
        denoiser = Denoiser(self.graph, self.noise)
        sampling_score_fn = mutils.get_score_fn(
            self.model, train=False, sampling=True)

        # Initialize from noise (limit distribution)
        batch_dims = (batch_size, max_len)
        x = self.graph.sample_limit(*batch_dims).to(self.device)

        # Timestep schedule
        eps = 1e-5
        timesteps = torch.linspace(
            1, eps, self.num_steps + 1, device=self.device)
        dt = (1 - eps) / self.num_steps

        # Run reverse diffusion
        with torch.no_grad():
            for i in range(self.num_steps):
                t = timesteps[i] * \
                    torch.ones(x.shape[0], 1, device=self.device)

                # CRITICAL: Save state BEFORE projection
                # This captures the natural evolution of tokens
                if i % save_every == 0:
                    for pos in chunk_positions:
                        trajectories[pos].append({
                            'step': i,
                            'timestep': t[0, 0].item(),
                            'token': x[0, pos].item(),
                            'target_token': tokens[0, pos].item(),
                            'is_correct': x[0, pos].item() == tokens[0, pos].item()
                        })

                # Apply left-to-right projection
                x = left_to_right_projection(x)

                # Predictor step (model processes with causal mask if enabled)
                x = predictor.update_fn(sampling_score_fn, x, t, dt)

            # Final denoising step
            x = left_to_right_projection(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            x = denoiser.update_fn(sampling_score_fn, x, t)

            # Save final state
            for pos in chunk_positions:
                trajectories[pos].append({
                    'step': self.num_steps,
                    'timestep': eps,
                    'token': x[0, pos].item(),
                    'target_token': tokens[0, pos].item(),
                    'is_correct': x[0, pos].item() == tokens[0, pos].item()
                })

        # Compute metrics for each word in chunk
        chunk_metrics = []
        for pos in chunk_positions:
            metrics = self._compute_word_metrics(
                trajectories[pos], pos, tokens)
            chunk_metrics.append(metrics)

        return chunk_metrics

    def _compute_word_metrics(self, trajectory, position, tokens):
        """Compute metrics from a single word's trajectory."""

        if len(trajectory) == 0:
            return None

        # Extract time series
        steps = [t['step'] for t in trajectory]
        timesteps = [t['timestep'] for t in trajectory]
        tokens_over_time = [t['token'] for t in trajectory]
        is_correct = [t['is_correct'] for t in trajectory]

        # Get word string
        word = self.tokenizer.decode([tokens[0, position]])

        # 1. Convergence timestep (when it becomes correct and STAYS correct)
        convergence_timestep = None
        for i, correct in enumerate(is_correct):
            if correct and all(is_correct[i:]):  # Correct from here onwards
                convergence_timestep = timesteps[i]
                break

        if convergence_timestep is None:
            # Never converged or didn't stay correct
            convergence_timestep = timesteps[-1] if is_correct[-1] else 1.0

        # 2. First correct timestep (when it first becomes correct, may change later)
        first_correct_timestep = None
        for i, correct in enumerate(is_correct):
            if correct:
                first_correct_timestep = timesteps[i]
                break
        if first_correct_timestep is None:
            first_correct_timestep = 1.0

        # 3. Token changes (trajectory variability)
        token_changes = sum(
            1 for i in range(1, len(tokens_over_time))
            if tokens_over_time[i] != tokens_over_time[i-1]
        )

        # 4. Correctness ratio
        correct_count = sum(is_correct)
        correctness_ratio = correct_count / \
            len(is_correct) if len(is_correct) > 0 else 0.0

        # 5. Instability (how many times it was wrong after being correct)
        instability = 0
        was_correct = False
        for correct in is_correct:
            if was_correct and not correct:
                instability += 1
            was_correct = correct

        # 6. Unique tokens visited
        unique_tokens = len(set(tokens_over_time))

        # 7. Early vs late correctness
        mid = len(is_correct) // 2
        early_correctness = sum(is_correct[:mid]) / mid if mid > 0 else 0.0
        late_correctness = sum(
            is_correct[mid:]) / (len(is_correct) - mid) if len(is_correct) > mid else 0.0

        # Package metrics
        metrics = {
            'word_position': position,
            'word': word,
            'target_token_id': tokens[0, position].item(),
            'convergence_timestep': convergence_timestep,
            'first_correct_timestep': first_correct_timestep,
            'token_changes': token_changes,
            'correctness_ratio': correctness_ratio,
            'instability': instability,
            'unique_tokens_visited': unique_tokens,
            'early_correctness': early_correctness,
            'late_correctness': late_correctness,
            'final_is_correct': is_correct[-1],
            'final_token': tokens_over_time[-1]
        }

        return metrics


def process_csv(input_csv, output_csv, model_name="louaaron/sedd-medium",
                num_steps=1024, device='cuda', save_every=10, chunk_size=5,):
    """
    Process CSV file with sentences and extract trajectories.

    Args:
        input_csv: Path to CSV with 'Sentence' column
        output_csv: Path to save results
        model_name: SEDD model name
        num_steps: Number of diffusion timesteps
        device: 'cuda' or 'cpu'
        save_every: Save trajectory every N steps
        chunk_size: Words per diffusion run (tradeoff: larger=faster but less accurate left context)
    """

    print(f"\n{'='*60}")
    print(f"SEDD Left-to-Right Trajectory Extraction")
    print(f"{'='*60}")
    print(f"Input: {input_csv}")
    print(f"Output: {output_csv}")
    print(f"Chunk size: {chunk_size} words")
    print(f"{'='*60}\n")

    # Load CSV
    whole_df = pd.read_csv(input_csv)
    # Test the first sentence
    df = whole_df.iloc[0:1]
    if 'Sentence' not in df.columns:
        raise ValueError("CSV must have 'Sentence' column")

    print(f"Found {len(df)} sentences to process\n")

    # Initialize extractor
    extractor = SEDDTrajectoryExtractor(
        model_name=model_name,
        device=device,
        num_steps=num_steps,
    )

    # Process each sentence
    all_results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing sentences"):
        sentence = row['Sentence']

        try:
            # Extract trajectories with left-to-right projection
            word_metrics = extractor.extract_trajectories(
                sentence,
                save_every=save_every,
                chunk_size=chunk_size
            )

            # Add sentence-level info to each word
            for word_metric in word_metrics:
                if word_metric is not None:
                    result = {
                        'sentence_id': idx,
                        'sentence': sentence,
                        'chunk_size': chunk_size,
                        **{col: row[col] for col in df.columns if col != 'Sentence'},
                        **word_metric
                    }
                    all_results.append(result)

        except Exception as e:
            print(f"\n✗ Error on sentence {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    if len(all_results) == 0:
        print("\n✗ No results to save!")
        return None

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_csv, index=False)

    print(f"\n{'='*60}")
    print(f"✓ Results saved: {output_csv}")
    print(f"✓ Total word tokens: {len(results_df)}")
    print(f"{'='*60}\n")

    # Summary statistics
    print("=== Trajectory Metrics Summary ===")
    metric_cols = [
        'convergence_timestep', 'token_changes', 'correctness_ratio',
        'instability', 'unique_tokens_visited'
    ]

    for col in metric_cols:
        if col in results_df.columns:
            mean_val = results_df[col].mean()
            std_val = results_df[col].std()
            min_val = results_df[col].min()
            max_val = results_df[col].max()
            print(f"{col:30s}: μ={mean_val:6.3f}, σ={std_val:6.3f}, "
                  f"range=[{min_val:6.3f}, {max_val:6.3f}]")

    # Show some example trajectories
    print(f"\n=== Example Word Trajectories ===")
    sample_words = results_df.head(5)[['word', 'convergence_timestep',
                                       'token_changes', 'correctness_ratio']]
    print(sample_words.to_string(index=False))

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Extract SEDD trajectories with left-to-right incremental processing'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV with Sentence column')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV for trajectory metrics')
    parser.add_argument('--model', type=str, default='louaaron/sedd-medium',
                        help='SEDD model name (default: louaaron/sedd-medium)')
    parser.add_argument('--steps', type=int, default=1024,
                        help='Number of diffusion timesteps (default: 1024)')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save trajectory every N steps (default: 10)')
    parser.add_argument('--chunk-size', type=int, default=5,
                        help='Words per diffusion run (default: 5, larger=faster but less accurate)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu (default: cuda)')

    args = parser.parse_args()

    # Process CSV
    results_df = process_csv(
        input_csv=args.input,
        output_csv=args.output,
        model_name=args.model,
        num_steps=args.steps,
        device=args.device,
        save_every=args.save_every,
        chunk_size=args.chunk_size,
    )

    if results_df is not None:
        print("\n✓ Processing complete!")
    else:
        print("\n✗ Processing failed!")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
