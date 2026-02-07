"""
SEDD Trajectory Extractor - Soft Left-to-Right Projection
Allows previous words to remain ambiguous (more realistic)
"""

from transformers import GPT2TokenizerFast
from model import utils as mutils
from sampling import get_predictor, Denoiser
import sampling
from load_model import load_model
import torch
import torch.nn.functional as F
import pandas as pd
import argparse
from tqdm import tqdm
import numpy as np


class SoftProjectionExtractor:
    """
    Extract trajectories with SOFT left-to-right projection.
    
    Key difference from hard projection:
    - Previous words are NOT locked to ground truth
    - They retain some uncertainty (noise)
    - More realistic model of human incremental processing
    """

    def __init__(self, model_name="louaaron/sedd-medium", device='cuda',
                 num_steps=1024, noise_schedule='exponential', base_noise=0.3):
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu')
        self.num_steps = num_steps
        self.noise_schedule = noise_schedule
        self.base_noise = base_noise

        print(f"\n{'='*60}")
        print(f"Loading SEDD with SOFT Projection")
        print(f"Device: {self.device}")
        print(f"Noise schedule: {noise_schedule}")
        print(f"Base noise level: {base_noise}")
        print(f"{'='*60}\n")

        self.model, self.graph, self.noise = load_model(
            model_name, self.device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

        print("✓ Model loaded successfully")

    def compute_noise_level(self, distance_from_current):
        """
        Compute noise level for a previous position.
        
        Args:
            distance_from_current: How many positions before current
                                   (1 = immediately before, 2 = two positions back, etc.)
        
        Returns:
            noise_level: Probability of replacing with noise (0-1)
                        0 = fully certain (always correct)
                        1 = fully uncertain (always noise)
        """
        if self.noise_schedule == 'exponential':
            # Exponential decay: recent words more uncertain
            # distance=1 → ~0.22, distance=2 → ~0.16, distance=5 → ~0.05
            return self.base_noise * np.exp(-distance_from_current / 2)
        
        elif self.noise_schedule == 'linear':
            # Linear decay
            return max(0, self.base_noise * (1 - distance_from_current / 10))
        
        elif self.noise_schedule == 'step':
            # Step function: recent words uncertain, old words certain
            if distance_from_current <= 2:
                return self.base_noise
            elif distance_from_current <= 5:
                return self.base_noise * 0.3
            else:
                return 0.01  # Almost certain
        
        elif self.noise_schedule == 'none':
            # No noise (equivalent to hard projection)
            return 0.0
        
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")

    def soft_left_to_right_projection(self, x, tokens, current_pos):
        """
        Soft projection: previous words retain some uncertainty.
        
        Args:
            x: Current state [batch, seq_len]
            tokens: Ground truth tokens [batch, seq_len]
            current_pos: Position currently being processed
        
        Returns:
            x: State with soft left context
        """
        batch_size = x.shape[0]
        
        for pos in range(current_pos):
            distance = current_pos - pos
            noise_level = self.compute_noise_level(distance)
            
            if noise_level > 0:
                # Probabilistic replacement
                # Higher noise_level = more likely to be noisy
                for b in range(batch_size):
                    if torch.rand(1).item() < noise_level:
                        # Replace with noise (sample from vocabulary)
                        x[b, pos] = self.graph.sample_limit(1, 1).squeeze()
                    else:
                        # Keep ground truth
                        x[b, pos] = tokens[b, pos]
            else:
                # No noise: use ground truth
                x[:, pos] = tokens[:, pos]
        
        return x

    def extract_trajectories_soft(self, sentence, save_every=10):
        """
        Extract trajectories with soft projection (word-by-word).
        
        Args:
            sentence: Input sentence string
            save_every: Save trajectory every N steps
        
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
        print(f"  Mode: Soft projection (previous words remain ambiguous)")

        all_word_metrics = []

        # Process word-by-word (chunk_size=1)
        for current_pos in range(seq_len):
            print(f"  Position {current_pos}: '{self.tokenizer.decode([tokens[0, current_pos]])}'")
            
            # Show noise levels for context
            if current_pos > 0:
                print(f"    Context noise levels:")
                for prev_pos in range(max(0, current_pos - 5), current_pos):
                    distance = current_pos - prev_pos
                    noise = self.compute_noise_level(distance)
                    word = self.tokenizer.decode([tokens[0, prev_pos]])
                    print(f"      Position {prev_pos} ('{word}'): {noise:.2%} uncertain")

            # Process this position
            metrics = self._process_position_soft(
                tokens, current_pos, batch_size, max_len, save_every
            )
            
            all_word_metrics.append(metrics)

        return all_word_metrics

    def _process_position_soft(self, tokens, current_pos, batch_size, max_len, save_every):
        """Process one position with soft projection."""

        # Setup sampling components
        predictor = get_predictor('analytic')(self.graph, self.noise)
        denoiser = Denoiser(self.graph, self.noise)
        sampling_score_fn = mutils.get_score_fn(
            self.model, train=False, sampling=True)

        # Initialize from noise
        batch_dims = (batch_size, max_len)
        x = self.graph.sample_limit(*batch_dims).to(self.device)

        # Timestep schedule
        eps = 1e-5
        timesteps = torch.linspace(
            1, eps, self.num_steps + 1, device=self.device)
        dt = (1 - eps) / self.num_steps

        # Storage for trajectory
        trajectory = []

        # Run reverse diffusion
        with torch.no_grad():
            for i in range(self.num_steps):
                t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)

                # Apply SOFT projection
                x = self.soft_left_to_right_projection(x, tokens, current_pos)

                # Compute score
                curr_sigma = self.noise(t)[0]
                score = sampling_score_fn(x, curr_sigma)

                # Save trajectory
                if i % save_every == 0:
                    current_token = x[0, current_pos].item()
                    target_token = tokens[0, current_pos].item()
                    
                    # Score metrics
                    probs = F.softmax(score[0, current_pos], dim=-1)
                    log_probs = F.log_softmax(score[0, current_pos], dim=-1)
                    
                    trajectory.append({
                        'step': i,
                        'timestep': t[0, 0].item(),
                        'token': current_token,
                        'target_token': target_token,
                        'is_correct': current_token == target_token,
                        'surprisal': -log_probs[target_token].item(),
                        'entropy': -(probs * log_probs).sum().item(),
                        'target_prob': probs[target_token].item(),
                    })

                # Predictor step
                x = predictor.update_fn(sampling_score_fn, x, t, dt)

            # Final denoising
            x = self.soft_left_to_right_projection(x, tokens, current_pos)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            x = denoiser.update_fn(sampling_score_fn, x, t)

            # Save final state
            current_token = x[0, current_pos].item()
            target_token = tokens[0, current_pos].item()
            
            final_sigma = self.noise(t)[0]
            final_score = sampling_score_fn(x, final_sigma)
            probs = F.softmax(final_score[0, current_pos], dim=-1)
            log_probs = F.log_softmax(final_score[0, current_pos], dim=-1)
            
            trajectory.append({
                'step': self.num_steps,
                'timestep': eps,
                'token': current_token,
                'target_token': target_token,
                'is_correct': current_token == target_token,
                'surprisal': -log_probs[target_token].item(),
                'entropy': -(probs * log_probs).sum().item(),
                'target_prob': probs[target_token].item(),
            })

        # Compute metrics
        metrics = self._compute_metrics(trajectory, current_pos, tokens)
        return metrics

    def _compute_metrics(self, trajectory, position, tokens):
        """Compute aggregate metrics from trajectory."""
        
        surprisals = [t['surprisal'] for t in trajectory]
        entropies = [t['entropy'] for t in trajectory]
        target_probs = [t['target_prob'] for t in trajectory]
        is_correct = [t['is_correct'] for t in trajectory]
        tokens_over_time = [t['token'] for t in trajectory]
        timesteps = [t['timestep'] for t in trajectory]

        word = self.tokenizer.decode([tokens[0, position]])

        # Convergence
        convergence_timestep = timesteps[-1] if is_correct[-1] else 1.0
        for i, correct in enumerate(is_correct):
            if correct and all(is_correct[i:]):
                convergence_timestep = timesteps[i]
                break

        # Token changes
        token_changes = sum(
            1 for i in range(1, len(tokens_over_time))
            if tokens_over_time[i] != tokens_over_time[i-1]
        )

        metrics = {
            'word_position': position,
            'word': word,
            'target_token_id': tokens[0, position].item(),
            
            'surprisal_mean': np.mean(surprisals),
            'surprisal_final': surprisals[-1],
            'entropy_mean': np.mean(entropies),
            'entropy_final': entropies[-1],
            'target_prob_final': target_probs[-1],
            
            'convergence_timestep': convergence_timestep,
            'token_changes': token_changes,
            'correctness_ratio': sum(is_correct) / len(is_correct),
            'final_is_correct': is_correct[-1],
        }

        return metrics


def compare_projections(sentence):
    """Compare hard vs soft projection on same sentence."""
    
    print("\n" + "="*70)
    print(" COMPARING HARD vs SOFT PROJECTION ".center(70))
    print("="*70 + "\n")
    
    results = {}
    
    # Test different noise schedules
    schedules = ['none', 'exponential', 'step']
    
    for schedule in schedules:
        print(f"\n{'='*70}")
        print(f"Testing: {schedule.upper()} noise schedule")
        print(f"{'='*70}")
        
        extractor = SoftProjectionExtractor(
            num_steps=256,  # Reduced for testing
            noise_schedule=schedule,
            base_noise=0.3
        )
        
        metrics = extractor.extract_trajectories_soft(
            sentence, save_every=20
        )
        
        df = pd.DataFrame(metrics)
        results[schedule] = df
        
        print(f"\nResults ({schedule}):")
        print(df[['word', 'surprisal_mean', 'token_changes']].to_string(index=False))
    
    # Compare
    print("\n" + "="*70)
    print(" COMPARISON SUMMARY ".center(70))
    print("="*70 + "\n")
    
    for schedule in schedules:
        df = results[schedule]
        print(f"{schedule.upper()} projection:")
        print(f"  Mean surprisal: {df['surprisal_mean'].mean():.3f}")
        print(f"  Mean token changes: {df['token_changes'].mean():.3f}")
        print(f"  Tokens with changes > 0: {(df['token_changes'] > 0).sum()}/{len(df)}")
        print()
    
    print("INTERPRETATION:")
    print("  'none' = Hard projection (old method)")
    print("  'exponential' = Recent words ambiguous")
    print("  'step' = Last 2-3 words ambiguous")
    print()
    print("Expected: Soft projection → MORE variation in metrics")
    print("="*70 + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Test soft projection for more realistic incremental processing'
    )
    parser.add_argument('--mode', type=str, choices=['test', 'compare'], default='compare',
                       help='test = single run, compare = compare methods')
    parser.add_argument('--sentence', type=str, 
                       default="The cat sat on the mat.",
                       help='Test sentence')
    parser.add_argument('--noise-schedule', type=str, 
                       choices=['exponential', 'linear', 'step', 'none'],
                       default='exponential',
                       help='Noise schedule for soft projection')
    parser.add_argument('--base-noise', type=float, default=0.3,
                       help='Base noise level (0-1)')
    
    args = parser.parse_args()
    
    if args.mode == 'compare':
        results = compare_projections(args.sentence)
    else:
        extractor = SoftProjectionExtractor(
            num_steps=1024,
            noise_schedule=args.noise_schedule,
            base_noise=args.base_noise
        )
        
        metrics = extractor.extract_trajectories_soft(
            args.sentence, save_every=10
        )
        
        df = pd.DataFrame(metrics)
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(df[['word', 'surprisal_mean', 'entropy_mean', 'token_changes']].to_string(index=False))
        print("="*70 + "\n")


if __name__ == '__main__':
    main()

