"""
SEDD Full Trajectory Tracking with Soft Projection
Tracks how each word evolves not only during its processing,
but also how it appears as context for later words.
"""

from transformers import GPT2TokenizerFast
from model import utils as mutils
from sampling import get_predictor, Denoiser
import sampling
from load_model import load_model
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import pickle


class FullTrackingExtractor:
    """
    Extract complete processing trajectories including:
    1. Primary trajectory: Full diffusion when word is focus
    2. Context states: How word appears when processing later words
    """

    def __init__(self, model_name="louaaron/sedd-medium", device='cuda',
                 num_steps=1024, noise_schedule='exponential', base_noise=0.3):
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu')
        self.num_steps = num_steps
        self.noise_schedule = noise_schedule
        self.base_noise = base_noise

        print(f"\n{'='*60}")
        print(f"FULL TRACKING with Soft Projection")
        print(f"{'='*60}\n")

        self.model, self.graph, self.noise = load_model(
            model_name, self.device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    def compute_noise_level(self, distance):
        """Compute noise level based on distance from current position."""
        if self.noise_schedule == 'exponential':
            return self.base_noise * np.exp(-distance / 2)
        elif self.noise_schedule == 'none':
            return 0.0
        else:
            return self.base_noise * max(0, 1 - distance / 10)

    def soft_projection_with_tracking(self, x, tokens, current_pos):
        """
        Soft projection that TRACKS what happens to previous positions.
        
        Returns:
            x: Projected state
            context_states: Dict mapping position → state info
        """
        context_states = {}
        
        for pos in range(current_pos):
            distance = current_pos - pos
            noise_level = self.compute_noise_level(distance)
            
            # Sample: correct or noise?
            if torch.rand(1).item() < noise_level:
                sampled_token = self.graph.sample_limit(1, 1).squeeze().item()
                is_correct = False
            else:
                sampled_token = tokens[0, pos].item()
                is_correct = True
            
            x[0, pos] = sampled_token
            
            # TRACK what happened
            context_states[pos] = {
                'token_sampled': sampled_token,
                'target_token': tokens[0, pos].item(),
                'is_correct': is_correct,
                'noise_level': noise_level,
                'certainty': 1 - noise_level,
                'processing_stage': current_pos,  # When this state occurred
            }
        
        return x, context_states

    def extract_full_trajectories(self, sentence, save_every=10):
        """
        Extract complete trajectories with full tracking.
        
        Returns:
            full_data: Dict with:
                - primary_trajectories: Main processing for each word
                - context_states: How words appear as context later
                - matrix_view: 2D view of all states
        """
        tokens = self.tokenizer.encode(
            sentence, return_tensors='pt').to(self.device)
        batch_size, seq_len = tokens.shape

        max_len = 1024
        if seq_len > max_len:
            tokens = tokens[:, :max_len]
            seq_len = max_len

        print(f"\nProcessing: '{sentence}'")
        print(f"Tokens: {seq_len}")
        print(f"Full tracking: {seq_len} × {seq_len} matrix\n")

        # Storage for all data
        full_data = {
            'sentence': sentence,
            'tokens': [self.tokenizer.decode([tokens[0, i]]) for i in range(seq_len)],
            'token_ids': tokens[0].cpu().numpy(),
            'primary_trajectories': {},  # position → trajectory when processing that position
            'context_states': {},  # (position, stage) → state when used as context
            'noise_schedule': self.noise_schedule,
            'base_noise': self.base_noise,
        }

        # Process each position
        for current_pos in tqdm(range(seq_len), desc="Processing positions"):
            word = self.tokenizer.decode([tokens[0, current_pos]])
            print(f"\n  Position {current_pos}: '{word}'")

            # Get primary trajectory AND context states
            primary_traj, context_states_at_stage = self._process_position_full(
                tokens, current_pos, batch_size, max_len, save_every
            )

            # Save primary trajectory
            full_data['primary_trajectories'][current_pos] = primary_traj

            # Save context states
            for pos, state in context_states_at_stage.items():
                key = (pos, current_pos)  # (which word, which processing stage)
                full_data['context_states'][key] = state

        # Create matrix view for easy analysis
        full_data['matrix_view'] = self._create_matrix_view(full_data, seq_len)

        return full_data

    def _process_position_full(self, tokens, current_pos, batch_size, max_len, save_every):
        """
        Process one position and track BOTH:
        1. Primary trajectory for current position
        2. Context states for previous positions
        """
        predictor = get_predictor('analytic')(self.graph, self.noise)
        denoiser = Denoiser(self.graph, self.noise)
        sampling_score_fn = mutils.get_score_fn(
            self.model, train=False, sampling=True)

        # Initialize
        batch_dims = (batch_size, max_len)
        x = self.graph.sample_limit(*batch_dims).to(self.device)

        eps = 1e-5
        timesteps = torch.linspace(1, eps, self.num_steps + 1, device=self.device)
        dt = (1 - eps) / self.num_steps

        # Storage
        primary_trajectory = []
        context_states_accumulated = {}

        # Diffusion loop
        with torch.no_grad():
            for i in range(self.num_steps):
                t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)

                # Apply soft projection WITH tracking
                x, context_states = self.soft_projection_with_tracking(
                    x, tokens, current_pos
                )

                # Accumulate context states (track how often each position is correct)
                for pos, state in context_states.items():
                    if pos not in context_states_accumulated:
                        context_states_accumulated[pos] = {
                            'correct_count': 0,
                            'total_count': 0,
                            'tokens_seen': [],
                            'noise_level': state['noise_level'],
                        }
                    context_states_accumulated[pos]['total_count'] += 1
                    if state['is_correct']:
                        context_states_accumulated[pos]['correct_count'] += 1
                    context_states_accumulated[pos]['tokens_seen'].append(
                        state['token_sampled']
                    )

                # Compute score for PRIMARY position
                curr_sigma = self.noise(t)[0]
                score = sampling_score_fn(x, curr_sigma)

                # Save primary trajectory
                if i % save_every == 0:
                    current_token = x[0, current_pos].item()
                    target_token = tokens[0, current_pos].item()

                    probs = F.softmax(score[0, current_pos], dim=-1)
                    log_probs = F.log_softmax(score[0, current_pos], dim=-1)

                    primary_trajectory.append({
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
            x, context_states = self.soft_projection_with_tracking(
                x, tokens, current_pos
            )
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            x = denoiser.update_fn(sampling_score_fn, x, t)

            # Save final primary state
            current_token = x[0, current_pos].item()
            target_token = tokens[0, current_pos].item()

            final_sigma = self.noise(t)[0]
            final_score = sampling_score_fn(x, final_sigma)
            probs = F.softmax(final_score[0, current_pos], dim=-1)
            log_probs = F.log_softmax(final_score[0, current_pos], dim=-1)

            primary_trajectory.append({
                'step': self.num_steps,
                'timestep': eps,
                'token': current_token,
                'target_token': target_token,
                'is_correct': current_token == target_token,
                'surprisal': -log_probs[target_token].item(),
                'entropy': -(probs * log_probs).sum().item(),
                'target_prob': probs[target_token].item(),
            })

        # Summarize context states
        for pos in context_states_accumulated:
            context_states_accumulated[pos]['empirical_certainty'] = (
                context_states_accumulated[pos]['correct_count'] /
                context_states_accumulated[pos]['total_count']
            )

        return primary_trajectory, context_states_accumulated

    def _create_matrix_view(self, full_data, seq_len):
        """
        Create matrix view: rows = word positions, cols = processing stages.
        
        Matrix[i, j] = state of word i when processing word j
        """
        matrix = {}

        for row in range(seq_len):  # Word position
            matrix[row] = {}
            for col in range(seq_len):  # Processing stage
                if col < row:
                    # Word not yet processed
                    matrix[row][col] = {'status': 'not_yet_processed'}
                elif col == row:
                    # Primary processing
                    traj = full_data['primary_trajectories'][row]
                    matrix[row][col] = {
                        'status': 'primary_processing',
                        'trajectory_length': len(traj),
                        'final_surprisal': traj[-1]['surprisal'],
                        'mean_surprisal': np.mean([t['surprisal'] for t in traj]),
                        'converged': traj[-1]['is_correct'],
                    }
                else:
                    # Used as context
                    key = (row, col)
                    if key in full_data['context_states']:
                        state = full_data['context_states'][key]
                        matrix[row][col] = {
                            'status': 'used_as_context',
                            'noise_level': state['noise_level'],
                            'empirical_certainty': state['empirical_certainty'],
                            'theoretical_certainty': 1 - state['noise_level'],
                        }
                    else:
                        matrix[row][col] = {'status': 'no_data'}

        return matrix

    def save_full_data(self, full_data, output_path):
        """Save full tracking data."""
        with open(output_path, 'wb') as f:
            pickle.dump(full_data, f)
        print(f"\n✓ Saved full tracking data: {output_path}")

    def create_summary_csv(self, full_data, output_csv):
        """Create summary CSV with key metrics."""
        rows = []

        seq_len = len(full_data['tokens'])
        
        for pos in range(seq_len):
            word = full_data['tokens'][pos]
            traj = full_data['primary_trajectories'][pos]

            # Primary metrics
            surprisals = [t['surprisal'] for t in traj]
            entropies = [t['entropy'] for t in traj]

            row = {
                'word_position': pos,
                'word': word,
                'surprisal_mean': np.mean(surprisals),
                'surprisal_final': surprisals[-1],
                'entropy_mean': np.mean(entropies),
                'final_is_correct': traj[-1]['is_correct'],
            }

            # Context metrics: how was this word when used as context?
            context_certainties = []
            for stage in range(pos + 1, seq_len):
                key = (pos, stage)
                if key in full_data['context_states']:
                    state = full_data['context_states'][key]
                    context_certainties.append(state['empirical_certainty'])

            if len(context_certainties) > 0:
                row['context_certainty_mean'] = np.mean(context_certainties)
                row['context_certainty_final'] = context_certainties[-1]
            else:
                row['context_certainty_mean'] = 1.0
                row['context_certainty_final'] = 1.0

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        print(f"✓ Saved summary CSV: {output_csv}")
        return df


def visualize_matrix(full_data, output_path='processing_matrix.png'):
    """Visualize the processing matrix."""
    import matplotlib.pyplot as plt

    matrix_view = full_data['matrix_view']
    seq_len = len(full_data['tokens'])

    # Create certainty matrix for visualization
    certainty_matrix = np.zeros((seq_len, seq_len))

    for row in range(seq_len):
        for col in range(seq_len):
            cell = matrix_view[row][col]
            if cell['status'] == 'not_yet_processed':
                certainty_matrix[row, col] = -1  # Grey
            elif cell['status'] == 'primary_processing':
                certainty_matrix[row, col] = 0.5  # Yellow
            elif cell['status'] == 'used_as_context':
                certainty_matrix[row, col] = cell['empirical_certainty']
            else:
                certainty_matrix[row, col] = 0

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(certainty_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    # Labels
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels(full_data['tokens'], rotation=45, ha='right')
    ax.set_yticklabels(full_data['tokens'])

    ax.set_xlabel('Processing Stage (which word is being processed)', fontsize=12)
    ax.set_ylabel('Word Position (which word we\'re tracking)', fontsize=12)
    ax.set_title('Full Processing Matrix\n(Color = certainty when word is used)', fontsize=14)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Certainty (green=certain, red=uncertain)', fontsize=10)

    # Grid
    ax.set_xticks(np.arange(seq_len) - 0.5, minor=True)
    ax.set_yticks(np.arange(seq_len) - 0.5, minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved matrix visualization: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Full tracking of SEDD trajectories with soft projection'
    )
    parser.add_argument('--sentence', type=str,
                       default="The cat sat on the mat.",
                       help='Test sentence')
    parser.add_argument('--output-prefix', type=str, default='full_tracking',
                       help='Prefix for output files')
    parser.add_argument('--noise-schedule', type=str, default='exponential',
                       choices=['exponential', 'linear', 'none'])
    parser.add_argument('--base-noise', type=float, default=0.3)
    parser.add_argument('--steps', type=int, default=256,
                       help='Diffusion steps (reduced for speed)')

    args = parser.parse_args()

    # Extract
    extractor = FullTrackingExtractor(
        num_steps=args.steps,
        noise_schedule=args.noise_schedule,
        base_noise=args.base_noise
    )

    full_data = extractor.extract_full_trajectories(
        args.sentence, save_every=10
    )

    # Save
    extractor.save_full_data(full_data, f'{args.output_prefix}_data.pkl')
    df = extractor.create_summary_csv(full_data, f'{args.output_prefix}_summary.csv')

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(df.to_string(index=False))

    # Visualize
    visualize_matrix(full_data, f'{args.output_prefix}_matrix.png')

    print("\n" + "="*70)
    print("✓ Full tracking complete!")
    print(f"  Data: {args.output_prefix}_data.pkl")
    print(f"  CSV: {args.output_prefix}_summary.csv")
    print(f"  Plot: {args.output_prefix}_matrix.png")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

