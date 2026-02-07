"""
Quick test script for enhanced SEDD extraction
Tests on a single sentence to verify all metrics are computed
"""

import torch
from sedd_ltr_proj_enhanced import SEDDTrajectoryExtractor
import pandas as pd

def test_basic_extraction():
    """Test basic extraction on a short sentence."""
    
    print("="*60)
    print("Testing Enhanced SEDD Extraction")
    print("="*60)
    
    # Short test sentence
    test_sentence = "The horse raced past the barn fell."
    
    print(f"\nTest sentence: '{test_sentence}'")
    
    # Initialize extractor
    print("\n1. Initializing extractor...")
    try:
        extractor = SEDDTrajectoryExtractor(
            model_name="louaaron/sedd-medium",
            device='cuda' if torch.cuda.is_available() else 'cpu',
            num_steps=128,  # Reduced for testing
            extract_hidden_states=True
        )
        print("   ✓ Extractor initialized")
    except Exception as e:
        print(f"   ✗ Failed to initialize: {e}")
        return False
    
    # Extract trajectories
    print("\n2. Extracting trajectories...")
    try:
        word_metrics = extractor.extract_trajectories(
            test_sentence,
            save_every=10,
            chunk_size=1
        )
        print(f"   ✓ Extracted metrics for {len(word_metrics)} words")
    except Exception as e:
        print(f"   ✗ Failed to extract: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify metrics
    print("\n3. Verifying metrics...")
    if len(word_metrics) == 0:
        print("   ✗ No metrics extracted")
        return False
    
    first_word = word_metrics[0]
    print(f"   First word: '{first_word['word']}'")
    
    # Check original metrics
    original_metrics = [
        'convergence_timestep', 'token_changes', 'correctness_ratio'
    ]
    for metric in original_metrics:
        if metric in first_word:
            print(f"   ✓ {metric}: {first_word[metric]:.3f}")
        else:
            print(f"   ✗ Missing {metric}")
            return False
    
    # Check score metrics
    score_metrics = [
        'surprisal_mean', 'entropy_mean', 'target_prob_final'
    ]
    for metric in score_metrics:
        if metric in first_word:
            print(f"   ✓ {metric}: {first_word[metric]:.3f}")
        else:
            print(f"   ✗ Missing {metric}")
            return False
    
    # Check hidden state metrics
    if 'hidden_norm_mean' in first_word:
        print(f"   ✓ hidden_norm_mean: {first_word['hidden_norm_mean']:.3f}")
    else:
        print("   ⚠ Hidden state metrics not extracted (may be disabled)")
    
    # Create summary dataframe
    print("\n4. Creating summary...")
    df = pd.DataFrame(word_metrics)
    
    print("\nExtracted metrics summary:")
    print(df[['word', 'surprisal_mean', 'entropy_mean', 'token_changes']].to_string(index=False))
    
    # Check for variance
    print("\n5. Checking for metric variance...")
    if df['token_changes'].std() > 0:
        print(f"   ✓ token_changes has variance (σ={df['token_changes'].std():.3f})")
    else:
        print(f"   ⚠ token_changes has no variance (all same)")
    
    if df['surprisal_mean'].std() > 0:
        print(f"   ✓ surprisal_mean has variance (σ={df['surprisal_mean'].std():.3f})")
    else:
        print(f"   ⚠ surprisal_mean has no variance (all same)")
    
    print("\n" + "="*60)
    print("✓ Test completed successfully!")
    print("="*60)
    
    return True


def test_comparison():
    """Compare chunk_size=1 vs chunk_size=5 on same sentence."""
    
    print("\n" + "="*60)
    print("Testing chunk_size comparison")
    print("="*60)
    
    test_sentence = "The quick brown fox jumps."
    
    results = {}
    
    for chunk_size in [5, 1]:
        print(f"\nTesting chunk_size={chunk_size}...")
        
        extractor = SEDDTrajectoryExtractor(
            model_name="louaaron/sedd-medium",
            device='cuda' if torch.cuda.is_available() else 'cpu',
            num_steps=128,
            extract_hidden_states=False  # Disable for speed
        )
        
        word_metrics = extractor.extract_trajectories(
            test_sentence,
            save_every=10,
            chunk_size=chunk_size
        )
        
        df = pd.DataFrame(word_metrics)
        results[chunk_size] = df
        
        print(f"  Mean token_changes: {df['token_changes'].mean():.3f}")
        print(f"  Mean surprisal: {df['surprisal_mean'].mean():.3f}")
    
    print("\n" + "="*60)
    print("Comparison:")
    print(f"  chunk_size=5: {results[5]['token_changes'].mean():.3f} token changes")
    print(f"  chunk_size=1: {results[1]['token_changes'].mean():.3f} token changes")
    print("  Expected: chunk_size=1 should have MORE variance")
    print("="*60)


if __name__ == '__main__':
    import sys
    
    # Basic test
    success = test_basic_extraction()
    
    if not success:
        print("\n✗ Basic test failed!")
        sys.exit(1)
    
    # Comparison test (optional, slower)
    print("\n\nRun comparison test? (chunk_size=1 vs 5)")
    response = input("This will take ~5-10 minutes [y/N]: ")
    
    if response.lower() == 'y':
        test_comparison()
    else:
        print("Skipping comparison test.")
    
    print("\n✓ All tests passed!")

