"""
Verify and prepare GPT-2 surprisals from sapbenchmark for comparison.

The sapbenchmark provides pre-computed GPT-2 surprisals at:
    sapbenchmark/Surprisals/data/gpt2/items_{subset}.gpt2.csv.scaled

These were extracted using get_gpt2_full.py with:
    tokenizer = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)
    inputs = tokenizer("<|endoftext|> " + sentence, return_tensors="pt")

Then rescaled using rescale.py to add z-scored columns:
    sum_surprisal_s, logfreq_s, length_s

This script verifies all required files exist and prints summary statistics,
ensuring we can reuse them directly for fair comparison.

Usage:
  python LTR_SAP/analysis/verify_gpt2_surprisals.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from utils import GPT2_SURP_DIR, REPO_ROOT
import pandas as pd

SUBSETS = ["filler", "Agreement", "ClassicGP", "RelativeClause", "AttachmentAmbiguity"]


def main():
    print("Verifying GPT-2 surprisals from sapbenchmark\n")
    print(f"Source directory: {GPT2_SURP_DIR}\n")

    all_ok = True
    for subset in SUBSETS:
        fname = f"items_{subset}.gpt2.csv.scaled"
        path = GPT2_SURP_DIR / fname

        if not path.exists():
            print(f"  [MISSING] {fname}")
            all_ok = False
            continue

        df = pd.read_csv(path)
        required_cols = ["Sentence", "word", "word_pos", "sum_surprisal",
                         "sum_surprisal_s", "logfreq_s", "length_s"]
        missing_cols = [c for c in required_cols if c not in df.columns]

        if missing_cols:
            print(f"  [ERROR]   {fname}: missing columns {missing_cols}")
            all_ok = False
            continue

        n_sentences = df["Sentence"].nunique()
        n_words = len(df)
        mean_surp = df["sum_surprisal"].mean()
        std_surp = df["sum_surprisal"].std()

        print(f"  [OK]      {fname}")
        print(f"            {n_sentences} sentences, {n_words} word-tokens")
        print(f"            surprisal: mean={mean_surp:.3f}, std={std_surp:.3f}")
        print(f"            columns: {list(df.columns)}")
        print()

    # Also check for the ClassicGP post-processed file
    post_path = GPT2_SURP_DIR / "items_ClassicGP.gpt2.post.csv.scaled"
    if post_path.exists():
        print(f"  [OK]      items_ClassicGP.gpt2.post.csv.scaled (post-processed)")
    else:
        print(f"  [NOTE]    items_ClassicGP.gpt2.post.csv.scaled not found (optional)")

    print()
    if all_ok:
        print("All GPT-2 surprisal files verified. Ready for comparison.")
        print("\nThese files can be loaded via:")
        print("  from utils import load_gpt2_surprisals")
        print("  df = load_gpt2_surprisals('Agreement')")
    else:
        print("Some files are missing or malformed. Run the sapbenchmark extraction pipeline:")
        print("  cd sapbenchmark/Surprisals")
        print("  python get_gpt2_full.py --aligned --input data/items_{subset}.csv --output data/gpt2/items_{subset}.gpt2.csv")
        print("  python rescale.py --path data/gpt2/ --freqs analysis/freqs_coca.csv")


if __name__ == "__main__":
    main()
