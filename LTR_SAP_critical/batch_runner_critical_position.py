"""
Batch runner for critical-position experiment.

For experimental subsets (Agreement, ClassicGP, RelativeClause, AttachmentAmbiguity):
  Run 6 passes per item targeting [crit-2, crit-1, crit, crit+1, crit+2, crit+3].

For filler subset:
  Run one pass per word position (all positions).

Results saved to LTR_SAP_critical/{subset}/{condition}/item_{id}_pos_{offset}.json

Usage (on cloud cluster):
  python LTR_SAP_critical/batch_runner_critical_position.py \
      --model_path louaaron/sedd-medium --steps 1024

  # Single subset:
  python LTR_SAP_critical/batch_runner_critical_position.py \
      --model_path louaaron/sedd-medium --subset Agreement
"""

import argparse
import sys
import os
import time
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformer_critical_position import run_critical_position

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "LTR_SAP", "analysis"))
from utils import get_sap_files, get_subset_name, get_critical_pos_col, load_sap_csv

from sedd_helpers import load_sedd_model

RESULT_DIR = Path(__file__).resolve().parent

EXPERIMENTAL_OFFSETS = [-2, -1, 0, 1, 2, 3]


def get_critical_output_path(subset, condition, item_id, offset):
    """Get output path for a critical-position run."""
    if condition:
        d = RESULT_DIR / subset / condition
    else:
        d = RESULT_DIR / subset
    d.mkdir(parents=True, exist_ok=True)
    return d / f"item_{item_id}_pos_{offset:+d}.json"


def get_filler_output_path(subset, item_id, word_pos):
    """Get output path for a filler per-position run."""
    d = RESULT_DIR / subset
    d.mkdir(parents=True, exist_ok=True)
    return d / f"item_{item_id}_wpos_{word_pos}.json"


def main():
    parser = argparse.ArgumentParser(description="Batch critical-position experiment")
    parser.add_argument("--model_path", type=str, default="louaaron/sedd-medium")
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--subset", type=str, default=None, help="Process only this subset")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--future_window", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading SEDD model...")
    model, graph, noise, tokenizer = load_sedd_model(args.model_path, device)
    model_bundle = (model, graph, noise, tokenizer)
    print("Model loaded.\n")

    sap_files = get_sap_files()
    total_runs = 0
    processed = 0
    skipped = 0
    errors = []

    for csv_path in sap_files:
        subset = get_subset_name(csv_path)
        if args.subset and subset != args.subset:
            continue

        crit_col = get_critical_pos_col(csv_path)
        is_filler = (crit_col is None)

        print(f"\n{'='*70}")
        print(f"Subset: {subset} ({'filler - all positions' if is_filler else f'critical window [{EXPERIMENTAL_OFFSETS[0]:+d}..{EXPERIMENTAL_OFFSETS[-1]:+d}]'})")
        print(f"{'='*70}\n")

        df = load_sap_csv(csv_path)

        for _, row in df.iterrows():
            item_id = row.get("item", row.get("item#_in_Provo", None))
            sentence = row["Sentence"]
            words = sentence.split()
            n_words = len(words)

            cond_col = "condition" if "condition" in row.index else None
            condition = row[cond_col] if cond_col else None

            if is_filler:
                # Filler: one pass per word position (1-indexed)
                for wpos in range(1, n_words + 1):
                    total_runs += 1
                    out_path = get_filler_output_path(subset, item_id, wpos)

                    if args.skip_existing and out_path.exists():
                        skipped += 1
                        continue

                    print(f"  [{subset}] item={item_id} wpos={wpos}/{n_words}")
                    try:
                        t0 = time.time()
                        run_critical_position(
                            sentence=sentence,
                            target_word_position=wpos,
                            steps=args.steps,
                            device=device,
                            seed=args.seed,
                            output_path=str(out_path),
                            model_bundle=model_bundle,
                            future_window=args.future_window,
                        )
                        elapsed = time.time() - t0
                        processed += 1
                        print(f"    -> saved ({elapsed:.1f}s)\n")
                    except Exception as e:
                        errors.append((subset, item_id, wpos, str(e)))
                        print(f"    -> ERROR: {e}\n")
            else:
                # Experimental: 6 passes around critical position
                crit_pos = int(row[crit_col])

                for offset in EXPERIMENTAL_OFFSETS:
                    target_wpos = crit_pos + offset
                    if target_wpos < 1 or target_wpos > n_words:
                        continue

                    total_runs += 1
                    out_path = get_critical_output_path(subset, condition, item_id, offset)

                    if args.skip_existing and out_path.exists():
                        skipped += 1
                        continue

                    print(f"  [{subset}] item={item_id} cond={condition} "
                          f"crit={crit_pos} offset={offset:+d} -> wpos={target_wpos}")
                    try:
                        t0 = time.time()
                        run_critical_position(
                            sentence=sentence,
                            target_word_position=target_wpos,
                            steps=args.steps,
                            device=device,
                            seed=args.seed,
                            output_path=str(out_path),
                            model_bundle=model_bundle,
                            future_window=args.future_window,
                        )
                        elapsed = time.time() - t0
                        processed += 1
                        print(f"    -> saved ({elapsed:.1f}s)\n")
                    except Exception as e:
                        errors.append((subset, item_id, f"offset={offset}", str(e)))
                        print(f"    -> ERROR: {e}\n")

    print(f"\n{'='*70}")
    print(f"Batch complete: {processed} processed, {skipped} skipped, "
          f"{len(errors)} errors, {total_runs} total")
    if errors:
        print(f"\nErrors:")
        for info in errors:
            print(f"  {info}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
