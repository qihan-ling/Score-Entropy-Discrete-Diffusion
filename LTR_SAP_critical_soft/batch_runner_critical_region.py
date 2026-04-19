"""
Batch runner for the sequential-rescheduling soft-context experiment.

For experimental subsets (Agreement, ClassicGP, RelativeClause, AttachmentAmbiguity):
  Run one 6-position sequential rescheduling pass per item, centered on the
  critical/target position.

For filler subset:
  Sliding-window approach: for each filler item, run a 6-token window starting
  at every valid center position from word 3 (0-indexed) onward.

Results saved to:
  LTR_SAP_critical_soft/results/lambda_{val}/{subset}/{condition}/item_{id}.json
  LTR_SAP_critical_soft/results/lambda_{val}/filler/item_{id}_window_{center}.json

Usage:
  python LTR_SAP_critical_soft/batch_runner_critical_region.py \
      --model_path louaaron/sedd-medium --steps 1024 --lambda_val 1.0

  # Single subset:
  python LTR_SAP_critical_soft/batch_runner_critical_region.py \
      --model_path louaaron/sedd-medium --lambda_val 0.5 --subset Agreement
"""

import argparse
import sys
import os
import time
from pathlib import Path

_this_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.join(_this_dir, "..")
_analysis_dir = os.path.join(_repo_root, "LTR_SAP", "analysis")

sys.path.insert(0, _analysis_dir)
from utils import get_sap_files, get_subset_name, get_critical_pos_col, load_sap_csv

sys.path.insert(0, _this_dir)
from transformer_critical_region import run_critical_region

sys.path.append(_repo_root)
import torch
from sedd_helpers import load_sedd_model

RESULT_BASE = Path(__file__).resolve().parent / "results"


def get_experimental_output_path(lambda_val, subset, condition, item_id):
    lam_str = f"lambda_{lambda_val:.2f}"
    if condition:
        d = RESULT_BASE / lam_str / subset / condition
    else:
        d = RESULT_BASE / lam_str / subset
    d.mkdir(parents=True, exist_ok=True)
    return d / f"item_{item_id}.json"


def get_filler_output_path(lambda_val, item_id, center_word_pos):
    lam_str = f"lambda_{lambda_val:.2f}"
    d = RESULT_BASE / lam_str / "filler"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"item_{item_id}_window_{center_word_pos}.json"


def main():
    parser = argparse.ArgumentParser(
        description="Batch runner: sequential rescheduling with soft context"
    )
    parser.add_argument("--model_path", type=str, default="louaaron/sedd-medium")
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--lambda_val", type=float, default=1.0,
                        help="Soft-context lambda (0.0=pure model, 1.0=hard gt)")
    parser.add_argument("--pad_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--subset", type=str, default=None, help="Process only this subset")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--track_token_groups", action="store_true")
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
        if is_filler:
            print(f"Subset: {subset} (filler - sliding window, lambda={args.lambda_val})")
        else:
            print(f"Subset: {subset} (critical region, lambda={args.lambda_val})")
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
                # Sliding window: center from word 3 (1-indexed) to word N-3 (1-indexed)
                # so that the window [center-2, center+3] fits within [word 1, word N]
                earliest_center = 3   # 1-indexed; crit-2 = word 1
                latest_center = n_words - 3  # 1-indexed; crit+3 = word N

                if earliest_center > latest_center:
                    print(f"  [{subset}] item={item_id}: sentence too short ({n_words} words), skipping")
                    continue

                for center in range(earliest_center, latest_center + 1):
                    total_runs += 1
                    out_path = get_filler_output_path(args.lambda_val, item_id, center)

                    if args.skip_existing and out_path.exists():
                        skipped += 1
                        continue

                    print(f"  [{subset}] item={item_id} window_center={center}/{n_words}")
                    try:
                        t0 = time.time()
                        run_critical_region(
                            sentence=sentence,
                            crit_word_pos=center,
                            lambda_val=args.lambda_val,
                            steps=args.steps,
                            pad_length=args.pad_length,
                            device=device,
                            seed=args.seed,
                            output_path=str(out_path),
                            model_bundle=model_bundle,
                            track_token_groups=args.track_token_groups,
                        )
                        elapsed = time.time() - t0
                        processed += 1
                        print(f"    -> saved ({elapsed:.1f}s)\n")
                    except Exception as e:
                        errors.append((subset, item_id, f"window={center}", str(e)))
                        print(f"    -> ERROR: {e}\n")
            else:
                # Experimental subset: one run per item, centered on critical position
                crit_pos = int(row[crit_col])
                total_runs += 1
                out_path = get_experimental_output_path(
                    args.lambda_val, subset, condition, item_id
                )

                if args.skip_existing and out_path.exists():
                    skipped += 1
                    continue

                print(f"  [{subset}] item={item_id} cond={condition} crit={crit_pos}")
                try:
                    t0 = time.time()
                    run_critical_region(
                        sentence=sentence,
                        crit_word_pos=crit_pos,
                        lambda_val=args.lambda_val,
                        steps=args.steps,
                        pad_length=args.pad_length,
                        device=device,
                        seed=args.seed,
                        output_path=str(out_path),
                        model_bundle=model_bundle,
                        track_token_groups=args.track_token_groups,
                    )
                    elapsed = time.time() - t0
                    processed += 1
                    print(f"    -> saved ({elapsed:.1f}s)\n")
                except Exception as e:
                    errors.append((subset, item_id, f"cond={condition}", str(e)))
                    print(f"    -> ERROR: {e}\n")

    print(f"\n{'='*70}")
    print(f"Batch complete (lambda={args.lambda_val}): "
          f"{processed} processed, {skipped} skipped, "
          f"{len(errors)} errors, {total_runs} total")
    if errors:
        print(f"\nErrors:")
        for info in errors:
            print(f"  {info}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
