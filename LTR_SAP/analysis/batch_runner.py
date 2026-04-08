"""
Batch runner: iterate over all SAP stimuli CSVs and run enforce-prefix LTR
on each sentence, saving results per condition to LTR_SAP/.

Usage (on cloud cluster):
  python LTR_SAP/analysis/batch_runner.py --model_path louaaron/sedd-medium --steps 1024

The script loads the SEDD model once, then processes all sentences sequentially.
Results are saved as JSON files: LTR_SAP/{subset}/{condition}/item_{item}.json
"""

import argparse
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch
from utils import get_sap_files, iter_sap_items, get_subset_name, get_output_path
from sedd_helpers import load_sedd_model
from transformer_strict_ltr import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Batch enforce-prefix LTR over SAP stimuli")
    parser.add_argument("--model_path", type=str, default="louaaron/sedd-medium")
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--subset", type=str, default=None,
        help="Process only this subset (e.g., 'Agreement'). Default: all subsets.",
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip items that already have output files",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading SEDD model (this may take a minute)...")
    model, graph, noise, tokenizer = load_sedd_model(args.model_path, device)
    model_bundle = (model, graph, noise, tokenizer)
    print("Model loaded.\n")

    sap_files = get_sap_files()
    total_items = 0
    processed = 0
    skipped = 0
    errors = []

    for csv_path in sap_files:
        subset = get_subset_name(csv_path)
        if args.subset and subset != args.subset:
            continue

        print(f"\n{'='*70}")
        print(f"Processing subset: {subset} ({csv_path.name})")
        print(f"{'='*70}\n")

        for item_info in iter_sap_items(csv_path):
            total_items += 1
            item_id = item_info["item"]
            condition = item_info["condition"]
            sentence = item_info["sentence"]

            output_path = get_output_path(subset, condition, item_id)

            if args.skip_existing and output_path.exists():
                skipped += 1
                continue

            print(f"  [{subset}] item={item_id}, cond={condition}")
            print(f"    sentence: {sentence[:80]}{'...' if len(sentence)>80 else ''}")

            try:
                t0 = time.time()
                run_experiment(
                    model_path=args.model_path,
                    prefix=None,
                    target=None,
                    sentence=sentence,
                    steps=args.steps,
                    device=device,
                    ltr=True,
                    causal=False,
                    renoise=False,
                    renoise_sigma=1.0,
                    enforce_prefix=True,
                    batch_size=1,
                    seed=args.seed,
                    output_path=str(output_path),
                    model_bundle=model_bundle,
                )
                elapsed = time.time() - t0
                processed += 1
                print(f"    -> saved to {output_path} ({elapsed:.1f}s)\n")
            except Exception as e:
                errors.append((subset, item_id, condition, str(e)))
                print(f"    -> ERROR: {e}\n")

    print(f"\n{'='*70}")
    print(f"Batch complete: {processed} processed, {skipped} skipped, {len(errors)} errors, {total_items} total")
    if errors:
        print(f"\nErrors:")
        for subset, item_id, cond, msg in errors:
            print(f"  [{subset}] item={item_id} cond={cond}: {msg}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
