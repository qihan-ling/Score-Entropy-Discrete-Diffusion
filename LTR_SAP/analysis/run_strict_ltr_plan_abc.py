"""
Run the full Plan A/B/C analysis pipeline on strict-LTR results.

Prerequisite: filler strict-LTR results must be collected first.

Pipeline:
  1. plan_c_alignment.py -> builds merged data files
  2. filler_conversion.R -> fits filler conversion models
  3. plan_a_scatter.py -> steps vs surprisal analysis
  4. plan_a_entropy.py -> entropy trajectory visualizations
  5. plan_b_trajectories.py -> trajectory typology clustering
  6. plan_c_regression.py -> predicted vs empirical effects
  7. plan_c_spillover.py -> spillover analysis

Usage:
  python LTR_SAP/analysis/run_strict_ltr_plan_abc.py

  # Or run individual steps:
  python LTR_SAP/analysis/run_strict_ltr_plan_abc.py --step alignment
  python LTR_SAP/analysis/run_strict_ltr_plan_abc.py --step plan_a
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent
DATA_DIR = ANALYSIS_DIR / "data"
FIG_DIR_A = ANALYSIS_DIR / "figures" / "plan_a"
FIG_DIR_B = ANALYSIS_DIR / "figures" / "plan_b"
FIG_DIR_C = ANALYSIS_DIR / "figures" / "plan_c"


def run_step(name, cmd, check=True):
    """Run a pipeline step, printing its output."""
    print(f"\n{'='*70}")
    print(f"Step: {name}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if check and result.returncode != 0:
        print(f"  WARNING: {name} exited with code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run Plan A/B/C on strict-LTR results")
    parser.add_argument("--step", type=str, default=None,
                        choices=["alignment", "filler_conversion", "plan_a", "plan_b", "plan_c", "all"],
                        help="Run a specific step (default: all)")
    args = parser.parse_args()

    step = args.step or "all"
    python = sys.executable

    steps_to_run = {
        "alignment": True,
        "filler_conversion": True,
        "plan_a": True,
        "plan_b": True,
        "plan_c": True,
    }
    if step != "all":
        steps_to_run = {k: (k == step) for k in steps_to_run}

    # Step 1: Data alignment
    if steps_to_run["alignment"]:
        run_step("Plan C Alignment (data merge)",
                 [python, str(ANALYSIS_DIR / "plan_c_alignment.py"),
                  "--output_dir", str(DATA_DIR)])

    # Step 2: Filler conversion model (R script)
    if steps_to_run["filler_conversion"]:
        r_script = ANALYSIS_DIR / "filler_conversion.R"
        if r_script.exists():
            run_step("Filler conversion model",
                     ["Rscript", str(r_script)], check=False)
        else:
            print("  Skipping filler conversion: R script not found")

    # Step 3: Plan A
    if steps_to_run["plan_a"]:
        run_step("Plan A: Steps vs Surprisal",
                 [python, str(ANALYSIS_DIR / "plan_a_scatter.py"),
                  "--output_dir", str(FIG_DIR_A)])
        run_step("Plan A: Entropy trajectories",
                 [python, str(ANALYSIS_DIR / "plan_a_entropy.py"),
                  "--output_dir", str(FIG_DIR_A)])

    # Step 4: Plan B
    if steps_to_run["plan_b"]:
        run_step("Plan B: Trajectory typology",
                 [python, str(ANALYSIS_DIR / "plan_b_trajectories.py"),
                  "--output_dir", str(FIG_DIR_B)])

    # Step 5: Plan C
    if steps_to_run["plan_c"]:
        run_step("Plan C: Regression effects",
                 [python, str(ANALYSIS_DIR / "plan_c_regression.py"),
                  "--output_dir", str(FIG_DIR_C)])
        run_step("Plan C: Spillover analysis",
                 [python, str(ANALYSIS_DIR / "plan_c_spillover.py"),
                  "--output_dir", str(FIG_DIR_C)])

    print(f"\n{'='*70}")
    print("Plan A/B/C pipeline complete.")
    print(f"  Data: {DATA_DIR}")
    print(f"  Plan A figures: {FIG_DIR_A}")
    print(f"  Plan B figures: {FIG_DIR_B}")
    print(f"  Plan C figures: {FIG_DIR_C}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
