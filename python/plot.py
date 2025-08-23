"""
Main entry point for generating benchmark plots for compressed-intvec.
"""
import argparse
import os
from plotting import plot_random_access, plot_fixed_access
from utils import OUTPUT_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from compressed-intvec benchmark results.")
    parser.add_argument("--random-access", action="store_true", help="Generate the variable-width random access slowdown plot.")
    parser.add_argument("--fixed-access", action="store_true", help="Generate the fixed-width random access performance plot.")
    parser.add_argument("--all", action="store_true", help="Generate all available plots.")
    args = parser.parse_args()

    # Determine which plots to run.
    run_all = args.all or not any([args.random_access, args.fixed_access])
    run_random_access = args.random_access or run_all
    run_fixed_access = args.fixed_access or run_all

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if run_random_access:
        plot_random_access()
    
    if run_fixed_access:
        plot_fixed_access()