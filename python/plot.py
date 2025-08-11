"""
Main entry point for generating benchmark plots for compressed-intvec.
"""
import argparse
import os
from plotting import plot_random_access
from utils import OUTPUT_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from compressed-intvec benchmark results.")
    parser.add_argument("--random-access", action="store_true", help="Generate the random access slowdown plot.")
    parser.add_argument("--all", action="store_true", help="Generate all available plots.")
    args = parser.parse_args()

    # Determine which plots to run. If no specific plot is requested, run all.
    run_all = args.all or not any([args.random_access])
    run_random_access = args.random_access or run_all

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if run_random_access:
        plot_random_access()