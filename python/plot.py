"""
Main entry point for generating benchmark plots for compressed-intvec.
"""
import argparse
import os
from plotting import (
    plot_size,
    plot_random_access,
    plot_parallel,
    plot_unchecked
)

OUTPUT_DIR = "images"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from compressed-intvec benchmark results.")
    parser.add_argument("--size", action="store_true", help="Generate plots for memory size benchmarks.")
    parser.add_argument("--random-access", action="store_true", help="Generate plots for random access (variable codes) benchmarks.")
    parser.add_argument("--parallel", action="store_true", help="Generate plots for parallel performance benchmarks.")
    parser.add_argument("--unchecked", action="store_true", help="Generate plots for fixed-width unchecked access benchmarks.")
    parser.add_argument("--all", action="store_true", help="Generate all benchmark plots.")
    args = parser.parse_args()

    run_all = args.all
    run_any = any([args.size, args.random_access, args.parallel, args.unchecked, run_all])

    if not run_any:
        parser.print_help()
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        if args.size or run_all:
            plot_size()
        if args.random_access or run_all:
            plot_random_access()
        if args.parallel or run_all:
            plot_parallel()
        if args.unchecked or run_all:
            plot_unchecked()
