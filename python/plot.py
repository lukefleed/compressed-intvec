import argparse
import os
from plotting import (
    plot_random_access, 
    plot_fixed_access, 
    plot_atomic_scaling
)
from utils import OUTPUT_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from compressed-intvec benchmark results.")
    parser.add_argument("--random-access", action="store_true", help="Generate the variable-width random access slowdown plot.")
    parser.add_argument("--fixed-access", action="store_true", help="Generate the fixed-width random access performance plot.")
    parser.add_argument("--atomic-scaling", action="store_true", help="Generate atomic write scaling plots.")
    parser.add_argument("--all", action="store_true", help="Generate all available plots.")
    args = parser.parse_args()

    run_all = args.all or not any(vars(args).values())
    run_random_access = args.random_access or run_all
    
    run_fixed_access = args.fixed_access or run_all
    run_atomic_scaling = args.atomic_scaling or run_all

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if run_random_access:
        plot_random_access()
    
    if run_fixed_access:
        plot_fixed_access()

    if run_atomic_scaling:
        plot_atomic_scaling(
            group_prefix="LockFreeScaling_Diffuse",
            bit_width=16,
            baseline_name="Baseline Vec<AtomicU16>",
            filename_suffix="lock_free_diffuse",
            title_suffix="16-bit, Lock-Free Path"
        )
        
        plot_atomic_scaling(
            group_prefix="LockedScaling_Diffuse",
            bit_width=15,
            baseline_name="Baseline Vec<AtomicU16>",
            filename_suffix="locked_path_diffuse",
            title_suffix="15-bit, Locked Path"
        )