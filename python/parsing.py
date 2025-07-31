"""
Module for parsing benchmark results from Criterion JSON outputs and CSV files.
"""
import os
import re
import json
import pandas as pd

CRITERION_DIR = "target/criterion"
BENCH_SIZE_CSV = "bench_results/size_results.csv"

def parse_size_results():
    """Parses the memory size benchmark results from its dedicated CSV file."""
    if not os.path.exists(BENCH_SIZE_CSV):
        print(f"Error: Size benchmark CSV not found at {BENCH_SIZE_CSV}")
        return pd.DataFrame()
    return pd.read_csv(BENCH_SIZE_CSV).drop_duplicates()

def parse_random_access_results():
    """
    Parses the criterion JSON files for the random_access benchmark.
    """
    results = []
    base_path = CRITERION_DIR
    if not os.path.isdir(base_path):
        print(f"Error: Criterion output directory not found at {base_path}")
        return pd.DataFrame(results)

    try:
        all_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    except FileNotFoundError:
        return pd.DataFrame()

    ra_dirs = [d for d in all_dirs if d.startswith("RandomAccess_")]

    for ra_dir in ra_dirs:
        try:
            distribution = ra_dir.split('_', 1)[1]
        except IndexError:
            continue

        dist_path = os.path.join(base_path, ra_dir)
        for root, _, files in os.walk(dist_path):
            if "estimates.json" in files and root.endswith(os.path.join('base')):
                benchmark_id = os.path.basename(os.path.dirname(root))
                name, k = None, 0

                if benchmark_id == "Baseline_get":
                    name, k = "Baseline", 0
                elif benchmark_id == "FixedLength_get":
                    name, k = "FixedLength", 0
                else:
                    match = re.match(r"(.+)_k=(\d+)_get", benchmark_id)
                    if match:
                        name = match.group(1)
                        k = int(match.group(2))

                if name is None:
                    continue

                with open(os.path.join(root, 'estimates.json'), 'r') as f:
                    data = json.load(f)
                    elapsed_seconds = data['mean']['point_estimate'] / 1e9
                    results.append({
                        "name": name,
                        "k": k,
                        "distribution": distribution,
                        "access_elapsed_seconds": elapsed_seconds
                    })

    return pd.DataFrame(results)

def parse_parallel_results():
    """
    Parses the criterion JSON files for the parallel (access methods) benchmark.
    """
    results = []
    base_path = CRITERION_DIR
    if not os.path.isdir(base_path):
        print(f"Error: Criterion output directory not found at {base_path}")
        return pd.DataFrame()

    try:
        all_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    except FileNotFoundError:
        return pd.DataFrame()

    dist_dirs = [d for d in all_dirs if not d.startswith("RandomAccess") and d != "report"]
    possible_methods = ["par_get_many", "get_many", "get_loop"]

    for dist in dist_dirs:
        dist_path = os.path.join(base_path, dist)
        for root, _, files in os.walk(dist_path):
            if "estimates.json" in files and root.endswith(os.path.join('base')):
                benchmark_id = os.path.basename(os.path.dirname(root))
                method_found, codec_name = None, None

                for method in possible_methods:
                    suffix = f"_{method}"
                    if benchmark_id.endswith(suffix):
                        method_found = method
                        codec_name = benchmark_id[:-len(suffix)]
                        break

                if not method_found:
                    continue

                with open(os.path.join(root, 'estimates.json'), 'r') as f:
                    data = json.load(f)
                    elapsed_seconds = data['mean']['point_estimate'] / 1e9
                    results.append({
                        "distribution": dist,
                        "codec": codec_name,
                        "method": method_found,
                        "elapsed_seconds": elapsed_seconds
                    })
    return pd.DataFrame(results)

def parse_unchecked_results():
    """Parses criterion JSON for the unchecked_access benchmark."""
    results = []
    base_path = CRITERION_DIR
    if not os.path.isdir(base_path):
        print(f"Error: Criterion output directory not found at {base_path}")
        return pd.DataFrame()

    group_dirs = [d for d in os.listdir(base_path) if d.startswith("RandomAccess")]

    for group_dir in group_dirs:
        bit_width_match = re.search(r"(\d+)bit", group_dir)
        if not bit_width_match:
            continue
        bit_width = int(bit_width_match.group(1))

        group_path = os.path.join(base_path, group_dir)
        for bench_name in os.listdir(group_path):
            bench_path = os.path.join(group_path, bench_name, 'base', 'estimates.json')
            if os.path.exists(bench_path):
                parts = bench_name.split('/')

                if len(parts) == 2: # e.g., "LEFixedVec/Checked"
                    impl = parts[0].replace("sux__BitFieldVec", "sux::BitFieldVec")
                    access = parts[1]
                elif len(parts) == 1: # e.g., "Checked" for the baseline
                    impl = "Baseline_Vec<u64>"
                    access = parts[0]
                else:
                    continue

                with open(bench_path, 'r') as f:
                    data = json.load(f)
                    elapsed_seconds = data['mean']['point_estimate'] / 1e9
                    results.append({
                        "implementation": impl,
                        "access_type": access,
                        "bit_width": bit_width,
                        "time_s": elapsed_seconds,
                    })
    return pd.DataFrame(results)
