import os
import json
import re
import pandas as pd
from utils import CRITERION_DIR

def parse_random_access_results():
    """
    Parses the Criterion JSON output files for the random_access benchmark.
    It navigates the directory structure created by Criterion to extract
    benchmark results for different implementations, codecs, and parameters.
    """
    results = []
    base_path = CRITERION_DIR
    if not os.path.isdir(base_path):
        print(f"Error: Criterion output directory not found at '{base_path}'")
        return pd.DataFrame()

    try:
        # Find directories matching the pattern "RandomAccess_DistributionName".
        group_dirs = [d for d in os.listdir(base_path) if d.startswith("RandomAccess_") and os.path.isdir(os.path.join(base_path, d))]
    except FileNotFoundError:
        return pd.DataFrame()

    for group_dir in group_dirs:
        try:
            # Extract the distribution name from the directory name.
            distribution = group_dir.split('_', 1)[1]
        except IndexError:
            continue

        dist_path = os.path.join(base_path, group_dir)
        for bench_dir in os.listdir(dist_path):
            estimates_path = os.path.join(dist_path, bench_dir, 'base', 'estimates.json')
            if not os.path.exists(estimates_path):
                continue
            
            name = "Unknown"
            k = 0  # 0 indicates not applicable (e.g., for fixed-width vectors)

            # Regex to parse variable-length codecs with a 'k' parameter.
            match_k = re.match(r"(.+)_k=(\d+)_get_unchecked", bench_dir)
            if match_k:
                name = match_k.group(1)
                k = int(match_k.group(2))
            # Handle fixed-width vectors and the baseline.
            elif bench_dir.endswith("_get_unchecked"):
                name = bench_dir.replace("_get_unchecked", "")
            elif bench_dir.endswith("_get"):
                name = bench_dir.replace("_get", "")

            # Clean up names where '::' was replaced by '__'.
            name = name.replace('__', '::')

            with open(estimates_path, 'r') as f:
                data = json.load(f)
                # Time is in nanoseconds, convert to seconds.
                elapsed_seconds = data['mean']['point_estimate'] / 1e9
                results.append({
                    "name": name,
                    "k": k,
                    "distribution": distribution,
                    "time_seconds": elapsed_seconds,
                })

    return pd.DataFrame(results)