import os
import json
import re
import pandas as pd
from utils import CRITERION_DIR

def parse_fixed_access_results():
    """
    Parses the Criterion JSON output files for the fixed-width random_access benchmark.
    Handles directory names like 'RandomAccess_4bit', 'RandomAccess_8bit', etc.
    """
    results = []
    base_path = CRITERION_DIR
    if not os.path.isdir(base_path):
        print(f"Error: Criterion output directory not found at '{base_path}'")
        return pd.DataFrame()

    try:
        group_dirs = [d for d in os.listdir(base_path) if d.startswith("RandomAccess_") and os.path.isdir(os.path.join(base_path, d))]
    except FileNotFoundError:
        return pd.DataFrame()

    for group_dir in group_dirs:
        match_bw = re.match(r"RandomAccess_(\d+)bit", group_dir)
        if not match_bw:
            continue
        bit_width = int(match_bw.group(1))

        group_path = os.path.join(base_path, group_dir)
        for bench_name in os.listdir(group_path):
            estimates_path = os.path.join(group_path, bench_name, 'base', 'estimates.json')
            if not os.path.exists(estimates_path):
                continue
            
            with open(estimates_path, 'r') as f:
                data = json.load(f)
                time_ns = data['mean']['point_estimate']
                results.append({
                    "name": bench_name.replace('__', '::'),
                    "bit_width": bit_width,
                    "time_ns": time_ns,
                })
    
    if not results:
         print(f"Warning: No benchmark data found in '{base_path}'. Did you run 'cargo bench --bench bench_random_access'?")

    return pd.DataFrame(results)


def parse_random_access_results():
    """
    Parses the Criterion JSON output files for the variable-width random_access benchmark.
    """
    results = []
    base_path = CRITERION_DIR
    if not os.path.isdir(base_path):
        print(f"Error: Criterion output directory not found at '{base_path}'")
        return pd.DataFrame()

    for root, dirs, files in os.walk(base_path):
        if "estimates.json" in files and os.path.basename(root) == "base" and "RandomAccess/" in root:
            estimates_path = os.path.join(root, "estimates.json")
            
            relative_path = os.path.relpath(os.path.dirname(root), base_path)
            parts = relative_path.replace('\\', '/').split('/')
            
            if len(parts) < 2 or not parts[0].startswith("RandomAccess"):
                continue
            
            distribution = parts[1]
            bench_details = "/".join(parts[2:])
            
            name = "Unknown"
            k = 0

            match_k = re.match(r"(.+)/k=(\d+)/get_unchecked", bench_details)
            if match_k:
                name = match_k.group(1)
                k = int(match_k.group(2))
            elif "/get_unchecked" in bench_details:
                name = bench_details.split('/')[0]
            elif "/get" in bench_details:
                name = bench_details.split('/')[0]

            with open(estimates_path, 'r') as f:
                data = json.load(f)
                elapsed_seconds = data['mean']['point_estimate'] / 1e9
                results.append({
                    "name": name.replace('__', '::'),
                    "k": k,
                    "distribution": distribution,
                    "time_seconds": elapsed_seconds,
                })

    return pd.DataFrame(results)