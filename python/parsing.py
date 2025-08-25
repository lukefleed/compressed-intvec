import os
import json
import re
import pandas as pd
from utils import CRITERION_DIR, OPS_PER_THREAD

def parse_fixed_access_results():
    results = []
    base_path = CRITERION_DIR
    if not os.path.isdir(base_path):
        print(f"Error: Criterion output directory not found at '{base_path}'")
        return pd.DataFrame()

    group_path = os.path.join(base_path, "RandomAccess")
    if not os.path.isdir(group_path):
         print(f"Warning: No benchmark data found in '{group_path}'. Did you run 'cargo bench --bench bench_random_access'?")
         return pd.DataFrame()

    for bit_width_dir in os.listdir(group_path):
        match_bw = re.match(r"(\d+)bit", bit_width_dir)
        if not match_bw:
            continue
        bit_width = int(match_bw.group(1))

        bw_path = os.path.join(group_path, bit_width_dir)
        for bench_name in os.listdir(bw_path):
            estimates_path = os.path.join(bw_path, bench_name, 'base', 'estimates.json')
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
         print(f"Warning: No benchmark data found in '{group_path}'.")

    return pd.DataFrame(results)


def parse_atomic_scaling_results(group_name):
    results = []
    
    # Look for directories that match the group_name pattern with thread counts
    pattern = re.compile(rf"{re.escape(group_name)}_(\d+)Threads")
    
    if not os.path.isdir(CRITERION_DIR):
        print(f"Warning: Criterion directory not found at '{CRITERION_DIR}'.")
        return pd.DataFrame()
    
    found_directories = False
    for dir_name in os.listdir(CRITERION_DIR):
        match = pattern.match(dir_name)
        if not match:
            continue
            
        found_directories = True
        num_threads = int(match.group(1))
        thread_path = os.path.join(CRITERION_DIR, dir_name)
        
        for bench_dir in os.listdir(thread_path):
            if bench_dir == 'report':  # Skip report directory
                continue
                
            estimates_path = os.path.join(thread_path, bench_dir, 'base', 'estimates.json')
            if not os.path.exists(estimates_path):
                continue
            
            implementation_name = bench_dir.split('/')[0]
            
            # Clean up implementation names
            if implementation_name.endswith('_store'):
                implementation_name = implementation_name[:-6]  # Remove '_store' suffix
            
            # Convert naming conventions
            if implementation_name.startswith('Baseline_Vec_'):
                # Convert Baseline_Vec_AtomicU16_ to Baseline Vec<AtomicU16>
                type_part = implementation_name.replace('Baseline_Vec_', '').rstrip('_')
                implementation_name = f"Baseline Vec<{type_part}>"
            elif '__' in implementation_name:
                # Convert sux__AtomicBitFieldVec to sux::AtomicBitFieldVec
                implementation_name = implementation_name.replace('__', '::')

            with open(estimates_path, 'r') as f:
                data = json.load(f)
                time_ns = data['mean']['point_estimate']
                time_s = time_ns / 1e9
                
                total_ops = OPS_PER_THREAD * num_threads
                throughput_mops = (total_ops / 1e6) / time_s

                results.append({
                    "implementation": implementation_name,
                    "num_threads": num_threads,
                    "throughput_mops": throughput_mops,
                })
    
    if not found_directories:
        print(f"Warning: No benchmark directories found for pattern '{group_name}_*Threads'.")
    
    return pd.DataFrame(results)


def parse_random_access_results():
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