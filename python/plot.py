import argparse
import os
import re
import json
import pandas as pd
import plotly.graph_objects as go

# --- Configuration ---
CRITERION_DIR = "target/criterion"
OUTPUT_DIR = "images"
BENCH_SIZE_CSV = "bench_results/size_results.csv"
VECTOR_SIZE = 1_000_000

# --- Data Parsing from Criterion JSON ---

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
        return pd.DataFrame(results)

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

# --- Helper Functions ---

def format_codec_name(name):
    """Converts a raw codec name to a human-readable format."""
    if name == "Baseline": return "Uncompressed Vec<u64>"
    if name == "FixedLength" or (isinstance(name, str) and name.startswith("Fixednum")): return "Fixed Length"
    if isinstance(name, str) and name.startswith("Explicit_"):
        name = name.replace("Explicit_", "")
        if name == "VByteLe": return "VByte (LE)"
        if name == "VByteBe": return "VByte (BE)"
    return name

def format_distribution_subtitle(dist_string):
    """Converts a raw distribution string into a detailed subtitle."""
    dist_map = {
        "UniformLow": f"Uniform (0 to 1,000), {VECTOR_SIZE:,} elements",
        "UniformHigh": f"Uniform (0 to 2<sup>32</sup>), {VECTOR_SIZE:,} elements",
        "Geometric": f"Geometric, {VECTOR_SIZE:,} elements",
        "PowerLaw": f"Power-Law, {VECTOR_SIZE:,} elements",
    }
    # Handle both clean and raw distribution names from CSV
    clean_dist = dist_string.split('_')[0]
    return dist_map.get(clean_dist, f"{clean_dist}, {VECTOR_SIZE:,} elements")

def save_plots(fig, base_name):
    """Saves interactive and static plots."""
    benchmark_output_dir = os.path.join(OUTPUT_DIR, base_name)
    single_distr_dir = os.path.join(benchmark_output_dir, "single_distr")
    os.makedirs(single_distr_dir, exist_ok=True)

    interactive_path = os.path.join(benchmark_output_dir, f"{base_name}_interactive.html")
    print(f"  Saving interactive plot to {interactive_path}")
    fig.write_html(interactive_path, include_plotlyjs='cdn')

    if not (fig.layout.updatemenus and fig.layout.updatemenus[0].buttons):
        return

    distributions = [button['label'] for button in fig.layout.updatemenus[0].buttons]
    main_title = fig.layout.title.text.split('<br>')[0]

    for dist in distributions:
        static_fig = go.Figure(fig)

        # Manually set visibility for traces and shapes
        for trace in static_fig.data:
            trace.visible = hasattr(trace, 'customdata') and dist in trace.customdata
        if static_fig.layout.shapes:
             for shape in static_fig.layout.shapes:
                shape.visible = hasattr(shape, 'name') and dist in shape.name

        static_fig.layout.updatemenus = None
        static_fig.layout.annotations = [ann for ann in static_fig.layout.annotations if "Select Data" not in ann.text]
        subtitle = format_distribution_subtitle(dist)
        static_fig.update_layout(title_text=f"{main_title}<br>{subtitle}")

        base_filename = os.path.join(single_distr_dir, f"{base_name}_{dist}")
        html_path = f"{base_filename}.html"
        print(f"  Saving static plot to {html_path}")
        static_fig.write_html(html_path, include_plotlyjs='cdn')
        try:
            svg_path = f"{base_filename}.svg"
            print(f"  Saving static plot to {svg_path}")
            static_fig.write_image(svg_path, width=1400, height=787)
        except ValueError as e:
            print(f"    Could not save SVG for {dist}: {e}. Ensure 'kaleido' is installed.")

    print(f"Finished processing for {base_name}.")

# --- Plotting Logic ---

def plot_size():
    """Generates plots for memory size benchmark results using dedicated logic."""
    print("Processing size benchmarks...")
    if not os.path.exists(BENCH_SIZE_CSV):
        print(f"Error: File not found at {BENCH_SIZE_CSV}"); return

    df = pd.read_csv(BENCH_SIZE_CSV).drop_duplicates()
    df["space_kb"] = df["space_bytes"] / 1024
    df["clean_distribution"] = df["distribution"].apply(lambda x: x.split('_')[0])

    distributions = sorted(df["clean_distribution"].unique())
    fig = go.Figure()

    default_dist_name = "Geometric" if "Geometric" in distributions else (distributions[0] if distributions else None)
    active_index = distributions.index(default_dist_name) if default_dist_name else 0

    for dist in distributions:
        df_dist = df[df["clean_distribution"] == dist]
        is_visible = (dist == default_dist_name)

        baselines_df = df_dist[df_dist['name'].str.startswith("Vec") | df_dist['name'].str.startswith("Fixed")]
        sampled_df = df_dist.drop(baselines_df.index)

        for codec_name in sorted(sampled_df["name"].unique()):
            df_plot = sampled_df[sampled_df["name"] == codec_name].sort_values("k")
            fig.add_trace(go.Scatter(
                x=df_plot["k"], y=df_plot["space_kb"], mode="lines+markers",
                name=codec_name, customdata=[dist], visible=is_visible,
                hovertemplate="<b>" + codec_name + "</b><br>k=%{x}<br>Space=%{y:.1f} KB<extra></extra>",
            ))

        for _, row in baselines_df.iterrows():
            codec_name = row['name']
            y_val = row['space_kb']
            if codec_name.startswith("Vec"):
                style = {"dash": "dash", "color": "black", "name": "Uncompressed Vec<u64>"}
            else:
                style = {"dash": "dot", "color": "red", "name": codec_name}

            fig.add_hline(y=y_val, line_dash=style["dash"], line_color=style["color"],
                          name=f"{style['name']}_{dist}", visible=is_visible)
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=style["name"],
                                     line=dict(color=style["color"], dash=style["dash"]),
                                     visible=is_visible, customdata=[dist]))

    main_title = "Memory Space vs. Sampling Rate (k)"
    default_subtitle = format_distribution_subtitle(default_dist_name) if default_dist_name else ""

    fig.update_layout(
        width=1400, height=787,
        title_text=f"{main_title}<br>{default_subtitle}",
        xaxis=dict(title="Sampling Rate (k)"),
        yaxis=dict(title="Total Space Usage (KB)"),
        legend_title_text="Codec Type", hovermode="x unified",
        margin=dict(r=250),
        annotations=[dict(text="Select Data Distribution:", showarrow=False, x=1, y=1.15,
                          xref="paper", yref="paper", xanchor='right', yanchor='bottom')]
    )

    buttons = []
    for dist in distributions:
        visibility_arg = [dist in trace.customdata for trace in fig.data]
        shapes_arg = [dict(visible=(dist in shape.name)) for shape in fig.layout.shapes]
        new_subtitle = format_distribution_subtitle(dist)
        buttons.append(dict(label=dist, method="update", args=[
            {"visible": visibility_arg},
            {"shapes": shapes_arg, "title.text": f"{main_title}<br>{new_subtitle}"}
        ]))

    fig.update_layout(updatemenus=[dict(
        buttons=buttons, direction="down", showactive=True, active=active_index,
        x=1, xanchor="right", y=1.1, yanchor="top"
    )])

    save_plots(fig, "size")

def plot_random_access():
    """Generates plots for random access benchmark results using a slowdown factor."""
    print("Processing random access benchmarks...")
    df = parse_random_access_results()
    if df.empty:
        print("No random access benchmark data found to plot."); return

    df["access_elapsed_ms"] = df["access_elapsed_seconds"] * 1000
    df["codec_display_name"] = df["name"].apply(format_codec_name)

    distributions = sorted(df["distribution"].unique())

    # Calculate slowdown factor for each distribution
    for dist in distributions:
        baseline_time_series = df[(df["distribution"] == dist) & (df["codec_display_name"] == "Uncompressed Vec<u64>")]
        if not baseline_time_series.empty:
            baseline_time = baseline_time_series["access_elapsed_seconds"].iloc[0]
            df.loc[df["distribution"] == dist, "slowdown"] = df["access_elapsed_seconds"] / baseline_time
        else:
            df.loc[df["distribution"] == dist, "slowdown"] = float('nan')

    fig = go.Figure()

    default_dist_name = "Geometric" if "Geometric" in distributions else (distributions[0] if distributions else None)
    active_index = distributions.index(default_dist_name) if default_dist_name else 0

    for dist in distributions:
        df_dist = df[df["distribution"] == dist].copy()
        is_visible = (dist == default_dist_name)

        # Separate baselines from other codecs
        baselines_df = df_dist[df_dist["codec_display_name"].isin(["Uncompressed Vec<u64>", "Fixed Length"])]
        sampled_df = df_dist[~df_dist["codec_display_name"].isin(["Uncompressed Vec<u64>", "Fixed Length"])]

        # Plot sampled data (curves)
        for codec_name in sorted(sampled_df["codec_display_name"].unique()):
            df_plot = sampled_df[sampled_df["codec_display_name"] == codec_name].sort_values("k")
            fig.add_trace(go.Scatter(
                x=df_plot["k"], y=df_plot["slowdown"], mode="lines+markers",
                name=codec_name, customdata=[dist], visible=is_visible,
                text=[f"{t:.2f} ms" for t in df_plot["access_elapsed_ms"]],
                hovertemplate=("<b>" + codec_name + "</b><br>" +
                               "k=%{x}<br>" +
                               "Slowdown: %{y:.1f}x<br>" +
                               "Absolute Time: %{text}<extra></extra>")
            ))

        # Plot baselines (horizontal lines)
        # 1. Uncompressed Vec is always at y=1
        fig.add_hline(y=1, line_dash="dash", line_color="black", name=f"Uncompressed Vec<u64>_{dist}", visible=is_visible)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Uncompressed Vec<u64>",
                                 line=dict(color="black", dash="dash"), visible=is_visible, customdata=[dist]))

        # 2. Fixed Length has a calculated slowdown
        fixed_series = baselines_df[baselines_df["codec_display_name"] == "Fixed Length"]
        if not fixed_series.empty:
            y_val = fixed_series["slowdown"].iloc[0]
            fig.add_hline(y=y_val, line_dash="dot", line_color="red", name=f"Fixed Length_{dist}", visible=is_visible)
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Fixed Length",
                                     line=dict(color="red", dash="dot"), visible=is_visible, customdata=[dist]))

    main_title = "Random Access Slowdown vs. Sampling Rate (k)"
    default_subtitle = format_distribution_subtitle(default_dist_name) if default_dist_name else ""

    fig.update_layout(
        width=1400, height=787,
        title_text=f"{main_title}<br>{default_subtitle}",
        xaxis=dict(title="Sampling Rate (k)"),
        yaxis=dict(title="Slowdown Factor (Log Scale, relative to Uncompressed Vec)", type='log'),
        legend_title_text="Codec Type", hovermode="x unified",
        margin=dict(r=250),
        annotations=[dict(text="Select Data Distribution:", showarrow=False, x=1, y=1.15,
                          xref="paper", yref="paper", xanchor='right', yanchor='bottom')]
    )

    buttons = []
    for dist in distributions:
        visibility_arg = [dist in trace.customdata for trace in fig.data]
        shapes_arg = [dict(visible=(dist in shape.name)) for shape in fig.layout.shapes]
        new_subtitle = format_distribution_subtitle(dist)
        buttons.append(dict(label=dist, method="update", args=[
            {"visible": visibility_arg},
            {"shapes": shapes_arg, "title.text": f"{main_title}<br>{new_subtitle}"}
        ]))

    fig.update_layout(updatemenus=[dict(
        buttons=buttons, direction="down", showactive=True, active=active_index,
        x=1, xanchor="right", y=1.1, yanchor="top"
    )])

    save_plots(fig, "random_access")

def plot_parallel():
    """Generates plots for parallel performance benchmark results."""
    print("Processing parallel benchmarks...")
    df = parse_parallel_results()
    if df.empty:
        print("No parallel benchmark data found to plot."); return

    df["elapsed_ms"] = df["elapsed_seconds"] * 1000
    df["codec_display_name"] = df["codec"].apply(format_codec_name)
    distributions = sorted(df["distribution"].unique())

    default_dist_name = "Geometric" if "Geometric" in distributions else (distributions[0] if distributions else None)
    active_index = distributions.index(default_dist_name) if default_dist_name else 0

    fig = go.Figure()

    for dist in distributions:
        df_dist = df[df["distribution"] == dist].copy()
        is_visible = (dist == default_dist_name)

        baseline_times = df_dist[df_dist["method"] == "get_loop"].set_index("codec_display_name")["elapsed_ms"]
        df_dist["baseline_ms"] = df_dist["codec_display_name"].map(baseline_times)

        df_dist["speedup"] = (df_dist["baseline_ms"] / df_dist["elapsed_ms"]).round(2)
        methods_to_plot = ["get_loop", "get_many", "par_get_many"]

        for method in methods_to_plot:
            df_m = df_dist[df_dist["method"] == method].sort_values("codec_display_name")
            fig.add_trace(go.Bar(
                x=df_m["codec_display_name"], y=df_m["elapsed_ms"], name=method,
                customdata=[dist], visible=is_visible,
                text=df_m.apply(lambda r: f"{r['speedup']:.2f}x" if r['method'] != 'get_loop' else "", axis=1),
                textposition='outside',
                hovertemplate="<b>%{x}</b><br>%{data.name}<br>Time: %{y:.2f} ms<br>Speedup: %{text}<extra></extra>"
            ))

    default_subtitle = format_distribution_subtitle(default_dist_name) if default_dist_name else ""
    main_title = "Access Method Performance Comparison"

    fig.update_layout(
        width=1400, height=787, barmode='group',
        title_text=f"{main_title}<br>{default_subtitle}",
        xaxis=dict(title="Codec"),
        yaxis=dict(title="Time for 10,000 Accesses (ms)"),
        legend_title_text="Access Method",
        margin=dict(r=250),
        annotations=[dict(text="Select Data Distribution:", showarrow=False, x=1, y=1.15,
                          xref="paper", yref="paper", xanchor='right', yanchor='bottom')]
    )

    buttons = []
    for d in distributions:
        new_subtitle = format_distribution_subtitle(d)
        buttons.append(dict(label=d, method="update", args=[
            {"visible": [d in t.customdata for t in fig.data]},
            {"title.text": f"{main_title}<br>{new_subtitle}"}
        ]))

    fig.update_layout(updatemenus=[dict(
        buttons=buttons, direction="down", showactive=True, active=active_index,
        x=1, xanchor="right", y=1.1, yanchor="top"
    )])

    save_plots(fig, "parallel")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from compressed-intvec benchmark results.")
    parser.add_argument("--random-access", action="store_true", help="Generate plots for random access benchmarks.")
    parser.add_argument("--size", action="store_true", help="Generate plots for memory size benchmarks.")
    parser.add_argument("--parallel", action="store_true", help="Generate plots for parallel performance benchmarks.")
    parser.add_argument("--all", action="store_true", help="Generate all benchmark plots.")
    args = parser.parse_args()

    if not any([args.random_access, args.size, args.parallel, args.all]):
        parser.print_help()
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        if args.random_access or args.all:
            plot_random_access()
        if args.size or args.all:
            plot_size()
        if args.parallel or args.all:
            plot_parallel()
