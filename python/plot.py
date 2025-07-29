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
    Handles directory structures like "RandomAccess_UniformLow/Gamma_k=32_get/".

    Returns:
        A pandas DataFrame with columns: ['name', 'k', 'distribution', 'access_elapsed_seconds']
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
    Handles directory structures like "Geometric/Gamma_get_loop/".

    Returns:
        A pandas DataFrame with columns: ['distribution', 'codec', 'method', 'elapsed_seconds']
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
    """Converts a raw codec name from the benchmark ID to a human-readable format."""
    if name == "Baseline":
        return "Uncompressed Vec<u64>"
    if name == "FixedLength":
        return "Fixed Length"
    if name.startswith("Explicit_"):
        name = name.replace("Explicit_", "")
        if name == "VByteLe":
            return "VByte (LE)"
        if name == "VByteBe":
            return "VByte (BE)"
        return name
    return name

def format_distribution_subtitle(dist_string):
    """Converts a raw distribution string into a detailed, human-readable subtitle."""
    dist_map = {
        "UniformLow": f"Uniform (0 to 1,000), {VECTOR_SIZE:,} elements",
        "UniformHigh": f"Uniform (0 to 2<sup>32</sup>), {VECTOR_SIZE:,} elements",
        "Geometric": f"Geometric, {VECTOR_SIZE:,} elements",
        "PowerLaw": f"Power-Law, {VECTOR_SIZE:,} elements",
    }
    return dist_map.get(dist_string, f"{dist_string}, {VECTOR_SIZE:,} elements")


def save_plots(fig, base_name):
    """Saves the interactive and individual static/HTML plots."""
    benchmark_output_dir = os.path.join(OUTPUT_DIR, base_name)
    single_distr_dir = os.path.join(benchmark_output_dir, "single_distr")
    os.makedirs(single_distr_dir, exist_ok=True)

    interactive_path = os.path.join(benchmark_output_dir, f"{base_name}_interactive.html")
    print(f"  Saving interactive plot to {interactive_path}")
    fig.write_html(interactive_path)

    if not (fig.layout.updatemenus and fig.layout.updatemenus[0].buttons):
        print(f"  No dropdown found for {base_name}, skipping individual plots.")
        return

    distributions = [button['label'] for button in fig.layout.updatemenus[0].buttons]

    for dist in distributions:
        static_fig = go.Figure(fig)

        # Manually set visibility for traces in the static figure
        for trace in static_fig.data:
            is_visible = False
            # Check if customdata exists and is not empty
            if hasattr(trace, 'customdata') and trace.customdata is not None and len(trace.customdata) > 0:
                # Check if the target distribution is in the customdata list
                if dist in trace.customdata:
                    is_visible = True
            trace.visible = is_visible

        # Manually set visibility for shapes (e.g., hlines)
        if static_fig.layout.shapes:
             for shape in static_fig.layout.shapes:
                shape.visible = hasattr(shape, 'name') and shape.name is not None and dist in shape.name

        static_fig.layout.updatemenus = None
        # Remove the "Select Data Distribution" annotation
        static_fig.layout.annotations = [ann for ann in static_fig.layout.annotations if "Select Data" not in ann.text]
        subtitle = format_distribution_subtitle(dist)
        static_fig.update_layout(title_text=f"{fig.layout.title.text}<br>{subtitle}")

        base_filename = os.path.join(single_distr_dir, f"{base_name}_{dist}")
        html_path = f"{base_filename}.html"
        print(f"  Saving static plot to {html_path}")
        static_fig.write_html(html_path)
        try:
            svg_path = f"{base_filename}.svg"
            print(f"  Saving static plot to {svg_path}")
            # Ensure static images are saved with the same large aspect ratio
            static_fig.write_image(svg_path, width=1600, height=900)
        except ValueError as e:
            print(f"    Could not save SVG for {dist}: {e}. Ensure 'kaleido' is installed.")

    print(f"Finished processing for {base_name}.")


# --- Plotting Logic ---

def create_line_plot_figure(df, plot_params):
    """Creates a Plotly figure with line plots and a dropdown menu."""
    distributions = sorted(df["distribution"].unique())
    fig = go.Figure()

    default_dist_name = "Geometric" if "Geometric" in distributions else (distributions[0] if distributions else None)
    active_index = distributions.index(default_dist_name) if default_dist_name else 0

    for dist in distributions:
        df_dist = df[df["distribution"] == dist]
        is_visible = (dist == default_dist_name)

        # Plot sampled data (k > 0)
        sampled_df = df_dist[df_dist["k"] > 0]
        dist_codecs = sorted(sampled_df["codec_display_name"].unique())
        for codec_name in dist_codecs:
            df_plot = sampled_df[sampled_df["codec_display_name"] == codec_name].sort_values("k")
            fig.add_trace(go.Scatter(
                x=df_plot["k"], y=df_plot[plot_params["y_col"]], mode="lines+markers",
                name=codec_name, customdata=[dist] * len(df_plot), visible=is_visible,
                hovertemplate=plot_params["hover_template"].format(codec_name=codec_name),
            ))

        # Plot baselines (k = 0)
        baseline_df = df_dist[df_dist["k"] == 0]
        baselines_to_draw = [
            {"name": "Uncompressed Vec<u64>", "dash": "dash", "color": "black"},
            {"name": "Fixed Length", "dash": "dot", "color": "red"}
        ]
        for baseline in baselines_to_draw:
            series = baseline_df[baseline_df["codec_display_name"] == baseline["name"]]
            if not series.empty:
                y_val = series[plot_params["y_col"]].iloc[0]
                fig.add_hline(y=y_val, line_dash=baseline["dash"], line_color=baseline["color"],
                              name=f"baseline_{baseline['name']}_{dist}", visible=is_visible)
                fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=baseline["name"],
                                         line=dict(color=baseline["color"], dash=baseline["dash"]),
                                         visible=is_visible, customdata=[dist]))

    # Set initial title with subtitle for default distribution
    default_subtitle = format_distribution_subtitle(default_dist_name) if default_dist_name else ""
    main_title = plot_params["title"]

    fig.update_layout(
        width=1600, height=900,
        title_text=f"{main_title}<br>{default_subtitle}",
        xaxis_title="Sampling Rate (k)", yaxis_title=plot_params["yaxis_title"],
        legend_title_text="Codec Type", hovermode="x unified", xaxis=dict(type='category'),
        annotations=[dict(text="Select Data Distribution:", showarrow=False, x=1, y=1.15,
                          xref="paper", yref="paper", xanchor='right', yanchor='bottom', align="right")]
    )

    buttons = []
    for dist in distributions:
        visibility_arg = [(dist in trace.customdata if hasattr(trace, 'customdata') and trace.customdata is not None else False) for trace in fig.data]
        shapes_arg = [dict(visible=(dist in shape.name if hasattr(shape, 'name') and shape.name is not None else False)) for shape in fig.layout.shapes]
        new_subtitle = format_distribution_subtitle(dist)
        buttons.append(dict(
            label=dist,
            method="update",
            args=[
                {"visible": visibility_arg},
                {"shapes": shapes_arg, "title.text": f"{main_title}<br>{new_subtitle}"}
            ]
        ))

    fig.update_layout(updatemenus=[dict(
        buttons=buttons, direction="down", showactive=True, active=active_index,
        x=1, xanchor="right", y=1.1, yanchor="top"
    )])

    return fig

def plot_size():
    """Generates plots for memory size benchmark results."""
    print("Processing size benchmarks...")
    if not os.path.exists(BENCH_SIZE_CSV):
        print(f"Error: File not found at {BENCH_SIZE_CSV}"); return

    df = pd.read_csv(BENCH_SIZE_CSV).drop_duplicates()
    df["space_kb"] = df["space_bytes"] / 1024
    df["codec_display_name"] = df["name"].apply(format_codec_name)
    # Standardize baseline names and assign k=0 for plotting
    df.loc[df["codec_display_name"] == "Fixed Length", "k"] = 0
    df.loc[df["codec_display_name"] == "Uncompressed Vec<u64>", "k"] = 0

    plot_params = {
        "y_col": "space_kb",
        "title": "Memory Space vs. Sampling Rate (k)",
        "yaxis_title": "Total Space Usage (KB)",
        "hover_template": "<b>{codec_name}</b><br>k=%{{x}}<br>Space=%{{y:.1f}} KB<extra></extra>",
    }

    fig = create_line_plot_figure(df, plot_params)
    save_plots(fig, "size")

def plot_random_access():
    """Generates plots for random access benchmark results."""
    print("Processing random access benchmarks...")
    df = parse_random_access_results()
    if df.empty:
        print("No random access benchmark data found to plot."); return

    df["access_elapsed_ms"] = df["access_elapsed_seconds"] * 1000
    df["codec_display_name"] = df["name"].apply(format_codec_name)

    plot_params = {
        "y_col": "access_elapsed_ms",
        "title": "Random Access Performance vs. Sampling Rate (k)",
        "yaxis_title": "Time for 10,000 Accesses (ms)",
        "hover_template": "<b>{codec_name}</b><br>k=%{{x}}<br>Time=%{{y:.2f}} ms<extra></extra>",
    }
    fig = create_line_plot_figure(df, plot_params)
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
                x=df_m["codec_display_name"],
                y=df_m["elapsed_ms"],
                name=method,
                customdata=[dist] * len(df_m),
                visible=is_visible,
                text=df_m.apply(lambda r: f"{r['speedup']:.2f}x" if r['method'] != 'get_loop' else "", axis=1),
                textposition='outside',
                hovertemplate="<b>%{x}</b><br>%{data.name}<br>Time: %{y:.2f} ms<br>Speedup: %{text}<extra></extra>"
            ))

    # Set initial title with subtitle for default distribution
    default_subtitle = format_distribution_subtitle(default_dist_name) if default_dist_name else ""
    main_title = "Access Method Performance Comparison"

    fig.update_layout(
        width=1600, height=900, barmode='group',
        title_text=f"{main_title}<br>{default_subtitle}",
        xaxis_title="Codec", yaxis_title="Time for 10,000 Accesses (ms)",
        legend_title_text="Access Method",
        annotations=[dict(text="Select Data Distribution:", showarrow=False, x=1, y=1.15,
                          xref="paper", yref="paper", xanchor='right', yanchor='bottom', align="right")]
    )

    buttons = []
    for d in distributions:
        new_subtitle = format_distribution_subtitle(d)
        buttons.append(dict(
            label=d,
            method="update",
            args=[
                {"visible": [d in t.customdata if hasattr(t, 'customdata') else False for t in fig.data]},
                {"title.text": f"{main_title}<br>{new_subtitle}"}
            ]
        ))

    fig.update_layout(updatemenus=[dict(
        buttons=buttons,
        direction="down",
        showactive=True,
        active=active_index,
        x=1, xanchor="right",
        y=1.1, yanchor="top"
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
