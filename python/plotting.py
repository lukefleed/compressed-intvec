"""
Module for generating plots from processed benchmark data.
"""
import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
from parsing import (
    parse_size_results,
    parse_random_access_results,
    parse_parallel_results,
    parse_unchecked_results
)
from utils import (
    format_codec_name,
    format_distribution_subtitle,
    save_plots,
    save_static_plot
)


def plot_size():
    """Generates plots for memory size benchmark results using dedicated logic."""
    print("Processing size benchmarks...")
    df = parse_size_results()
    if df.empty:
        return

    df["space_kb"] = df["space_bytes"] / 1024
    df["clean_distribution"] = df["distribution"].apply(lambda x: x.split('_')[0])

    distributions = sorted(df["clean_distribution"].unique())
    fig = go.Figure()

    default_dist_name = "UniformLow" if "UniformLow" in distributions else (distributions[0] if distributions else None)
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
                name=codec_name, customdata=[[dist]] * len(df_plot), visible=is_visible,
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
                                     visible=is_visible, customdata=[[dist]]))

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
        visibility_arg = [dist in trace.customdata[0] for trace in fig.data]
        shapes_arg = [dict(visible=(hasattr(shape, 'name') and dist in shape.name)) for shape in (fig.layout.shapes or [])]
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

    default_dist_name = "UniformLow" if "UniformLow" in distributions else (distributions[0] if distributions else None)
    active_index = distributions.index(default_dist_name) if default_dist_name else 0

    for dist in distributions:
        df_dist = df[df["distribution"] == dist].copy()
        is_visible = (dist == default_dist_name)

        baselines_df = df_dist[df_dist["codec_display_name"].isin(["Uncompressed Vec<u64>", "Fixed Length"])]
        sampled_df = df_dist[~df_dist["codec_display_name"].isin(["Uncompressed Vec<u64>", "Fixed Length"])]

        for codec_name in sorted(sampled_df["codec_display_name"].unique()):
            df_plot = sampled_df[sampled_df["codec_display_name"] == codec_name].sort_values("k")
            fig.add_trace(go.Scatter(
                x=df_plot["k"], y=df_plot["slowdown"], mode="lines+markers",
                name=codec_name, customdata=[[dist]] * len(df_plot), visible=is_visible,
                text=[f"{t:.2f} ms" for t in df_plot["access_elapsed_ms"]],
                hovertemplate=("<b>" + codec_name + "</b><br>" +
                               "k=%{x}<br>" +
                               "Slowdown: %{y:.1f}x<br>" +
                               "Absolute Time: %{text}<extra></extra>")
            ))

        fig.add_hline(y=1, line_dash="dash", line_color="black", name=f"Uncompressed Vec<u64>_{dist}", visible=is_visible)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Uncompressed Vec<u64>",
                                 line=dict(color="black", dash="dash"), visible=is_visible, customdata=[[dist]]))

        fixed_series = baselines_df[baselines_df["codec_display_name"] == "Fixed Length"]
        if not fixed_series.empty:
            y_val = fixed_series["slowdown"].iloc[0]
            fig.add_hline(y=y_val, line_dash="dot", line_color="red", name=f"Fixed Length_{dist}", visible=is_visible)
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Fixed Length",
                                     line=dict(color="red", dash="dot"), visible=is_visible, customdata=[[dist]]))

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
        visibility_arg = [dist in trace.customdata[0] for trace in fig.data]
        shapes_arg = [dict(visible=(hasattr(shape, 'name') and dist in shape.name)) for shape in (fig.layout.shapes or [])]
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

    default_dist_name = "UniformLow" if "UniformLow" in distributions else (distributions[0] if distributions else None)
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
                customdata=[[dist]] * len(df_m), visible=is_visible,
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
        yaxis=dict(title="Time for 1,000,000 Accesses (ms)"),
        legend_title_text="Access Method",
        margin=dict(r=250),
        annotations=[dict(text="Select Data Distribution:", showarrow=False, x=1, y=1.15,
                          xref="paper", yref="paper", xanchor='right', yanchor='bottom')]
    )

    buttons = []
    for d in distributions:
        new_subtitle = format_distribution_subtitle(d)
        buttons.append(dict(label=d, method="update", args=[
            {"visible": [d in t.customdata[0] for t in fig.data]},
            {"title.text": f"{main_title}<br>{new_subtitle}"}
        ]))

    fig.update_layout(updatemenus=[dict(
        buttons=buttons, direction="down", showactive=True, active=active_index,
        x=1, xanchor="right", y=1.1, yanchor="top"
    )])

    save_plots(fig, "parallel")

def plot_unchecked():
    """Generates a static bar plot for unchecked random access benchmarks."""
    print("\nProcessing unchecked access benchmarks...")
    df = parse_unchecked_results()
    if df.empty:
        print("No unchecked access benchmark data found to plot.")
        return

    df["time_ms"] = df["time_s"] * 1000

    # Filter for the specific bit widths and only unchecked access
    target_bit_widths = [7, 8, 15, 16, 31, 32]
    df_filtered = df[df['bit_width'].isin(target_bit_widths) & (df['access_type'] == 'Unchecked')]

    # Separate the baseline for the reference line
    baseline_df = df[(df['implementation'] == 'Baseline_Vec<u64>') & (df['access_type'] == 'Unchecked')]
    baseline_time_ms = baseline_df['time_ms'].mean() if not baseline_df.empty else 0

    # Implementations to plot as bars (excluding the baseline which will be a line)
    impl_to_plot = [name for name in df_filtered['implementation'].unique() if name != 'Baseline_Vec<u64>']
    df_plot = df_filtered[df_filtered['implementation'].isin(impl_to_plot)]

    fig = go.Figure()

    for impl_name in sorted(impl_to_plot):
        df_impl = df_plot[df_plot['implementation'] == impl_name]
        fig.add_trace(go.Bar(
            x=df_impl["bit_width"],
            y=df_impl["time_ms"],
            name=impl_name,
            hovertemplate=f"<b>{impl_name}</b><br>Bit Width: %{{x}} bits<br>Time: %{{y:.2f}} ms<extra></extra>"
        ))

    # Add the baseline as a horizontal dashed line
    if baseline_time_ms > 0:
        fig.add_hline(
            y=baseline_time_ms,
            line_dash="dash",
            line_color="red",
            annotation_text="Baseline Vec<u64> (Unchecked)",
            annotation_position="bottom right"
        )

    fig.update_layout(
        title_text="Unchecked Random Access Performance vs. Bit Width",
        xaxis_title="Bit Width per Integer",
        yaxis_title="Time for 1M Accesses (ms)",
        barmode='group',
        xaxis=dict(type='category'),
        legend_title_text="Implementation",
        height=700,
        width=1400,
    )

    save_static_plot(fig, "unchecked_access_performance_bars")
