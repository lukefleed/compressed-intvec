import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from parsing import (
    parse_random_access_results, 
    parse_fixed_access_results, 
    parse_atomic_scaling_results
)
from utils import (
    format_codec_name, format_distribution_subtitle, format_fixed_access_name, 
    save_plot, NUM_ACCESSES, VECTOR_SIZE, format_number, OPS_PER_THREAD
)

def plot_fixed_access():
    print("Processing fixed-width random access benchmarks...")
    df = parse_fixed_access_results()
    if df.empty:
        print("No fixed-width random access benchmark data found to plot.")
        return

    df["time_ms"] = df["time_ns"] / 1e6
    df["display_name"] = df["name"].apply(format_fixed_access_name)

    fig = go.Figure()

    baseline_df = df[df["name"].str.startswith("Baseline_Vec")].copy()
    other_df = df[~df["name"].str.startswith("Baseline_Vec")]

    if not baseline_df.empty:
        baseline_df = baseline_df.sort_values("bit_width")
        fig.add_trace(go.Scatter(
            x=baseline_df["bit_width"], y=baseline_df["time_ms"],
            mode="lines+markers", name="Vec<T>",
            line=dict(color='black', dash='dash'),
            hovertemplate=("<b>%{text}</b><br>" + "Bit Width: %{x}<br>" + "Time: %{y:.2f} ms<extra></extra>"),
            text=baseline_df["display_name"]
        ))

    color_palette = px.colors.qualitative.Plotly
    unique_names = sorted(other_df["display_name"].unique())

    for i, name in enumerate(unique_names):
        df_plot = other_df[other_df["display_name"] == name].sort_values("bit_width")
        color = color_palette[i % len(color_palette)]
        fig.add_trace(go.Scatter(
            x=df_plot["bit_width"], y=df_plot["time_ms"],
            mode="lines+markers", name=name,
            line=dict(color=color), marker=dict(color=color),
            hovertemplate=("<b>" + name + "</b><br>" + "Bit Width: %{x}<br>" + "Time: %{y:.2f} ms<extra></extra>")
        ))

    main_title = "Fixed-Width Random Access Performance"
    subtitle = f"{format_number(NUM_ACCESSES)} random reads on a vector of {format_number(VECTOR_SIZE)} elements"
    fig.update_layout(
        title_text=f"{main_title}<br><i>{subtitle}</i>",
        xaxis=dict(title="Bit Width", dtick=4),
        yaxis=dict(title=f"Time for {format_number(NUM_ACCESSES)} accesses (ms, lower is better, log scale)", type='log'),
        legend=dict(
            x=0.99, y=0.99, 
            xanchor='left', yanchor='top',
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)',
            borderwidth=1,
            font=dict(size=12)
        ),
        legend_title_text="Implementation", 
        hovermode="x unified",
        width=1200, height=900, 
        font=dict(size=14),
        template="ggplot2",
        margin=dict(l=80, r=80, t=80, b=60)
    )
    save_plot(fig, "fixed_random_access_performance")

def plot_atomic_scaling(group_prefix, bit_width, baseline_name, filename_suffix, title_suffix):
    """
    Generates a plot for atomic scaling benchmarks.
    """
    print(f"Processing atomic scaling benchmarks for {group_prefix}...")
    df = parse_atomic_scaling_results(group_prefix)
    if df.empty:
        print(f"No data found for benchmark group '{group_prefix}'.")
        return

    styles = {
        "AtomicFixedVec": {"color": "red", "dash": "solid"},
        "sux::AtomicBitFieldVec": {"color": "blue", "dash": "dot"},
        "Baseline Vec<AtomicU16>": {"color": "black", "dash": "dash"},
    }
    
    implementations_to_plot = ["AtomicFixedVec", "sux::AtomicBitFieldVec", baseline_name]
    df_filtered = df[df['implementation'].isin(implementations_to_plot)].copy()

    if df_filtered.empty:
        print(f"No relevant data found for the specified implementations in '{group_prefix}'.")
        return

    fig = go.Figure()

    legend_map = {
        "AtomicFixedVec": f"AtomicFixedVec ({bit_width}-bit)",
        "sux::AtomicBitFieldVec": f"sux::AtomicBitFieldVec ({bit_width}-bit)",
        baseline_name: baseline_name
    }

    for impl_name in sorted(df_filtered['implementation'].unique()):
        if impl_name not in implementations_to_plot:
            continue
            
        df_plot = df_filtered[df_filtered['implementation'] == impl_name].sort_values("num_threads")
        if df_plot.empty:
            continue
        
        legend_name = legend_map.get(impl_name, impl_name)
        style = styles.get(impl_name, {})
        fig.add_trace(go.Scatter(
            x=df_plot["num_threads"],
            y=df_plot["throughput_mops"],
            mode="lines+markers",
            name=legend_name,
            line=style,
            hovertemplate=(
                "<b>" + legend_name + "</b><br>" +
                "Threads: %{x}<br>" +
                "Throughput: %{y:.2f} M ops/sec<extra></extra>"
            )
        ))

    main_title = f"Atomic store Throughput vs. Thread Count ({title_suffix})"
    subtitle = f"Diffuse Contention: 100K random ops/thread"

    fig.update_layout(
        title_text=f"{main_title}<br><i>{subtitle}</i>",
        xaxis=dict(title="Number of Threads", type='category'),
        yaxis=dict(title="Throughput (Million ops/sec, higher is better)"),
        legend=dict(
            x=0.01, y=0.99, 
            xanchor='left', yanchor='top',
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)',
            borderwidth=1,
            font=dict(size=12)
        ),
        legend_title_text="Implementation",
        hovermode="x unified",
        width=1200,
        height=900,
        font=dict(size=14),
        template="ggplot2",
        margin=dict(l=80, r=30, t=80, b=60)
    )
    
    save_plot(fig, f"atomic_scaling_{filename_suffix}")

def plot_random_access():
    print("Processing variable-width random access benchmarks...")
    df = parse_random_access_results()
    if df.empty:
        print("No variable-width random access benchmark data found to plot.")
        return

    df["time_ms"] = df["time_seconds"] * 1000
    df["codec_display_name"] = df["name"].apply(format_codec_name)

    distributions = sorted(df["distribution"].unique())
    if not distributions:
        print("No distributions found in benchmark data.")
        return

    for dist in distributions:
        baseline_df = df[(df["distribution"] == dist) & (df["name"] == "Baseline")]
        if not baseline_df.empty:
            baseline_time = baseline_df["time_seconds"].iloc[0]
            df.loc[df["distribution"] == dist, "slowdown"] = df["time_seconds"] / baseline_time
        else:
            print(f"Warning: Baseline data not found for distribution '{dist}'. Cannot calculate slowdown.")
            df.loc[df["distribution"] == dist, "slowdown"] = float('nan')
            
    fixed_styles = {
        "Vec<u64>": {"color": "black", "dash": "dash"},
        "FixedVec": {"color": "#EF553B", "dash": "dot"},
        "sux::BitFieldVec": {"color": "#00CC96", "dash": "dot"},
        "succinct::IntVector": {"color": "#AB63FA", "dash": "dot"}
    }

    for dist in distributions:
        fig = go.Figure()
        df_dist = df[df["distribution"] == dist].copy()

        fixed_df = df_dist[df_dist["k"] == 0].sort_values(by="slowdown")
        sampled_df = df_dist[df_dist["k"] > 0]

        for codec_name in sorted(sampled_df["codec_display_name"].unique()):
            df_plot = sampled_df[sampled_df["codec_display_name"] == codec_name].sort_values("k")
            fig.add_trace(go.Scatter(
                x=df_plot["k"], y=df_plot["slowdown"], mode="lines+markers",
                name=codec_name,
                text=[f"{t:.2f} ms" for t in df_plot["time_ms"]],
                hovertemplate=("<b>" + codec_name + "</b><br>" + "k=%{x}<br>" + "Slowdown: %{y:.2f}x<br>" + "Absolute Time: %{text}<extra></extra>")
            ))

        for _, row in fixed_df.iterrows():
            name = row["codec_display_name"]
            slowdown = row["slowdown"]
            style = fixed_styles.get(name, {"color": "grey", "dash": "dot"})

            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='lines',
                line=dict(color=style["color"], dash=style["dash"]),
                name=name,
            ))
            fig.add_hline(
                y=slowdown,
                line_dash=style["dash"],
                line_color=style["color"]
            )

        main_title = "Random Access Performance: Slowdown vs. Sampling Rate (k)"
        subtitle = format_distribution_subtitle(dist)

        fig.update_layout(
            title_text=f"{main_title}<br><i>{subtitle}</i>",
            xaxis=dict(title="Sampling Rate (k)"),
            yaxis=dict(title="Slowdown Factor (Log Scale, relative to Vec&lt;u64&gt;)", type='log', tickformat=".1f"),
            legend=dict(
                x=0.01, y=0.99, 
                xanchor='left', yanchor='top',
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(0,0,0,0)',
                borderwidth=1,
                font=dict(size=12)
            ),
            legend_title_text="Implementation Type",
            hovermode="x unified",
            width=1200,
            height=900,
        )

        save_plot(fig, f"random_access_slowdown_{dist}")