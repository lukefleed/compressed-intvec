import pandas as pd
import plotly.graph_objects as go
from parsing import parse_random_access_results
from utils import format_codec_name, format_distribution_subtitle, save_plot

def plot_random_access():
    """Generates a separate plot for each distribution for the random access benchmark."""
    print("Processing random access benchmarks...")
    df = parse_random_access_results()
    if df.empty:
        print("No random access benchmark data found to plot.")
        return

    df["time_ms"] = df["time_seconds"] * 1000
    df["codec_display_name"] = df["name"].apply(format_codec_name)

    distributions = sorted(df["distribution"].unique())
    if not distributions:
        print("No distributions found in benchmark data.")
        return

    # Calculate slowdown factor relative to the baseline for each distribution.
    for dist in distributions:
        baseline_df = df[(df["distribution"] == dist) & (df["name"] == "Baseline")]
        if not baseline_df.empty:
            baseline_time = baseline_df["time_seconds"].iloc[0]
            df.loc[df["distribution"] == dist, "slowdown"] = df["time_seconds"] / baseline_time
        else:
            print(f"Warning: Baseline data not found for distribution '{dist}'. Cannot calculate slowdown.")
            df.loc[df["distribution"] == dist, "slowdown"] = float('nan')
            
    # Define styles for fixed-width vectors to ensure they are distinct.
    fixed_styles = {
        "Vec<u64>": {"color": "black", "dash": "dash"},
        "FixedVec": {"color": "#EF553B", "dash": "dot"},
        "sux::BitFieldVec": {"color": "#00CC96", "dash": "dot"},
        "succinct::IntVector": {"color": "#AB63FA", "dash": "dot"}
    }

    # Generate one plot per distribution.
    for dist in distributions:
        fig = go.Figure()
        df_dist = df[df["distribution"] == dist].copy()

        fixed_df = df_dist[df_dist["k"] == 0].sort_values(by="slowdown")
        sampled_df = df_dist[df_dist["k"] > 0]

        # Plot variable-width codecs.
        for codec_name in sorted(sampled_df["codec_display_name"].unique()):
            df_plot = sampled_df[sampled_df["codec_display_name"] == codec_name].sort_values("k")
            fig.add_trace(go.Scatter(
                x=df_plot["k"], y=df_plot["slowdown"], mode="lines+markers",
                name=codec_name,
                text=[f"{t:.2f} ms" for t in df_plot["time_ms"]],
                hovertemplate=("<b>" + codec_name + "</b><br>" +
                               "k=%{x}<br>" +
                               "Slowdown: %{y:.2f}x<br>" +
                               "Absolute Time: %{text}<extra></extra>")
            ))

        # Plot fixed-width vectors: add a dummy trace for the legend, then the hline.
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
            legend_title_text="Implementation Type",
            hovermode="x unified",
            width=1200,
            height=700,
        )

        save_plot(fig, f"random_access_slowdown_{dist}")