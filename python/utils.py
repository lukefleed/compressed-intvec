"""
Utility functions for plotting benchmark results.
"""
import os
import plotly.graph_objects as go

OUTPUT_DIR = "images"
VECTOR_SIZE = 1_000_000

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
    """Saves interactive and static plots for distribution-based benchmarks."""
    benchmark_output_dir = os.path.join(OUTPUT_DIR, base_name)
    single_distr_dir = os.path.join(benchmark_output_dir, "single_distr")
    os.makedirs(single_distr_dir, exist_ok=True)

    interactive_path = os.path.join(benchmark_output_dir, f"{base_name}_interactive.html")
    print(f"  Saving interactive plot to {interactive_path}")
    fig.write_html(interactive_path, include_plotlyjs='cdn')

    if not (hasattr(fig.layout, 'updatemenus') and fig.layout.updatemenus and fig.layout.updatemenus[0].buttons):
        return

    # Handle distribution-based plots
    if 'customdata' in fig.data[0] and isinstance(fig.data[0].customdata[0], list) and isinstance(fig.data[0].customdata[0][0], str):
        distributions = [button['label'] for button in fig.layout.updatemenus[0].buttons]
        main_title = fig.layout.title.text.split('<br>')[0]

        for dist in distributions:
            static_fig = go.Figure(fig)
            for trace in static_fig.data:
                trace.visible = hasattr(trace, 'customdata') and dist in trace.customdata[0]
            if static_fig.layout.shapes:
                for shape in static_fig.layout.shapes:
                    shape.visible = hasattr(shape, 'name') and dist in shape.name

            static_fig.layout.updatemenus = None
            static_fig.layout.annotations = [ann for ann in static_fig.layout.annotations if "Select" not in ann.text]
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

    # Handle access-type based plots (for unchecked)
    elif 'customdata' in fig.data[0] and isinstance(fig.data[0].customdata[0], list):
        access_types = [button['label'].replace(" Access", "") for button in fig.layout.updatemenus[0].buttons]
        main_title = fig.layout.title.text.split('(')[0].strip()

        for access_type in access_types:
            static_fig = go.Figure(fig)
            # Manually set visibility based on customdata
            for trace in static_fig.data:
                trace.visible = hasattr(trace, 'customdata') and trace.customdata is not None and access_type in trace.customdata[0]

            static_fig.layout.updatemenus = None
            static_fig.layout.annotations = [ann for ann in static_fig.layout.annotations if "Select" not in ann.text]
            static_fig.update_layout(title_text=f"{main_title} ({access_type})")

            base_filename = os.path.join(single_distr_dir, f"{base_name}_{access_type.lower()}")
            html_path = f"{base_filename}.html"
            print(f"  Saving static plot to {html_path}")
            static_fig.write_html(html_path, include_plotlyjs='cdn')
            try:
                svg_path = f"{base_filename}.svg"
                print(f"  Saving static plot to {svg_path}")
                static_fig.write_image(svg_path, width=1400, height=787)
            except ValueError as e:
                print(f"    Could not save SVG for {access_type}: {e}. Ensure 'kaleido' is installed.")

    print(f"Finished processing for {base_name}.")


def save_static_plot(fig, base_name):
    """Saves a single static plot to HTML and SVG."""
    output_dir = os.path.join(OUTPUT_DIR, base_name)
    os.makedirs(output_dir, exist_ok=True)

    html_path = os.path.join(output_dir, f"{base_name}.html")
    svg_path = os.path.join(output_dir, f"{base_name}.svg")

    print(f"  Saving plot to {html_path}")
    fig.write_html(html_path, include_plotlyjs='cdn')
    print(f"  Saving plot to {svg_path}")
    fig.write_image(svg_path, width=1400, height=787)
