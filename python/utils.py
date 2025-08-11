import os

# Directory where Criterion saves its output files.
CRITERION_DIR = "target/criterion"
# Directory where generated plot images will be saved.
OUTPUT_DIR = "images"

# Constants from the benchmark file for use in titles and labels.
VECTOR_SIZE = 1_000_000
NUM_ACCESSES = 100_000

def format_codec_name(name):
    """Converts a raw benchmark name to a human-readable format for legends."""
    name_map = {
        "Baseline": "Vec<u64>",
        "FixedVec": "FixedVec",
        "sux::BitFieldVec": "sux::BitFieldVec",
        "succinct::IntVector": "succinct::IntVector",
        "VByteLe": "VByte (LE)",
    }
    return name_map.get(name, name)

def format_distribution_subtitle(dist_string):
    """Converts a raw distribution string into a detailed subtitle for plots."""
    dist_map = {
        "UniformLow": f"Uniform (0 to 1,000), {VECTOR_SIZE:,} elements, {NUM_ACCESSES:,} accesses",
        "UniformHigh": f"Uniform (0 to 2^32), {VECTOR_SIZE:,} elements, {NUM_ACCESSES:,} accesses",
        "RiceImplied": f"Rice-Implied, {VECTOR_SIZE:,} elements, {NUM_ACCESSES:,} accesses",
        "ZetaImplied": f"Zeta-Implied, {VECTOR_SIZE:,} elements, {NUM_ACCESSES:,} accesses",
    }
    return dist_map.get(dist_string, f"{dist_string}, {VECTOR_SIZE:,} elements, {NUM_ACCESSES:,} accesses")

def save_plot(fig, filename_base):
    """Saves a Plotly figure to an SVG file."""
    svg_path = os.path.join(OUTPUT_DIR, f"{filename_base}.svg")
    fig.write_image(svg_path)
    print(f"Plot saved to {svg_path}")