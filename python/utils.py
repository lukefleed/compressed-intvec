import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
CRITERION_DIR = os.path.join(PROJECT_ROOT, "target", "criterion")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "images")

VECTOR_SIZE = 10_000_000
NUM_ACCESSES = 1_000_000
OPS_PER_THREAD = 10_000

def format_number(n):
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}k"
    return str(n)

def format_fixed_access_name(name):
    if name.startswith("Baseline_Vec"):
        match = re.search(r"Vec<(\w+)>", name)
        if match:
            return f"Vec<{match.group(1)}> (Baseline)"
        return name.replace("Baseline_", "").replace("/Unchecked", " (Baseline)")
    
    if name.startswith("LEFixedVec"):
        if "Unaligned-Unchecked" in name:
            return "FixedVec (Unaligned)"
        return "FixedVec (Aligned)"

    if name.startswith("sux::BitFieldVec"):
        if "Unaligned-Unchecked" in name:
            return "sux::BitFieldVec (Unaligned)"
        return "sux::BitFieldVec (Aligned)"
    
    if name.startswith("succinct::IntVector"):
        return "succinct::IntVector"
    
    if name.startswith("simple-sds-sbwt::IntVector"):
        return "simple-sds-sbwt::IntVector"

    return name

def format_codec_name(name):
    name_map = {
        "Baseline": "Vec<u64>",
        "FixedVec": "FixedVec",
        "sux::BitFieldVec": "sux::BitFieldVec",
        "succinct::IntVector": "succinct::IntVector",
        "VByteLe": "VByte (LE)",
    }
    return name_map.get(name, name)

def format_distribution_subtitle(dist_string):
    dist_map = {
        "UniformLow": f"Uniform (0 to 1,000), {format_number(VECTOR_SIZE)} elements, {format_number(NUM_ACCESSES)} accesses",
        "UniformHigh": f"Uniform (0 to 2^32), {format_number(VECTOR_SIZE)} elements, {format_number(NUM_ACCESSES)} accesses",
        "RiceImplied": f"Rice-Implied, {format_number(VECTOR_SIZE)} elements, {format_number(NUM_ACCESSES)} accesses",
        "ZetaImplied": f"Zeta-Implied, {format_number(VECTOR_SIZE)} elements, {format_number(NUM_ACCESSES)} accesses",
    }
    return dist_map.get(dist_string, f"{dist_string}, {format_number(VECTOR_SIZE)} elements, {format_number(NUM_ACCESSES)} accesses")

def save_plot(fig, filename_base):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    svg_path = os.path.join(OUTPUT_DIR, f"{filename_base}.svg")
    try:
        fig.write_image(svg_path)
        print(f"Plot saved to {svg_path}")
    except Exception as e:
        print(f"Error saving SVG plot: {e}")
        print("Please ensure you have Kaleido installed (`pip install kaleido`)")

    html_path = os.path.join(PROJECT_ROOT, "src", "pages", "bench-intvec", f"{filename_base}.html")
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    try:
        fig.write_html(html_path, full_html=False, include_plotlyjs='cdn')
        print(f"Plot saved to {html_path}")
    except Exception as e:
        print(f"Error saving HTML plot: {e}")

def normalize_atomic_implementation_name(raw_name):
    """
    Cleans up benchmark implementation names.
    e.g., "Baseline_Vec<AtomicU16>" -> "Baseline Vec<AtomicU16>"
    """
    name = raw_name
    if name.endswith("_store"):
        name = name[:-6]
    
    name = name.replace("__", "::") # sux::AtomicBitFieldVec
    name = name.replace("_", " ") # Baseline_Vec<...>

    return name