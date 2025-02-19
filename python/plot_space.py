"""
This script reads CSV data about space usage from various codecs and plots a line chart comparing
the space usage (in kB) for different codecs as sample size k increases. It also adds a horizontal
reference line for a "Standard Vec" (the baseline measurement) and writes the resulting plot as both
SVG and HTML files.

The CSV is expected to contain at least the following columns:
- 'name': Name of the codec.
- 'k': Sample size identifier (with k = 0 representing the standard vector).
- 'space': The space usage in bytes.

The codec names are processed to remove unnecessary prefixes/suffixes via the extract_codec_base() function.
"""

import pandas as pd
import plotly.express as px

# Read the CSV file containing benchmark results.
# The CSV file includes a 'space' column (in bytes) and other columns such as 'k' and 'name'.
df = pd.read_csv("../bench_results/bench_space.csv")

# Extract the baseline "Standard Vec" (where k == 0) and convert its space usage from byte to kB.
standard_row = df[df['k'] == 0].iloc[0]
standard_vec = standard_row['space']  # value in byte
standard_vec_kb = standard_vec / 1024  # convert to kilobytes

# Remove the baseline row from the dataframe so it doesn't interfere with plotting.
df = df[df['k'] != 0].copy()

# Ensure that the 'k' column is numeric.
df['k'] = pd.to_numeric(df['k'])
# Create a new column 'space_kb' by converting 'space' from bytes to kilobytes.
df['space_kb'] = df['space'] / 1024

# Define a helper function to extract the base name of the codec, removing unnecessary prefixes and suffixes.
def extract_codec_base(name):
    """
    Process the codec name to remove specific prefixes and suffixes:
    - Remove "LEIntVec " or "BEIntVec " prefix.
    - Remove "Param" prefix if present.
    - Remove "Codec" suffix if present.
    Returns the cleaned codec name.
    """
    # Remove "LEIntVec " or "BEIntVec " prefix.
    if name.startswith("LEIntVec "):
        s = name[len("LEIntVec "):]
    elif name.startswith("BEIntVec "):
        s = name[len("BEIntVec "):]
    else:
        s = name
    # Remove "Param" if present.
    if s.startswith("Param"):
        s = s[len("Param"):]
    # Remove "Codec" suffix if present.
    if s.endswith("Codec"):
        s = s[:-len("Codec")]
    return s.strip()

# Add a new column 'codec_base' with the cleaned codec names.
df['codec_base'] = df['name'].apply(extract_codec_base)

# --- Plotting the Total Space Usage per Codec ---
# Group the data by the cleaned codec base and 'k', aggregating the space usage (in kB) using mean.
df_total = df.groupby(['codec_base', 'k'], as_index=False)['space_kb'].mean()

# Create a line plot using Plotly Express.
# The x-axis represents 'k' (sample size), and the y-axis represents average space usage (in kB).
fig_total = px.line(
    df_total,
    x="k",
    y="space_kb",
    color="codec_base",
    markers=True,
    title="Space Usage per Codec",
    subtitle="Vector with 10k random elements with uniform distribution in the range [0, 10_000)",
    labels={
        "k": "Sample Size (k)",
        "space_kb": "Space Usage (kB)",
        "codec_base": "Codec Base"
    },
    height=900,
    width=1200
)

# Add a horizontal dashed line representing the "Standard Vec" baseline value.
fig_total.add_hline(
    y=standard_vec_kb,
    line_dash="dash",
    line_color="black",
    annotation_text="Standard Vec",
    annotation_position="bottom right"
)

# Display the plotted figure.
fig_total.show()

# Save the plot as an SVG image and an interactive HTML file in the specified directory.
fig_total.write_image("../images/space/space_total_10k.svg")
fig_total.write_html("../images/space/space_total_10k.html")
