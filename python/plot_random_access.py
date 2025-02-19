"""Plot Random Access Benchmark Results

This script reads benchmark results from a CSV file, processes the data, and creates an
interactive line plot using Plotly. It uses the elapsed time measurements (converted from
seconds to milliseconds) to compare the performance of different integer vector implementations.
"""

import pandas as pd
import plotly.express as px

# Read benchmark results from CSV file.
df = pd.read_csv("../bench_results/bench_random_access.csv")

# Extract the row corresponding to the standard vector (sample size k == 0) and its elapsed time.
standard_row = df[df['k'] == 0].iloc[0]
standard_vec = standard_row['elapsed']
standard_vec_ms = standard_vec  # Standard time in milliseconds

# Filter out the standard vector row from the DataFrame.
df = df[df['k'] != 0].copy()

# Convert the 'k' column to numeric for accurate processing.
df['k'] = pd.to_numeric(df['k'])
# Multiply elapsed times by 1000 to convert seconds to milliseconds.
df['elapsed'] = df['elapsed'] * 1000

def extract_codec_base(name):
    """
    Remove the 'LEIntVec ' or 'BEIntVec ' prefix from the codec name.

    Args:
        name (str): The original codec name.

    Returns:
        str: The cleaned codec name.
    """
    if name.startswith("LEIntVec "):
        s = name[len("LEIntVec "):]
    elif name.startswith("BEIntVec "):
        s = name[len("BEIntVec "):]
    else:
        s = name
    return s.strip()

# Clean codec names to get their base values.
df['codec_base'] = df['name'].apply(extract_codec_base)

# Group data by codec base and sample size, calculating the mean elapsed time.
df_total = df.groupby(['codec_base', 'k'], as_index=False)['elapsed'].mean()

# Create a line plot displaying the average access time versus sample size for each codec.
fig_total = px.line(
    df_total,
    x="k",
    y="elapsed",
    color="codec_base",
    markers=True,
    title="Time to Randomly Access Elements 10k elements",
    subtitle="Vector with 10k random elements with uniform distribution in the range [0, 100_000). Indices are randomly generated.",
    labels={
        "k": "Sample Size (k)",
        "elapsed": "Time to Access (ms)",
        "codec_base": "Codec Base"
    },
    height=600,
    width=1000
)

# Add a horizontal dashed line to indicate the standard vector's elapsed time.
fig_total.add_hline(
    y=standard_vec_ms,
    line_dash="dash",
    line_color="black",
    annotation_text="Standard Vec",
    annotation_position="bottom right"
)

# Display the interactive plot.
fig_total.show()

# Save the plot as an SVG image and an interactive HTML file.
fig_total.write_image("../images/random_access/time_total_100k.svg")
fig_total.write_html("../images/random_access/time_total_100k.html")
