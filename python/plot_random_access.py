import pandas as pd
import plotly.express as px

df = pd.read_csv("../bench_results/bench_random_access.csv")

standard_row = df[df['k'] == 0].iloc[0]
standard_vec = standard_row['elapsed']
standard_vec_ms = standard_vec

df = df[df['k'] != 0].copy()

df['k'] = pd.to_numeric(df['k'])
df['elapsed'] = df['elapsed'] * 1000

def extract_codec_base(name):
    # Rimuovi il prefisso "LEIntVec " o "BEIntVec "
    if name.startswith("LEIntVec "):
        s = name[len("LEIntVec "):]
    elif name.startswith("BEIntVec "):
        s = name[len("BEIntVec "):]
    else:
        s = name
    return s.strip()

df['codec_base'] = df['name'].apply(extract_codec_base)

df_total = df.groupby(['codec_base', 'k'], as_index=False)['elapsed'].mean()

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

fig_total.add_hline(
    y=standard_vec_ms,
    line_dash="dash",
    line_color="black",
    annotation_text="Standard Vec",
    annotation_position="bottom right"
)

fig_total.show()
# write the svg and the html in images/random_access
fig_total.write_image("../images/random_access/time_total_100k.svg")
fig_total.write_html("../images/random_access/time_total_100k.html")
