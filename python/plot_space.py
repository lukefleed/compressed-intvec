import pandas as pd
import plotly.express as px

# Leggi il CSV dal percorso specificato
df = pd.read_csv("../bench_results/bench_space.csv")

# Salva il valore di riferimento "Standard Vec" (k=0) e converti in kB
standard_row = df[df['k'] == 0].iloc[0]
standard_vec = standard_row['space']  # valore in byte
standard_vec_kb = standard_vec / 1024

# Rimuovi la riga di riferimento (k = 0)
df = df[df['k'] != 0].copy()

# Assicurati che 'k' sia numerico e converti lo spazio da byte a kB
df['k'] = pd.to_numeric(df['k'])
df['space_kb'] = df['space'] / 1024

# Funzione per estrarre il nome base del codec (senza prefissi/suffissi inutili)
def extract_codec_base(name):
    # Rimuovi il prefisso "LEIntVec " o "BEIntVec "
    if name.startswith("LEIntVec "):
        s = name[len("LEIntVec "):]
    elif name.startswith("BEIntVec "):
        s = name[len("BEIntVec "):]
    else:
        s = name
    # Rimuovi "Param" se presente all'inizio
    if s.startswith("Param"):
        s = s[len("Param"):]
    # Rimuovi il suffisso "Codec" se presente
    if s.endswith("Codec"):
        s = s[:-len("Codec")]
    return s.strip()

# Aggiungi la colonna 'codec_base'
df['codec_base'] = df['name'].apply(extract_codec_base)

# --- Grafico Totale ---
# Raggruppa per codec_base e per k (le implementazioni sono identiche, quindi si usa la media)
df_total = df.groupby(['codec_base', 'k'], as_index=False)['space_kb'].mean()

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

# Aggiungi linea orizzontale per "Standard Vec"
fig_total.add_hline(
    y=standard_vec_kb,
    line_dash="dash",
    line_color="black",
    annotation_text="Standard Vec",
    annotation_position="bottom right"
)
fig_total.show()
# write the svg and the html in images/space
fig_total.write_image("../images/space/space_total_10k.svg")
fig_total.write_html("../images/space/space_total_10k.html")
