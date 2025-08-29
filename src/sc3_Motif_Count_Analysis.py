import pandas as pd
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
import networkx as nx
from networkx.algorithms import isomorphism
import math
import numpy as np

# ------------------------------------------------------
# Script: sc3_Motif_Count_Analysis.py
# Purpose: EDA of the statistical link between orbit participation and class label
# , visualisation of chosen orbits
# Assumes:
#  - data/all_node_orbit_counts.parquet contains columns:
#      int_id, orbit_0...orbit_N, window_start
# ------------------------------------------------------

#Spark session
spark = (
    SparkSession.builder
        .appName("ReloadMotifResults")
        .getOrCreate()
)

# Load integer-ID mapping from Parquet

verts_int_spark = spark.read.parquet("data/verts_int.parquet")
verts_pd = verts_int_spark.select("txId", "int_id") \
                         .toPandas().astype({"txId": str})

#class labels from CSV
#  class==1 illicit, class==2 licit
classes_pd = pd.read_csv("data/txs_classes.csv", usecols=["txId","class"]) \
               .astype({"txId": str})

#  LOAD + AGGREGATE ORBIT COUNTS PER NODE
df_orbits_raw = pd.read_csv("data/all_node_orbit_counts.csv")
# Identify all orbit columns 
orbit_cols = [c for c in df_orbits_raw.columns if c.startswith("orbit_")]
# Sum across windows so each int_id appears exactly once,
# with its total motif counts over time:
df_orbits = (
    df_orbits_raw
      .groupby("int_id")[orbit_cols]
      .sum()
      .reset_index()
)

# mappings, class labels, and aggregated orbit counts all marged
merged = (
    verts_pd
      .merge(classes_pd, on="txId", how="inner")
      .merge(df_orbits,  on="int_id", how="inner")
)
# Keep only the two labeled classes for this analysis
merged = merged[merged["class"].isin([1, 2])]

#summing total orbit counts per class across all nodes
agg = (
    merged
      .groupby("class")[orbit_cols]
      .sum()
      .reset_index()
)

# Turn into tidy table for plotting
tidy = agg.melt(id_vars="class", var_name="orbit", value_name="count")
pivot = tidy.pivot(index="orbit", columns="class", values="count").fillna(0)
pivot.columns = ["Illicit", "Licit"]

# Illicit vs Licit count plot
fig, ax = plt.subplots(figsize=(8,4))
pivot.plot(kind="bar", ax=ax)
ax.set_title("4-Node Motif Orbit Counts: Illicit vs Licit")
ax.set_xlabel("Orbit ID")
ax.set_ylabel("Total Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Fisher's exact test per orbit
total_il = pivot["Illicit"].sum()
total_li = pivot["Licit"].sum()
results = []
for orbit, row in pivot.iterrows():
    a, b = int(row["Illicit"]), int(row["Licit"])
    if a + b == 0:
        continue
    _, p = fisher_exact([[a, b], [total_il - a, total_li - b]])
    results.append((orbit, a, b, p))

# Per-orbit Fisher p-value table (two-sided), with FDR and -log10(p)
rows = []
for orbit, row in pivot.iterrows():
    a = int(row["Illicit"])
    b = int(row["Licit"])
    if a + b == 0:
        continue
    c = int(total_il - a)
    d = int(total_li - b)
    or_val, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    rows.append({"orbit": orbit, "illicit": a, "licit": b, "odds_ratio": or_val, "p_value": p})

p_table = pd.DataFrame(rows)

# Benjamini–Hochberg FDR adjustment
p_table = p_table.sort_values("p_value").reset_index(drop=True)
m = len(p_table)
ranks = np.arange(1, m + 1, dtype=float)
p_sorted = p_table["p_value"].to_numpy()
# Monotone BH
q_sorted = np.minimum.accumulate((p_sorted * m / ranks)[::-1])[::-1]
p_table["q_value_fdr"] = np.clip(q_sorted, 0, 1)

# Convenience column for readability
tiny = np.finfo(float).tiny
p_table["minus_log10_p"] = -np.log10(np.clip(p_table["p_value"].to_numpy(), tiny, None))

#  printout
print("\nPer-orbit Fisher exact test (two-sided):")
print(p_table.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

# Optional: save to CSV
# p_table.to_csv("orbit_fisher_pvalues.csv", index=False)

# odds ratio plot
plot_df = (
    pivot
      .copy()
      .reset_index()
      .rename(columns={"index":"orbit"})
)
plot_df["odds_ratio"] = plot_df.apply(
    lambda r: fisher_exact(
        [[r["Illicit"], r["Licit"]],
         [total_il - r["Illicit"], total_li - r["Licit"]]]
    )[0], axis=1
)
plot_df = plot_df.sort_values("odds_ratio", ascending=False)

fig, ax = plt.subplots(figsize=(10,5))
ax.bar(plot_df["orbit"], plot_df["odds_ratio"])
ax.set_xlabel("Orbit ID")
ax.set_ylabel("Odds Ratio (Illicit vs Licit)")
ax.set_title("4-Node Motif Orbit Enrichment")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

print("Top 5 orbits most enriched in illicit transactions:")
print(plot_df.head(5).to_string(index=False))

# visualising the motifs
##Visualising orbits: 
import networkx as nx
import matplotlib.pyplot as plt

# Define orbit structures
def get_orbit_graph(orbit_id):
    G = nx.Graph()
    
    # Each graph structure based on standard ORCA orbit mappings
    if orbit_id == 0:
        G.add_edges_from([(0, 1)])  # single edge
    elif orbit_id == 1:
        G.add_edges_from([(0, 1), (0, 2)])  # 2-star
    elif orbit_id == 2:
        G.add_edges_from([(0, 1), (1, 2)])  # 2-path
    elif orbit_id == 3:
        G.add_edges_from([(0, 1), (1, 2), (0, 2)])  # triangle
    elif orbit_id == 4:
        G.add_edges_from([(0, 1), (0, 2), (0, 3)])  # 3-star
    elif orbit_id == 5:
        G.add_edges_from([(0, 1), (0, 2), (2, 3)])  # fork
    elif orbit_id == 6:
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])  # 3-path
    elif orbit_id == 7:
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 2)])  # path + triangle tail
    elif orbit_id == 8:
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 3)])  # square (cycle 4)
    elif orbit_id == 9:
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 2)])  # tailed triangle
    elif orbit_id == 10:
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 3)])  # path + triangle head
    elif orbit_id == 11:
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 2), (0, 3)])  # complex 4-node motif
    elif orbit_id == 12:
        G.add_edges_from([(0, 1), (0, 2), (0, 3), (2, 3)])  # 3-star + edge
    elif orbit_id == 13:
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 2), (0, 3)])  # diamond
    elif orbit_id == 14:
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)])  # complete 4-node graph (K4)
    
    return G

# Plot all orbits
plt.figure(figsize=(15, 8))
for i in range(15):
    G = get_orbit_graph(i)
    pos = nx.spring_layout(G, seed=42)  
    plt.subplot(3, 5, i + 1)
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=600, font_size=10)
    plt.title(f"Orbit {i}", fontsize=10)
    plt.axis("off")

plt.suptitle("Visualisation of ORCA 4-node Motif Orbits", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


spark.stop()
