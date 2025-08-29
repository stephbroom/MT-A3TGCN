## Motif detection 

# Suppress Python warnings
import warnings
warnings.filterwarnings("ignore")

# Suppress pyspark/py4j INFO/WARN logs
import logging
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)

# Optional: prevent Ivy and Spark console progress messages
import os
os.environ["PYSPARK_SUBMIT_ARGS"] = "--conf spark.ui.showConsoleProgress=false pyspark-shell"

#Imports
from pyspark.sql import SparkSession, functions as F, types as T
from graphframes import GraphFrame
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import chi2_contingency, fisher_exact
from pyspark.sql import functions as F
import pandas as pd 
import numpy as np 

#Initialising spark session 
spark = (
    SparkSession.builder
        .appName("AllMotifs_GraphFrames")
        .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12")
        .config("spark.executor.memory",       "8g")
        .config("spark.driver.memory",         "4g")
        .config("spark.sql.shuffle.partitions","200")
        .getOrCreate()
)

#Loading in edges and filtering out unlabelled data
edges = (spark.read.option("header", True).csv("data/txs_edgelist.csv")
           .selectExpr("cast(txId1 as string) as txId1",
                       "cast(txId2 as string) as txId2"))
ts = (spark.read.option("header", True).csv("data/txs_features.csv")
         .selectExpr("cast(txId   as string) as txId",
                     "cast(`Time step` as int)    as timestamp")
         .cache())
classes = (spark.read.option("header", True).csv("data/txs_classes.csv")
              .selectExpr("cast(txId as string) as txId",
                          "cast(class as int)     as class")
              .cache())

edges_ts = (edges
    .join(ts,      edges.txId2 == ts.txId,      "left")
    .join(classes, edges.txId2 == classes.txId, "left")
    .select("txId1","txId2","timestamp","class")
)
edges_filtered = edges_ts.filter(F.col("class") != 3).cache()
edges_filtered = edges_filtered.repartition(F.col("timestamp"))

#bulding graphframes graph 
# 3) Build GraphFrame
vertices = (edges_filtered.selectExpr("txId1 AS id")
                           .union(edges_filtered.selectExpr("txId2 AS id"))
                           .distinct())
g = GraphFrame(
    vertices,
    edges_filtered.selectExpr("txId1 AS src",
                              "txId2 AS dst",
                              "timestamp",
                              "class AS class_label")
)

#setting param for max timesteps between first and last node
delta = 1

#dict of motifs

MOTIF_SIGNATURES = {
 "A->B|A->B|A->B":"M1",  "A->B|A->B|A->C":"M2",  "A->B|A->B|B->A":"M3",
 "A->B|A->B|B->C":"M4",  "A->B|A->B|C->A":"M5",  "A->B|A->B|C->B":"M6",
 "A->B|A->C|A->B":"M7","A->B|A->C|A->C":"M8","A->B|A->C|B->A":"M9",
 "A->B|A->C|B->C":"M10","A->B|A->C|C->A":"M11","A->B|A->C|C->B":"M12",
 "A->B|B->A|A->B":"M13","A->B|B->A|A->C":"M14","A->B|B->A|B->A":"M15",
 "A->B|B->A|B->C":"M16","A->B|B->A|C->A":"M17","A->B|B->A|C->B":"M18",
 "A->B|B->C|A->B":"M19","A->B|B->C|A->C":"M20","A->B|B->C|B->A":"M21",
 "A->B|B->C|B->C":"M22","A->B|B->C|C->A":"M23","A->B|B->C|C->B":"M24",
 "A->B|C->A|A->B":"M25","A->B|C->A|A->C":"M26","A->B|C->A|B->A":"M27",
 "A->B|C->A|B->C":"M28","A->B|C->A|C->A":"M29","A->B|C->A|C->B":"M30",
 "A->B|C->B|A->B":"M31","A->B|C->B|A->C":"M32","A->B|C->B|B->A":"M33",
 "A->B|C->B|B->C":"M34","A->B|C->B|C->A":"M35","A->B|C->B|C->B":"M36"
}

#Plotting the set of 36 motifs for a visual aid
fig, axes = plt.subplots(6, 6, figsize=(10, 10))
axes = axes.flatten()

for ax, (sig, mid) in zip(axes, MOTIF_SIGNATURES.items()):
    edges_list = [tuple(e.split("->")) for e in sig.split("|")]
    Gm = nx.DiGraph()
    Gm.add_edges_from(edges_list)
    pos = nx.spring_layout(Gm, seed=42)
    nx.draw(Gm, pos, ax=ax, with_labels=True)
    ax.set_title(mid)
    ax.axis('off')

fig.suptitle("All 36 Three-Edge Motifs", y=1.02)
plt.tight_layout()
plt.show()


#GraphFrame DSL patterns
motif_patterns = {}
for sig, mid in MOTIF_SIGNATURES.items():
    parts = sig.split("|")
    # map A→a, B→b, C→c
    dsl = []
    for i, p in enumerate(parts, 1):
        u,v = p.split("->")
        dsl.append(f"({u.lower()})-[e{i}]->({v.lower()})")
    motif_patterns[mid] = "; ".join(dsl)

#Looping over patterns, count each
counts = []
for mid, pattern in motif_patterns.items():
    #find all matches for this pattern
    df = g.find(pattern).filter(
        (F.col("e1.timestamp") <= F.col("e2.timestamp")) &
        (F.col("e2.timestamp") <= F.col("e3.timestamp")) &
        (F.col("e3.timestamp") <= F.col("e1.timestamp") + delta)
    )
    #counts for each class 
    for cls in (1, 2):
        c = df.filter(F.col("e3.class_label") == cls).count()
        counts.append((mid, cls, c))

#summary DF with columns motif_id, class_label, count (for plotting)
summary = (
    spark.createDataFrame(counts, schema=["motif_id", "class_label", "count"])
         .orderBy("motif_id", "class_label")
)

#pulling into pandas for plotting
pdf = (
    summary
      .filter(F.col("class_label").isin(1, 2))
      .toPandas()
)
pivot = pdf.pivot(index="motif_id", columns="class_label", values="count").fillna(0)
pivot.columns = ['Illicit', 'Licit']

fig2, ax2 = plt.subplots(figsize=(5, 3))
pivot.plot(kind='bar', ax=ax2)
ax2.set_title("Motif Counts: Illicit vs Licit")
ax2.set_xlabel("Motif ID")
ax2.set_ylabel("Count")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#Table of ratios of licit to illicit activity

#pivot class label column
ratio_df = (
    summary
      .groupBy("motif_id")
      .pivot("class_label", [1, 2])
      .agg(F.first("count"))
      .withColumnRenamed("1", "illicit_count")
      .withColumnRenamed("2", "licit_count")
)

#full nans with 0 
ratio_df = ratio_df.na.fill(0, subset=["illicit_count", "licit_count"])

#computing ratio ( only for motifs which appear)
ratio_df = ratio_df.withColumn(
    "illicit_to_licit_ratio",
    F.when(F.col("licit_count") > 0,
           F.col("illicit_count") / F.col("licit_count")
    ).otherwise(None)
)

#Showing top 5 motifs, sorted by ratio descending
ratio_df.orderBy("illicit_to_licit_ratio", ascending=False).show(5, truncate=False)

###### HYPOTHESIS TESTING ######

#Pivot into a motif X class table and drop rows not seen
cont = (
    pdf
      .pivot(index="motif_id", columns="class_label", values="count")
      .fillna(0)
)
cont.columns = ["illicit", "licit"]

#only motifs with at least one occurrence
cont = cont.loc[(cont["illicit"] + cont["licit"]) > 0]

#Global chi‐test of independence
chi2, pval, dof, expected = chi2_contingency(cont.values)
print(f"Global χ² = {chi2:.2f}, p‐value = {pval:.3g}, dof = {dof}")

#Per‐motif Fisher’s exact tests (only for observed motifs)
total_il = cont["illicit"].sum()
total_li = cont["licit"].sum()

results = []
for motif_id, row in cont.iterrows():
    a = int(row["illicit"])
    b = int(row["licit"])
    if a + b == 0:
        continue
    c = total_il - a
    d = total_li - b
    oddsratio, p = fisher_exact([[a, b], [c, d]])
    results.append((motif_id, a, b, oddsratio, p))

fish_df = pd.DataFrame(
    results,
    columns=["motif_id","illicit","licit","odds_ratio","p_value"]
).sort_values("p_value")

#Show the top 5 motifs most weighted towards illicit flows
print("\nTop 5 motifs by Fisher p‐value:")
print(fish_df.head(5).to_string(index=False))

spark.stop()
