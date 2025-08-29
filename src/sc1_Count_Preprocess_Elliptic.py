import subprocess
import pandas as pd
import numpy as np
from pyspark.sql import Window
from pyspark.sql.functions import row_number, col
from pyspark.sql import SparkSession, functions as F
from graphframes import GraphFrame
import os
import warnings
import logging

# ------------------------------------------------------
# Script: sc1_Count_Preprocess_Elliptic.py
# Purpose: Extract per-node motif counts for each sliding window
#          using ORCA, preserving the node axis.
# Assumes:
#  - data/txs_edgelist.csv, data/txs_features.csv, data/txs_classes.csv exist
# ------------------------------------------------------

#ignore warnings
warnings.filterwarnings('ignore')
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)
os.environ["PYSPARK_SUBMIT_ARGS"] = "--conf spark.ui.showConsoleProgress=false pyspark-shell"

#spark session
spark = (
    SparkSession.builder
        .appName("AllMotifs_GraphFrames")
        .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12")
        .config("spark.executor.memory","8g")
        .config("spark.driver.memory","4g")
        .config("spark.sql.shuffle.partitions","200")
        .config("spark.driver.port", "6066")
        .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR") # supresses window func warnigns

#raw data load
edges = (spark.read.option("header", True).csv("data/txs_edgelist.csv")
           .selectExpr("cast(txId1 as string) as txId1",
                       "cast(txId2 as string) as txId2"))
ts = (spark.read.option("header", True).csv("data/txs_features.csv")
         .selectExpr("cast(txId as string) as txId",
                     "cast(`Time step` as int) as timestamp").cache())
classes = (spark.read.option("header", True).csv("data/txs_classes.csv")
              .selectExpr("cast(txId as string) as txId",
                          "cast(class as int) as class").cache())

#join edges by timestamp
edges_ts = (edges
    .join(ts,      edges.txId2 == ts.txId,      "left")
    .join(classes, edges.txId2 == classes.txId, "left")
    .select("txId1","txId2","timestamp","class")
)
#filter out class==3 if desired for edges
#edges_filtered = edges_ts.filter(F.col("class") != 3).cache()
#edges_filtered = edges_filtered.repartition(F.col("timestamp"))

#or alternatively
edges_filtered = edges_ts

#build graphframe
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
min_t, max_t = edges_filtered.agg(F.min("timestamp"), F.max("timestamp")).first()
window_size = 3    # steps per window
step_size   = 2    # slide increment
windows = list(range(min_t, max_t - window_size + 1, step_size))

# 5) Create integer mapping for nodes and save
verts = g.vertices.withColumnRenamed("id", "txId")
window_spec = Window.orderBy("txId")
verts_int = verts.withColumn("int_id", row_number().over(window_spec) - 1)
verts_int.write.mode("overwrite").parquet("data/verts_int.parquet")

#Loop over all  windows, run ORCA, preserving node axis
dfs_orb = []
os.makedirs("data/motif_counts", exist_ok=True)
N = verts_int.count()
for start in windows:
    end = start + window_size
    sub = edges_filtered.filter((col("timestamp") >= start) & (col("timestamp") < end))

    # remap to integer IDs
    edges_int = (sub.join(verts_int, sub.txId1 == verts_int.txId)
                     .withColumnRenamed("int_id", "src_int").drop("txId")
                     .join(verts_int, sub.txId2 == verts_int.txId)
                     .withColumnRenamed("int_id", "dst_int").drop("txId")
                     .select("src_int","dst_int"))

    M = edges_int.count()
    local_edges = edges_int.rdd.map(lambda r: f"{r.src_int} {r.dst_int}").collect()
    with open("graph_orca.txt","w") as f:
        f.write(f"{N} {M}\n")
        f.write("\n".join(local_edges))

    # run ORCA, output to per-window file
    out_fname = f"data/motif_counts/motifs_{start}.txt"
    subprocess.run(["orca","node","4","graph_orca.txt", out_fname], check=True)

    # load node-level counts and tag window
    df_orb = pd.read_csv(out_fname, sep=" ", header=None)
    orbit_cols = [f"orbit_{i}" for i in range(df_orb.shape[1])]
    df_orb.columns = orbit_cols
    df_orb["int_id"] = np.arange(len(df_orb))
    df_orb["window_start"] = start
    dfs_orb.append(df_orb)

# combining and saving counts
df_orb_window = pd.concat(dfs_orb, ignore_index=True)
df_orb_window.to_csv("data/all_node_orbit_counts.csv", index=False)

# write Parquet for fast downstream loading
df_orb_spark = spark.createDataFrame(df_orb_window)
df_orb_spark.write.mode("overwrite").parquet("data/all_node_orbit_counts.parquet")

#end spark
spark.stop()
