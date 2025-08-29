import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
import seaborn as sns

# ------------------------------------------------------
# Script: sc2_sliding_window_tuning.py
# Purpose: EDA of per-node motif counts across sliding windows
# Assumes:
#  - data/all_node_orbit_counts.parquet contains columns:
#      int_id, orbit_0...orbit_N, window_start
# ------------------------------------------------------

#spark
spark = SparkSession.builder.appName("NodeLevelMotifEDA").getOrCreate()

#load orbit counts into pandas
df_node = (
    spark.read.parquet("data/all_node_orbit_counts.parquet")
         .toPandas()
)

#Identify orbit columns
orbit_cols = [c for c in df_node.columns if c.startswith('orbit_')]

#descriptive statistics for each orbit
stats = df_node[orbit_cols].describe().T
print("Descriptive statistics per orbit:\n", stats)

# frac of nodes with non-zero count per orbit
nonzero_frac = (df_node[orbit_cols] > 0).mean().sort_values(ascending=False)
print("\nFraction of nodes with >0 count for each orbit:\n", nonzero_frac)

#top 10 orbits by non-zero participation
top10 = nonzero_frac.head(10)
plt.figure(figsize=(8, 4))
top10.plot(kind='bar')
plt.xlabel('Orbit ID')
plt.ylabel('Fraction of nodes with >0 count')
plt.title('Non-zero Participation Fraction by Orbit')
plt.tight_layout()
plt.show()

#Avg orbit count per window
avg_per_window = df_node.groupby('window_start')[orbit_cols].mean()
plt.figure(figsize=(10, 6))
for orbit in orbit_cols[:5]:
    plt.plot(avg_per_window.index, avg_per_window[orbit], label=orbit)
plt.xlabel('Window Start (time step)')
plt.ylabel('Average Orbit Count')
plt.title('Average Node-Level Orbit Counts Over Time (First 5 Orbits)')
plt.legend()
plt.tight_layout()
plt.show()

#node count per window
active_nodes = df_node.groupby('window_start')['int_id'].nunique()
plt.figure(figsize=(6, 4))
active_nodes.plot(kind='bar')
plt.xlabel('Window Start (time step)')
plt.ylabel('Number of Active Nodes')
plt.title('Active Nodes per Window')
plt.tight_layout()
plt.show()

# stop 
spark.stop()
