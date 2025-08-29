import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------
# Script: sc0_window_size_selection.py
# Purpose: Compare candidate sliding-window widths
#          by measuring per-window node and edge counts
# Usage: Run before motif extraction to choose window_size
# Assumes:
#  - data/txs_edgelist.csv has [txId1, txId2]
#  - data/txs_features.csv has [txId, Time step]
# ------------------------------------------------------

#Load edge list and transaction timestamps
edges = pd.read_csv("data/txs_edgelist.csv", usecols=["txId1","txId2"])  #edges 
times = pd.read_csv("data/txs_features.csv", usecols=["txId","Time step"])  # timestamps

# assign timestamp to each edge by mergine with its destination txId2
times = times.rename(columns={"txId":"txId2","Time step":"timestamp"})
times.dropna(inplace=True)
edges_ts = edges.merge(times, on="txId2", how="inner")

# grid of window sizes (in time steps) to evaluate
candidate_windows = [1, 2, 3, 4, 5]
results = []

min_t = edges_ts['timestamp'].min()
max_t = edges_ts['timestamp'].max()
print(f"Time steps range from {min_t} to {max_t}")

#partition non-overlapping windows and measure counts
for w in candidate_windows:
    step = w  # non-overlapping windows
    starts = np.arange(min_t, max_t + 1, step)
    node_counts = []
    edge_counts = []
    for start in starts:
        end = start + w
        mask = (edges_ts['timestamp'] >= start) & (edges_ts['timestamp'] < end)
        sub = edges_ts.loc[mask]
        edge_counts.append(len(sub))
        #count unique nodes in  window
        nodes = pd.concat([sub['txId1'], sub['txId2']]).unique()
        node_counts.append(len(nodes))
    results.append({
        'window_size': w,
        'n_windows': len(starts),
        'avg_edges_per_window': np.mean(edge_counts),
        'std_edges_per_window': np.std(edge_counts),
        'avg_nodes_per_window': np.mean(node_counts),
        'std_nodes_per_window': np.std(node_counts)
    })

#Summarise results, print
df_res = pd.DataFrame(results)
print(df_res)

#plots for visual aid
fig, axes = plt.subplots(2, 1, figsize=(8, 10))

ax = axes[0]
ax.errorbar(df_res['window_size'], df_res['avg_nodes_per_window'],
            yerr=df_res['std_nodes_per_window'], marker='o', capsize=5)
ax.set_title('Nodes per Window vs Window Size')
ax.set_xlabel('Window Size (time steps)')
ax.set_ylabel('Average Unique Nodes')

ax = axes[1]
ax.errorbar(df_res['window_size'], df_res['avg_edges_per_window'],
            yerr=df_res['std_edges_per_window'], marker='o', capsize=5)
ax.set_title('Edges per Window vs Window Size')
ax.set_xlabel('Window Size (time steps)')
ax.set_ylabel('Average Edge Count')

plt.tight_layout()
plt.show()

#save
os.makedirs('data/eda', exist_ok=True)
df_res.to_csv('data/eda/window_size_summary.csv', index=False)

print("Wrote window_size_summary.csv to data/eda/")
