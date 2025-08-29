import pandas as pd
import numpy as np
import glob, os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition   import PCA
import joblib

import matplotlib.pyplot as plt

# ------------------------------------------------------
# Script: sc4_PCA.py
# Purpose: Build BTC flow meta‑features, load per‑window motif counts
#          from the aggregated CSV, PCA‑compress residual static features,
#          and assemble the final dynamic per‑window feature matrix
# Assumes:
#  - data/txs_features.csv       has [txId, Time step, static features…]
#  - data/verts_int.parquet      has [txId, int_id]
#  - data/all_node_orbit_counts.csv has [orbit_0…orbit_N, int_id, window_start]
# ------------------------------------------------------

#load node mappings
verts_map = (
    pd.read_parquet("data/verts_int.parquet")
      .astype({"txId": str})
)  # columns: txId, int_id

#load features and join int id
df_feats = pd.read_csv("data/txs_features.csv", dtype={"txId": str})
df_feats.dropna(inplace=True)
df_static = (
    df_feats
      .merge(verts_map, on="txId", how="inner")
      .set_index("int_id", drop=True)
      .drop(columns=["txId", "Time step"])
)

#computing meta features
df_meta = pd.DataFrame(index=df_static.index)
df_meta["in_BTC_flow"]  = df_static["in_BTC_total"].astype(float)
df_meta["out_BTC_flow"] = df_static["out_BTC_total"].astype(float)
df_meta["net_flow"]     = df_meta["out_BTC_flow"] - df_meta["in_BTC_flow"]
df_meta["flow_ratio"]   = df_meta["out_BTC_flow"] / (df_meta["in_BTC_flow"] + 1e-6)
df_meta["in_spread"]    = df_static["in_BTC_max"]  - df_static["in_BTC_min"]
df_meta["out_spread"]   = df_static["out_BTC_max"] - df_static["out_BTC_min"]

for c in ["in_BTC_flow","out_BTC_flow","net_flow"]:
    df_meta[f"log_{c}"] = np.sign(df_meta[c]) * np.log1p(np.abs(df_meta[c]))

#prepare features for PCA
to_drop = [
    # raw BTC totals & mins/maxes 
    "in_BTC_min","in_BTC_max","in_BTC_mean","in_BTC_median","in_BTC_total",
    "out_BTC_min","out_BTC_max","out_BTC_mean","out_BTC_median","out_BTC_total",
    # the flow features and their logs
    "in_BTC_flow","out_BTC_flow","net_flow","flow_ratio",
    "in_spread","out_spread","log_in_BTC_flow","log_out_BTC_flow","log_net_flow",
    # any other transactional‑count features 
    "size","num_input_addresses","num_output_addresses",
    "total_BTC","in_txs_degree","out_txs_degree", "fees"
]
df_resid = df_static.drop(columns=to_drop, errors="ignore").fillna(0)

print("Residual static shape:", df_resid.shape)

#standardise and perform PCA
scaler = StandardScaler()
X_std  = scaler.fit_transform(df_resid)
pca    = PCA()
X_pca  = pca.fit_transform(X_std)

# how many components for 95% explained variance
cum_var = np.cumsum(pca.explained_variance_ratio_)
d95     = np.argmax(cum_var >= 0.95) + 1
print(f"→ {d95} PCs capture ≥95% variance ({cum_var[d95-1]:.3f}).")

# inspect top loadings 
loadings = pd.DataFrame(
    pca.components_[:d95],
    columns=df_resid.columns,
    index=[f"PC{i+1}" for i in range(d95)]
)
for pc in loadings.index:
    top = loadings.loc[pc].abs().nlargest(10)
    print(f"\nTop 10 loadings for {pc}:\n{top}")

# build a PCA DataFrame 
df_pca = pd.DataFrame(
    X_pca[:, :d95],
    index=df_resid.index,
    columns=[f"PC{i+1}" for i in range(d95)]
)

# full orbit counts
df_orb = pd.read_csv("data/all_node_orbit_counts.csv")

orbit_cols = [c for c in df_orb.columns if c.startswith("orbit_")]

# ensure numeric and non-negative 
df_orb[orbit_cols] = df_orb[orbit_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
df_orb[orbit_cols] = df_orb[orbit_cols].clip(lower=0.0)

# log1p to reduce heavy tails
df_orb[orbit_cols] = np.log1p(df_orb[orbit_cols].astype(float))


#assemble the final per‑window feature matrix — only including rows wheere all features present
X_final = (
    df_orb
    .merge(df_meta, on="int_id", how="inner")
    .merge(df_pca,  on="int_id", how="inner")
)

print("Final feature matrix shape:", X_final.shape)

#save to csv for training
X_final.to_csv("data/selected_features_windows.csv", index=False)

# Create scalers directory if it doesn't exist
os.makedirs("scalers", exist_ok=True)

orbit_cols = [c for c in X_final.columns if c.startswith("orbit_")]
excluded = ["int_id", "window_start"] + orbit_cols
meta_cols = [c for c in X_final.columns if c not in excluded]

# apply scaler to metadata
meta_scaler = StandardScaler()
X_final[meta_cols] = meta_scaler.fit_transform(X_final[meta_cols])
joblib.dump(meta_scaler, "scalers/meta_scaler.pkl")

#apply scaler to orbit counts
orbit_scaler = StandardScaler()
X_final[orbit_cols] = orbit_scaler.fit_transform(X_final[orbit_cols])
joblib.dump(orbit_scaler, "scalers/orbit_scaler.pkl")

# Saving final scaled features
X_final.to_csv("data/selected_features_windows_scaled.csv", index=False)
print("Saved scaled feature matrix to data/selected_features_windows_scaled.csv")
