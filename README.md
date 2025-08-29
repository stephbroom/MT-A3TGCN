# MT-A3TGCN
Graph-Based Modelling of Financial Transactions: Leveraging Temporal Structure and Motif Patterns for Illicit Behaviour Detection

Higher–order transaction patterns (graphlet orbits) correlate with laundering behaviours such as mixing and layering. This repo trains a temporal GNN on the transaction-level graph to (i) regress motif counts and (ii) classify illicit vs licit in a multi-task, semi-supervised setup. A companion classifier-only baseline is provided. We also create augmented datasets to stress-test robustness by degrading covariate signal while keeping motif regression easy.
