# MT-A3TGCN
Graph-Based Modelling of Financial Transactions: Leveraging Temporal Structure and Motif Patterns for Illicit Behaviour Detection

Higher–order transaction patterns (graphlet orbits) correlate with laundering behaviours such as mixing and layering. This repo trains a temporal GNN on the transaction-level graph to (i) regress motif counts and (ii) classify illicit vs licit in a multi-task, semi-supervised setup. A companion classifier-only baseline is provided. We also create augmented datasets to stress-test robustness by degrading covariate signal while keeping motif regression easy.

All data required to run the code contained in this repo is accessible via: https://github.com/git-disl/EllipticPlusPlus/tree/main. For the scripts to work without having to repoint the functions, ensure downloaded data is saved to a folder called data/.

All code required to process the data and regenerate the augmented data splits is also included in the folders notebooks/ and src/.
