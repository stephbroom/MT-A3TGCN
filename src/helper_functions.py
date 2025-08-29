#helper functions 
from typing import Tuple, Union
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch

# Classificatioin metric helper
def compute_classification_metrics(y_true, y_pred_logits):
    import numpy as np
    from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score

    y_true = np.array(y_true)
    y_pred_logits = np.array(y_pred_logits)
    y_pred = y_pred_logits.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auroc = roc_auc_score(y_true, y_pred_logits[:, 1])
    except ValueError:
        auroc = float('nan')
    try:
        auprc = average_precision_score(y_true, y_pred_logits[:, 1])
    except ValueError:
        auprc = float('nan')
    return acc, f1, auroc, auprc

# Load loss history if it exists
def load_loss_history(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame(columns=["epoch", "train_total", "train_reg", "train_clf", "val_total", "val_reg", "val_clf"])

# Plot loss curves
def plot_loss_curves(df, save_path="training_logs/loss_curves.png"):
    plt.figure(figsize=(12, 6))
    plt.plot(df["epoch"], df["train_total"], label="Train Total")
    plt.plot(df["epoch"], df["train_reg"], label="Train Reg")
    plt.plot(df["epoch"], df["train_clf"], label="Train Clf")
    plt.plot(df["epoch"], df["val_total"], label="Val Total")
    plt.plot(df["epoch"], df["val_reg"], label="Val Reg")
    plt.plot(df["epoch"], df["val_clf"], label="Val Clf")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Components Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

# 6) Helper to fetch graph for given window
def get_edge_data(window_start, id_map, edges_ts, WINDOW_SIZE = 3):
    import torch
    import numpy as np 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ws = int(window_start)
    mask = (
        (edges_ts['timestamp'] >= ws) &
        (edges_ts['timestamp'] <  ws + WINDOW_SIZE)
    )
    sub = edges_ts.loc[mask]
    src = sub['src'].astype(int).map(id_map)
    dst = sub['dst'].astype(int).map(id_map)

    # Drop edges with unmapped nodes (outside of current window)
    valid_mask = src.notna() & dst.notna()
    src = src[valid_mask].astype(int)
    dst = dst[valid_mask].astype(int)

    edge_index  = torch.tensor([src.to_numpy(), dst.to_numpy()], dtype=torch.long).to(device)
    edge_weight = torch.tensor(sub['timestamp'].to_numpy()[valid_mask], dtype=torch.float).to(device)
    return edge_index, edge_weight

#helper to compute regression metrics for the regression head

def compute_regression_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor]
) -> Tuple[float, float, float]:
    """
    Compute common regression metrics between true and predicted orbit counts.

    Args:
        y_true: array-like of shape (N, O) true orbit counts.
        y_pred: array-like of shape (N, O) predicted orbit counts.

    Returns:
        mse: Mean Squared Error across all nodes and orbits.
        mae: Mean Absolute Error across all nodes and orbits.
        r2:  R^2 score across all nodes and orbits.
    """
    # convert tensors to numpy if needed
    if hasattr(y_true, "detach"):
        y_true = y_true.detach().cpu().numpy()
    if hasattr(y_pred, "detach"):
        y_pred = y_pred.detach().cpu().numpy()

    # flatten to (N⋅O,)
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    mse = mean_squared_error(y_true_flat, y_pred_flat)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2  = r2_score(y_true_flat, y_pred_flat)

    return mse, mae, r2
