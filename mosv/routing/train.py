import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from mosv.routing.model import MoSVRouter


def build_router_dataset(
    prompt_vectors: np.ndarray,
    cluster_labels: np.ndarray,
    best_layer_pos: int,
) -> TensorDataset:
    """
    prompt_vectors: [N, n_layers, hidden_size]
    cluster_labels: [N]
    Returns a TensorDataset of (hidden_state, label) pairs.
    """
    X = prompt_vectors[:, best_layer_pos, :]
    X_tensor = torch.from_numpy(X.astype(np.float32))
    y_tensor = torch.from_numpy(cluster_labels.astype(np.int64))
    return TensorDataset(X_tensor, y_tensor)


def train_router(
    router: MoSVRouter,
    dataset: TensorDataset,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    load_balance_coef: float = 0.01,
    batch_size: int = 32,
    val_ratio: float = 0.15,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, List[float]]:
    router = router.to(device)

    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.Adam(router.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
    }

    for epoch in range(epochs):
        router.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits = router.route_logits(X_batch)
            ce_loss = F.cross_entropy(logits, y_batch)

            weights = F.softmax(logits, dim=-1)
            load_loss = weights.mean(0).var()

            loss = ce_loss + load_balance_coef * load_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += ce_loss.item() * len(X_batch)
            train_correct += (logits.argmax(-1) == y_batch).sum().item()
            train_total += len(X_batch)

        scheduler.step()

        router.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = router.route_logits(X_batch)
                val_loss += F.cross_entropy(logits, y_batch).item() * len(X_batch)
                val_correct += (logits.argmax(-1) == y_batch).sum().item()
                val_total += len(X_batch)

        history["train_loss"].append(train_loss / train_total)
        history["val_loss"].append(val_loss / val_total)
        history["train_acc"].append(train_correct / train_total)
        history["val_acc"].append(val_correct / val_total)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:3d}/{epochs} "
                f"train_loss={history['train_loss'][-1]:.4f} "
                f"train_acc={history['train_acc'][-1]:.4f} "
                f"val_acc={history['val_acc'][-1]:.4f}"
            )

    return history


def save_router(router: MoSVRouter, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    torch.save(router.state_dict(), os.path.join(out_dir, "router.pt"))
    meta = {"K": router.K, "top_k": router.top_k, "d_model": router.fc1.in_features}
    with open(os.path.join(out_dir, "router_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved router to {out_dir}")


def load_router(out_dir: str, device: torch.device) -> MoSVRouter:
    with open(os.path.join(out_dir, "router_meta.json")) as f:
        meta = json.load(f)
    router = MoSVRouter(d_model=meta["d_model"], K=meta["K"], top_k=meta["top_k"])
    router.load_state_dict(torch.load(os.path.join(out_dir, "router.pt"), map_location=device))
    router.eval()
    return router.to(device)
