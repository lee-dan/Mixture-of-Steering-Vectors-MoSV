from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def probe_layer(
    diff_vectors: np.ndarray,
    layer_idx: int,
    n_folds: int = 5,
) -> float:
    """Linear probe on diff vectors at one layer; returns mean cross-val accuracy as a measure of geometric structure."""
    X = diff_vectors[:, layer_idx, :]
    N = X.shape[0]
    y = np.array([1] * (N // 2) + [0] * (N - N // 2))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    scores = cross_val_score(clf, X_scaled, y, cv=n_folds, scoring="accuracy")
    return float(scores.mean())


def select_best_layer(
    diff_vectors: np.ndarray,
    layer_indices: List[int],
    n_folds: int = 5,
) -> Tuple[int, Dict[int, float]]:
    """Probe all layers; returns (best_layer_pos, {layer: accuracy})."""
    layer_scores: Dict[int, float] = {}
    for pos, layer_idx in enumerate(layer_indices):
        score = probe_layer(diff_vectors, pos, n_folds=n_folds)
        layer_scores[layer_idx] = score
        print(f"  Layer {layer_idx:2d}: probe accuracy = {score:.4f}")

    best_layer = max(layer_scores, key=layer_scores.__getitem__)
    best_pos = layer_indices.index(best_layer)
    print(f"Best layer: {best_layer} (accuracy={layer_scores[best_layer]:.4f})")
    return best_pos, layer_scores


def get_layer_vectors(diff_vectors: np.ndarray, layer_pos: int) -> np.ndarray:
    return diff_vectors[:, layer_pos, :]
