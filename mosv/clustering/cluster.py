import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def find_optimal_K(
    X: np.ndarray,
    K_range: Tuple[int, int] = (2, 8),
    pca_components: int = 50,
    random_state: int = 42,
    n_init: int = 20,
) -> Tuple[int, Dict[int, float]]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(pca_components, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca.fit_transform(X_scaled)

    scores: Dict[int, float] = {}
    k_min, k_max = K_range
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(X_reduced)
        score = silhouette_score(X_reduced, labels)
        scores[k] = score
        print(f"  K={k}: silhouette = {score:.4f}")

    best_K = max(scores, key=scores.__getitem__)
    print(f"Optimal K={best_K} (silhouette={scores[best_K]:.4f})")
    return best_K, scores


def cluster_diff_vectors(
    X: np.ndarray,
    K: int,
    pca_components: int = 50,
    random_state: int = 42,
    n_init: int = 20,
) -> Tuple[np.ndarray, np.ndarray, PCA, StandardScaler]:
    """
    Run K-means on PCA-reduced diff vectors.

    Returns:
        steering_vectors: [K, hidden_size]  centroids back-projected to original space
        cluster_labels:   [N]               per-sample cluster assignment
        pca:              fitted PCA object
        scaler:           fitted StandardScaler
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(pca_components, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca.fit_transform(X_scaled)

    km = KMeans(n_clusters=K, random_state=random_state, n_init=n_init)
    cluster_labels = km.fit_predict(X_reduced)

    centroids_reduced = km.cluster_centers_
    centroids_scaled = pca.inverse_transform(centroids_reduced)
    steering_vectors = scaler.inverse_transform(centroids_scaled)

    return steering_vectors.astype(np.float32), cluster_labels, pca, scaler


def _fit_dbscan(
    X_reduced: np.ndarray,
    eps: float,
    min_samples: int,
) -> Tuple[np.ndarray, float, int, int]:
    """Fit DBSCAN and return (labels, silhouette, n_clusters, n_noise)."""
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_reduced)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    if n_clusters >= 2:
        mask = labels != -1
        sil = float(silhouette_score(X_reduced[mask], labels[mask])) if mask.sum() > n_clusters else float("nan")
    else:
        sil = float("nan")
    return labels, sil, n_clusters, n_noise


def _estimate_eps(X_reduced: np.ndarray, min_samples: int) -> float:
    """
    Estimate a reasonable eps using the k-nearest-neighbour distance heuristic:
    fit k=min_samples neighbours, take the knee of the sorted distance curve
    (approximated as the 90th percentile to avoid manual inspection).
    """
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X_reduced)
    distances, _ = nbrs.kneighbors(X_reduced)
    kth_distances = np.sort(distances[:, -1])
    return float(np.percentile(kth_distances, 90))


def compare_clustering(
    X: np.ndarray,
    pca_components: int = 50,
    K_range: Tuple[int, int] = (2, 8),
    dbscan_min_samples: int = 10,
    random_state: int = 42,
    n_init: int = 20,
    out_dir: Optional[str] = None,
) -> List[dict]:
    """
    Compare K-means (silhouette-optimal K) and DBSCAN on the same PCA-reduced data.

    Returns a list of result dicts, one per method, with keys:
        method, K, silhouette, cluster_sizes, n_noise, eps (DBSCAN only)

    Also prints a formatted comparison table and optionally saves to out_dir/cluster_comparison.json.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_components = min(pca_components, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca.fit_transform(X_scaled)

    results = []

    # --- K-means sweep ---
    print("\nK-means silhouette sweep:")
    km_scores: Dict[int, float] = {}
    for k in range(K_range[0], K_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(X_reduced)
        sil = float(silhouette_score(X_reduced, labels))
        km_scores[k] = sil
        sizes = {int(c): int((labels == c).sum()) for c in range(k)}
        print(f"  K={k}: silhouette={sil:.4f}  sizes={sizes}")
        results.append({
            "method": f"kmeans_K{k}",
            "K": k,
            "silhouette": round(sil, 4),
            "cluster_sizes": sizes,
            "n_noise": 0,
        })

    best_K = max(km_scores, key=km_scores.__getitem__)
    print(f"  → Silhouette-optimal K = {best_K}")

    # --- DBSCAN with auto-estimated eps ---
    print("\nDBSCAN:")
    eps_auto = _estimate_eps(X_reduced, dbscan_min_samples)
    print(f"  Auto-estimated eps = {eps_auto:.4f}  (min_samples={dbscan_min_samples})")

    for eps in sorted({round(eps_auto * f, 4) for f in [0.5, 0.75, 1.0, 1.25, 1.5]}):
        labels, sil, n_clusters, n_noise = _fit_dbscan(X_reduced, eps, dbscan_min_samples)
        sizes = {int(c): int((labels == c).sum()) for c in set(labels) if c != -1}
        sil_str = f"{sil:.4f}" if not np.isnan(sil) else "N/A"
        print(f"  eps={eps:.4f}: clusters={n_clusters}  noise={n_noise}  silhouette={sil_str}  sizes={sizes}")
        results.append({
            "method": f"dbscan_eps{eps:.4f}",
            "K": n_clusters,
            "silhouette": round(sil, 4) if not np.isnan(sil) else None,
            "cluster_sizes": sizes,
            "n_noise": n_noise,
            "eps": eps,
        })

    # --- Summary table ---
    print("\n" + "=" * 72)
    print(f"{'Method':<25} {'K':>4} {'Silhouette':>12} {'Noise':>7} {'Sizes'}")
    print("-" * 72)
    for r in results:
        sil_str = f"{r['silhouette']:.4f}" if r["silhouette"] is not None else "  N/A"
        print(f"  {r['method']:<23} {r['K']:>4} {sil_str:>12} {r['n_noise']:>7}  {r['cluster_sizes']}")
    print("=" * 72)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "cluster_comparison.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved comparison to {path}")

    return results


def cluster_diff_vectors_dbscan(
    X: np.ndarray,
    pca_components: int = 50,
    min_samples: int = 10,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Cluster using DBSCAN with auto-estimated eps.
    Noise points (-1) are reassigned to the nearest cluster centroid.

    Returns:
        steering_vectors: [K, hidden_size]  mean diff vectors per cluster
        cluster_labels:   [N]               per-sample assignment (no -1s)
        K:                number of clusters found
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_components = min(pca_components, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca.fit_transform(X_scaled)

    eps = _estimate_eps(X_reduced, min_samples)
    print(f"  DBSCAN auto-eps={eps:.4f}, min_samples={min_samples}")
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_reduced)

    cluster_ids = sorted(c for c in set(labels) if c != -1)
    K = len(cluster_ids)
    if K == 0:
        raise ValueError("DBSCAN found no clusters. Try reducing min_samples.")

    n_noise = int((labels == -1).sum())

    # Compute centroids in original space
    centroids = np.array([X[labels == c].mean(axis=0) for c in cluster_ids])

    # Reassign noise points to nearest centroid
    if n_noise > 0:
        noise_idx = np.where(labels == -1)[0]
        noise_X = X[noise_idx]                              # (n_noise, hidden)
        dists = np.linalg.norm(                             # (n_noise, K)
            noise_X[:, None, :] - centroids[None, :, :], axis=2
        )
        nearest = np.argmin(dists, axis=1)
        labels = labels.copy()
        for idx, c_pos in zip(noise_idx, nearest):
            labels[idx] = cluster_ids[c_pos]

    # Re-index labels to 0..K-1 and recompute centroids
    id_map = {c: i for i, c in enumerate(cluster_ids)}
    final_labels = np.array([id_map[l] for l in labels])
    steering_vectors = np.array([X[final_labels == i].mean(axis=0) for i in range(K)])

    sizes = {i: int((final_labels == i).sum()) for i in range(K)}
    print(f"  DBSCAN: K={K}, {n_noise} noise points reassigned, sizes={sizes}")

    return steering_vectors.astype(np.float32), final_labels, K


def print_cluster_summary(cluster_labels: np.ndarray, questions: List[str]) -> None:
    K = int(cluster_labels.max()) + 1
    for k in range(K):
        mask = cluster_labels == k
        print(f"\nCluster {k} ({mask.sum()} samples):")
        indices = np.where(mask)[0][:10]
        for i in indices:
            print(f"  [{i}] {questions[i][:120]}")


def save_clustering(
    steering_vectors: np.ndarray,
    cluster_labels: np.ndarray,
    layer_scores: Dict[int, float],
    best_layer: int,
    K: int,
    silhouette_scores: Dict[int, float],
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "steering_vectors.npy"), steering_vectors)
    np.save(os.path.join(out_dir, "cluster_labels.npy"), cluster_labels)
    meta = {
        "layer_probe_scores": {str(k): float(v) for k, v in layer_scores.items()},
        "best_layer_idx": int(best_layer),
        "K": int(K),
        "silhouette_scores": {str(k): float(v) for k, v in silhouette_scores.items()},
    }
    with open(os.path.join(out_dir, "cluster_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved clustering to {out_dir}: K={K}, steering_vectors {steering_vectors.shape}")


def load_clustering(out_dir: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    steering_vectors = np.load(os.path.join(out_dir, "steering_vectors.npy"))
    cluster_labels = np.load(os.path.join(out_dir, "cluster_labels.npy"))
    with open(os.path.join(out_dir, "cluster_meta.json")) as f:
        meta = json.load(f)
    return steering_vectors, cluster_labels, meta
