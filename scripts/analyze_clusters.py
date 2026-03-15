"""analyze_clusters.py — Per-cluster coherence and silhouette scores for each sweep_K variant."""
import argparse, os, sys, json
import numpy as np
import yaml
from sklearn.metrics import silhouette_score, silhouette_samples

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cosine_sim_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-8)
    Xn = X / norms
    return Xn @ Xn.T


def cluster_coherence(X, labels, K, max_per_cluster=500, seed=42):
    rng = np.random.default_rng(seed)
    results = {}
    for k in range(K):
        idx = np.where(labels == k)[0]
        sample = rng.choice(idx, min(max_per_cluster, len(idx)), replace=False)
        Xk = X[sample]
        if len(Xk) < 2:
            results[k] = {"n": len(idx), "mean_cosine_sim": float("nan")}
            continue
        sim = cosine_sim_matrix(Xk)
        upper = sim[np.triu_indices(len(Xk), k=1)]
        results[k] = {"n": int(len(idx)), "mean_cosine_sim": float(upper.mean())}
    return results


def compute_silhouette(X, labels, K, max_samples=5000, seed=42):
    """Overall + per-cluster silhouette, subsampled for speed."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    if n > max_samples:
        idx = rng.choice(n, max_samples, replace=False)
        X_s, labels_s = X[idx], labels[idx]
    else:
        X_s, labels_s = X, labels

    if len(set(labels_s)) < 2:
        return float("nan"), {}

    overall = float(silhouette_score(X_s, labels_s, metric="cosine"))
    per_sample = silhouette_samples(X_s, labels_s, metric="cosine")
    per_cluster = {}
    for k in range(K):
        mask = labels_s == k
        per_cluster[k] = float(per_sample[mask].mean()) if mask.sum() > 0 else float("nan")
    return overall, per_cluster


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/defan.yaml")
    parser.add_argument("--k_tags", nargs="+", default=["K2", "K4", "K6", "K10", "K15", "K20"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from mosv.activation.extract import load_activations
    from mosv.clustering.cluster import load_clustering

    act_dir = cfg["paths"]["activations_dir"]
    out_dir = cfg["paths"]["outputs_dir"]
    layers  = cfg["activation"]["layers_to_probe"]
    os.makedirs(out_dir, exist_ok=True)

    diff_vectors, _ = load_activations(act_dir)
    n_total = diff_vectors.shape[0]

    print(f"\n{'K':<6} {'C':<4} {'N':>6} {'%':>6}  {'CosSim':>8}  {'Silhouette':>10}  {'Interp'}")
    print("-" * 70)

    all_results = {}
    for tag in args.k_tags:
        sub_dir = os.path.join(act_dir, f"sweep_{tag}")
        if not os.path.exists(sub_dir):
            print(f"Skipping {tag} — not found")
            continue

        _, labels, meta = load_clustering(sub_dir)
        best_layer_pos = layers.index(meta["best_layer_idx"])
        X = diff_vectors[:, best_layer_pos, :]
        K = meta["K"]

        coh = cluster_coherence(X, labels, K)
        sil_overall, sil_per_cluster = compute_silhouette(X, labels, K)

        tag_result = {"silhouette_overall": sil_overall, "clusters": {}}

        for k, v in coh.items():
            pct = v["n"] / n_total * 100
            sil_k = sil_per_cluster.get(k, float("nan"))
            cos = v["mean_cosine_sim"]
            interp = "coherent" if cos > 0.4 else ("moderate" if cos > 0.2 else "noisy")
            print(f"{tag:<6} {k:<4} {v['n']:>6} {pct:>5.1f}%  {cos:>8.4f}  {sil_k:>10.4f}  {interp}")
            tag_result["clusters"][k] = {
                "n": v["n"], "pct": round(pct, 2),
                "mean_cosine_sim": cos, "silhouette": sil_k,
            }

        print(f"{'':6} {'ALL':<4} {'':>6} {'':>6}  {'':>8}  {sil_overall:>10.4f}  overall silhouette")
        print()
        all_results[tag] = tag_result

    out_path = os.path.join(out_dir, "cluster_analysis.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {out_path}")

    # Summary table
    print(f"\n{'K':<6} {'Silhouette':>12}")
    print("-" * 20)
    for tag, r in all_results.items():
        print(f"{tag:<6} {r['silhouette_overall']:>12.4f}")


if __name__ == "__main__":
    main()
