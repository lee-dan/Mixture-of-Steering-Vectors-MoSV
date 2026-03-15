"""analyze_coherence.py — Intra-cluster diff vector cosine similarity per K variant."""
import argparse, os, sys, json
import numpy as np
import yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def cosine_sim_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-8)
    X_norm = X / norms
    return X_norm @ X_norm.T

def cluster_coherence(X, labels, K, max_per_cluster=500, seed=42):
    rng = np.random.default_rng(seed)
    results = {}
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx) > max_per_cluster:
            idx = rng.choice(idx, max_per_cluster, replace=False)
        Xk = X[idx]
        if len(Xk) < 2:
            results[k] = {"n": len(idx), "mean_cosine_sim": float("nan")}
            continue
        sim = cosine_sim_matrix(Xk)
        upper = sim[np.triu_indices(len(Xk), k=1)]
        results[k] = {"n": int(len(idx)), "mean_cosine_sim": float(upper.mean())}
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/defan.yaml")
    parser.add_argument("--k_tags", nargs="+", default=["K2", "K3", "K4", "K5"])
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    from mosv.activation.extract import load_activations
    from mosv.clustering.cluster import load_clustering
    act_dir = cfg["paths"]["activations_dir"]
    layers = cfg["activation"]["layers_to_probe"]
    diff_vectors, _ = load_activations(act_dir)
    print(f"\n{'K':<6} {'Cluster':<10} {'N':>6}  {'CosSim':>8}  {'Interpretation'}")
    print("-" * 60)
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
        all_results[tag] = coh
        overall = np.mean([v["mean_cosine_sim"] for v in coh.values() if not np.isnan(v["mean_cosine_sim"])])
        for k, v in coh.items():
            interp = "coherent" if v["mean_cosine_sim"] > 0.4 else ("moderate" if v["mean_cosine_sim"] > 0.2 else "noisy")
            print(f"{tag:<6} {'C'+str(k):<10} {v['n']:>6}  {v['mean_cosine_sim']:>8.4f}  {interp}")
        print(f"{'':6} {'OVERALL':<10} {'':>6}  {overall:>8.4f}")
        print()
    out_path = os.path.join(cfg["paths"]["outputs_dir"], "cluster_coherence.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
