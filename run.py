"""run.py — MoSV pipeline entry point. Stages: activations, sweep_K."""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml


def load_config(path: str = "configs/defan.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("No GPU found, using CPU")
    return dev


def load_model(cfg: dict, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    model_name = cfg["model"]["name"]
    print(f"Loading {model_name}...")

    quantization_config = None
    if cfg["model"].get("load_in_8bit") and device.type == "cuda":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=cfg["model"].get("device_map", "auto") if device.type == "cuda" else None,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params")
    return model, tokenizer


def stage_activations(cfg: dict, device: torch.device) -> None:
    from mosv.data.dataset import load_pairs
    from mosv.activation.extract import (
        ActivationExtractor,
        collect_contrastive_activations,
        save_activations,
    )

    model, tokenizer = load_model(cfg, device)
    layers = cfg["activation"]["layers_to_probe"]
    extractor = ActivationExtractor(model, layers)

    train_pairs = load_pairs(os.path.join(cfg["paths"]["data_dir"], "mc_train.jsonl"))
    print(f"Collecting activations for {len(train_pairs)} training pairs...")

    act_dir = cfg["paths"]["activations_dir"]
    os.makedirs(act_dir, exist_ok=True)
    diff_vectors, prompt_vectors, _ = collect_contrastive_activations(
        extractor, tokenizer, train_pairs, device, out_dir=act_dir
    )
    save_activations(diff_vectors, prompt_vectors, act_dir)


def _parse_k_variants(k_variant_strs):
    """Parse ['K2', 'K4', 'K10', 'DBSCAN'] -> [('K2', 2), ('K4', 4), ('K10', 10), ('DBSCAN', None)]"""
    variants = []
    for s in k_variant_strs:
        if s.upper() == "DBSCAN":
            variants.append(("DBSCAN", None))
        elif s.upper().startswith("K"):
            k_val = int(s[1:])
            variants.append((s, k_val))
        else:
            raise ValueError(f"Unknown k_variant: {s!r}. Expected 'K<int>' or 'DBSCAN'.")
    return variants


def stage_sweep_K(cfg: dict, device: torch.device, k_variant_strs=None) -> None:
    """
    For each K variant: cluster diff vectors, train router, save artifacts.
    Use scripts/evaluate.py for downstream evaluation.
    Results saved to activations/sweep_{tag}/, checkpoints/sweep_{tag}/, outputs/sweep_{tag}/.
    """
    from mosv.activation.extract import load_activations
    from mosv.clustering.probe import select_best_layer
    from mosv.clustering.cluster import (
        cluster_diff_vectors,
        cluster_diff_vectors_dbscan,
        save_clustering,
    )
    from mosv.routing.model import MoSVRouter
    from mosv.routing.train import build_router_dataset, train_router, save_router

    print("Loading activations...")
    diff_vectors, prompt_vectors = load_activations(cfg["paths"]["activations_dir"])
    layers = cfg["activation"]["layers_to_probe"]
    best_layer_pos, layer_scores = select_best_layer(diff_vectors, layers)
    best_layer = layers[best_layer_pos]
    X = diff_vectors[:, best_layer_pos, :]
    d_model = prompt_vectors.shape[2]
    print(f"Best layer: {best_layer} (pos {best_layer_pos}), d_model={d_model}")

    if k_variant_strs:
        variants = _parse_k_variants(k_variant_strs)
    else:
        variants = [("K2", 2), ("K3", 3), ("K4", 4), ("K5", 5), ("DBSCAN", None)]

    sweep_results = {}

    for tag, K_val in variants:
        print(f"\n{'='*60}")
        print(f"Variant: {tag}")

        sub_act_dir  = os.path.join(cfg["paths"]["activations_dir"],  f"sweep_{tag}")
        sub_ckpt_dir = os.path.join(cfg["paths"]["checkpoints_dir"], f"sweep_{tag}")
        sub_out_dir  = os.path.join(cfg["paths"]["outputs_dir"],      f"sweep_{tag}")
        for d in (sub_act_dir, sub_ckpt_dir, sub_out_dir):
            os.makedirs(d, exist_ok=True)

        # 1. Cluster
        print("  Clustering...")
        if K_val is not None:
            steering_vectors, cluster_labels, _, _ = cluster_diff_vectors(
                X, K=K_val,
                pca_components=cfg["activation"]["pca_components"],
                random_state=cfg["clustering"]["random_state"],
                n_init=cfg["clustering"]["n_init"],
            )
            K = K_val
        else:
            steering_vectors, cluster_labels, K = cluster_diff_vectors_dbscan(
                X,
                pca_components=cfg["activation"]["pca_components"],
                min_samples=cfg["clustering"].get("dbscan_min_samples", 10),
                random_state=cfg["clustering"]["random_state"],
            )
            if K < 2:
                print(f"  DBSCAN returned K={K} (degenerate) — skipping variant.")
                sweep_results[tag] = {"K": K, "skipped": True}
                continue

        save_clustering(
            steering_vectors=steering_vectors,
            cluster_labels=cluster_labels,
            layer_scores=layer_scores,
            best_layer=best_layer,
            K=K,
            silhouette_scores={},
            out_dir=sub_act_dir,
        )

        # Cluster separability probe
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score as _cv_score
        _X_scaled = StandardScaler().fit_transform(X)
        _clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        _sep_scores = _cv_score(_clf, _X_scaled, cluster_labels, cv=5, scoring="accuracy")
        sep_acc = float(_sep_scores.mean())
        print(f"  Cluster separability probe: {sep_acc:.4f} (random baseline={1/K:.4f})")

        # t-SNE
        from mosv.viz.plots import plot_tsne_clusters
        plot_tsne_clusters(
            X, cluster_labels,
            save_path=os.path.join(sub_act_dir, f"tsne_{tag}.png"),
        )

        # 2. Train router
        print(f"  Training router (K={K})...")
        router = MoSVRouter(
            d_model=d_model, K=K,
            top_k=min(cfg["router"]["top_k"], K),
            dropout=cfg["router"]["dropout"],
        )
        dataset = build_router_dataset(prompt_vectors, cluster_labels, best_layer_pos)
        history = train_router(
            router=router, dataset=dataset,
            epochs=cfg["router"]["epochs"],
            lr=cfg["router"]["lr"],
            weight_decay=cfg["router"]["weight_decay"],
            load_balance_coef=cfg["router"]["load_balance_coef"],
            batch_size=cfg["router"]["batch_size"],
            val_ratio=cfg["router"]["val_ratio"],
            device=device,
        )
        save_router(router, sub_ckpt_dir)
        print(f"  Router val_acc={history['val_acc'][-1]:.4f}")

        variant_summary = {
            "K": K,
            "cluster_sep_acc": sep_acc,
            "cluster_sep_baseline": round(1 / K, 4),
        }
        with open(os.path.join(sub_out_dir, "summary.json"), "w") as f:
            json.dump(variant_summary, f, indent=2)

        sweep_results[tag] = variant_summary

    # Final summary table
    print("\n" + "=" * 50)
    print(f"{'Variant':<10} {'K':>6} {'Sep Acc':>10} {'Baseline':>10}")
    print("-" * 50)
    for tag, _ in variants:
        res = sweep_results.get(tag, {})
        if res.get("skipped"):
            continue
        K_disp = res.get("K", "?")
        sep = res.get("cluster_sep_acc", 0)
        base = res.get("cluster_sep_baseline", 0)
        print(f"  {tag:<8} {K_disp:>6} {sep:>9.4f} {base:>9.4f}")
    print("=" * 50)

    sweep_summary_path = os.path.join(cfg["paths"]["outputs_dir"], "sweep_K_summary.json")
    with open(sweep_summary_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSaved sweep summary to {sweep_summary_path}")


def main():
    parser = argparse.ArgumentParser(description="MoSV pipeline — DefAn")
    parser.add_argument("--stage", required=True, choices=["activations", "sweep_K"])
    parser.add_argument("--config", default="configs/defan.yaml")
    parser.add_argument("--k_variants", nargs="+", default=None,
                        help="K variants for sweep_K, e.g. K2 K4 K6 K10 K15 K20 DBSCAN")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_tag", type=str, default="",
                        help="prefix for sweep subdirs, e.g. s123")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.run_tag:
        cfg["_run_tag"] = args.run_tag
    if args.seed is not None:
        cfg["clustering"]["random_state"] = args.seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = get_device()

    if args.stage == "activations":
        stage_activations(cfg, device)
    elif args.stage == "sweep_K":
        stage_sweep_K(cfg, device, k_variant_strs=args.k_variants)


if __name__ == "__main__":
    main()
