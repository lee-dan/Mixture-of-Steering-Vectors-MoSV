# Mixture-of-Steering-Vectors (MoSV)

**Stanford CS224N Final Project** · Daniel Lee, Feolu Kolawole, Vedant Srinivas

MoSV replaces single-vector inference-time steering with a bank of K specialized vectors, routing among them per prompt via a learned sparse MLP. No retraining of the base model required.

---

## Method

1. Extract contrastive activation diff vectors `(correct − hallucinated)` at each candidate layer
2. Select the best layer via linear probe (highest geometric structure)
3. K-means on diff vectors → K steering vectors
4. Train sparse MLP router: prompt hidden state → top-2 weighted combination of vectors
5. At inference, inject composite vector into residual stream at every decoding step

---

## Results

| System | Accuracy | Δ | p-value |
|---|---|---|---|
| Vanilla | 19.7% | — | — |
| Single-Vec CAA | 20.0% | +0.3pp | 0.292 (ns) |
| MoSV-K8 | **22.1%** | **+2.4pp** | **9.1e-6** |
| MoSV-K10 | **22.1%** | **+2.4pp** | **1.1e-5** |

All K≥2 variants pass Benjamini-Hochberg correction (m=10, α=0.05). Full sweep results in `outputs/defan_accuracy_results.json`.

---

## Dataset

[DefAn](https://arxiv.org/abs/2406.09155) — 8 factual domains (FIFA, census, Nobel, Oscars, UN dates, QS rankings, conferences, math). Short factual answers enable exact-match evaluation with no LLM judge. 85/15 per-domain train/eval split reserved before any inference.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.template .env   # add HF_TOKEN
```

On FarmShare: `bash setup_farmshare.sh`

---

## Running

```bash
python scripts/prepare_data.py --config configs/defan.yaml
python run.py --stage activations --config configs/defan.yaml
python run.py --stage sweep_K --config configs/defan.yaml --k_variants K2 K4 K6 K8 K10 K15 K20 K35 K50
python scripts/evaluate.py --config configs/defan.yaml --k_tags K2 K4 K6 K8 K10 K15 K20 K35 K50
python scripts/analysis_all_k.py
```

SLURM: `jobs/farmshare_train.sh` runs the full pipeline. Per-K jobs in `jobs/farmshare_sweep_K{N}.sh`.

---

## Repository Structure

```
run.py                         pipeline entry point
configs/defan.yaml             hyperparameters

mosv/
  activation/extract.py        contrastive activation extraction
  clustering/cluster.py        K-means + layer selection
  clustering/probe.py          linear probe
  routing/model.py             MoSVRouter (3-layer MLP, sparse top-k)
  routing/train.py             router training loop
  steering/mosv.py             inference-time steering hook
  steering/baselines.py        Vanilla and SingleVec baselines
  data/dataset.py              data loading utilities

scripts/
  prepare_data.py              build DefAn contrastive pairs
  evaluate.py                  exact-match accuracy eval
  analysis_all_k.py            silhouette, t-SNE, cluster charts, heatmap
  analyze_clusters.py          per-cluster stats
  analyze_coherence.py         intra-cluster cosine similarity
  plot_defan_eval_tsne.py      eval set t-SNE by domain
  plot_router_animation.py     router assignment GIF
  visualize.py                 t-SNE / PCA / UMAP decision boundaries

jobs/                          SLURM scripts (FarmShare)
outputs/                       results, figures, per-K summaries
manuscript/                    paper and poster
```

---

## References

- Rimsky et al. (2024). Steering Llama 2 via Contrastive Activation Addition.
- Turner et al. (2024). Steering Language Models With Activation Engineering.
- Rahman et al. (2024). DefAn: Definitive Answer Dataset for LLMs Hallucination Evaluation.
- Grattafiori et al. (2024). The Llama 3 Herd of Models.
