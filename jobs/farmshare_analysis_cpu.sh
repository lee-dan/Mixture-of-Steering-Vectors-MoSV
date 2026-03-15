#!/bin/bash
#SBATCH --job-name=mosv_analysis
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err

# CPU-only analysis job: silhouette scores, domain bar charts, t-SNE,
# cluster interpretability, correlation analysis, per-domain heatmap,
# router animation, and interpretability report.

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source mosv-env/bin/activate
source .env

export HF_TOKEN
export GCP_PROJECT_NAME
export STUDENT_EMAIL
SCRATCH="${SCRATCH:-/scratch/users/$USER}"
export SCRATCH
export HF_HOME="${SCRATCH}/hf_cache"
mkdir -p "$HF_HOME"

echo "Hostname: $(hostname)"
echo "CPUs available: $(nproc)"
echo "Start: $(date)"

echo "=== Step 1/3: Comprehensive K analysis ==="
PYTHONUNBUFFERED=1 python scripts/analysis_all_k.py

echo "=== Step 2/3: Router animation GIF ==="
PYTHONUNBUFFERED=1 python scripts/plot_router_animation.py

echo "=== Step 3/3: Interpretability report ==="
PYTHONUNBUFFERED=1 python scripts/analysis_interpretability_report.py

echo "Done. End: $(date)"
