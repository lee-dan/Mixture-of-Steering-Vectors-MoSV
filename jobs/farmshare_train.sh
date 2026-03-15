#!/bin/bash
#SBATCH --job-name=mosv_train
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Full MoSV training pipeline: data → activations → sweep_K → eval → analysis

set -e
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source mosv-env/bin/activate
source .env

export HF_TOKEN
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/application_default_credentials.json"
SCRATCH="${SCRATCH:-/scratch/users/$USER}"
export SCRATCH
export HF_HOME="${SCRATCH}/hf_cache"
mkdir -p "$HF_HOME"

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start: $(date)"

K_VARIANTS="K2 K4 K6 K8 K10 K15 K20 K35 K50"

echo "=== 1/4: Build contrastive pairs ==="
PYTHONUNBUFFERED=1 python scripts/prepare_data.py --config configs/defan.yaml

echo "=== 2/4: Extract activations ==="
PYTHONUNBUFFERED=1 python run.py --stage activations --config configs/defan.yaml

echo "=== 3/4: Sweep K (cluster + router + eval) ==="
for K in $K_VARIANTS; do
    echo "--- $K ---"
    PYTHONUNBUFFERED=1 python run.py \
        --stage sweep_K \
        --config configs/defan.yaml \
        --k_variants $K

    PYTHONUNBUFFERED=1 python scripts/evaluate.py \
        --config configs/defan.yaml \
        --k_tags $K
done

echo "=== 4/4: Analysis ==="
PYTHONUNBUFFERED=1 python scripts/analysis_all_k.py

echo "Done. End: $(date)"
