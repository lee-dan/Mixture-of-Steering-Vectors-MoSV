#!/bin/bash
#SBATCH --job-name=mosv_sweep_K12
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/sweep_K12_%j.out
#SBATCH --error=logs/sweep_K12_%j.err
#SBATCH --exclude=oat-02,oat-04

set -e
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source mosv-env/bin/activate
source .env

export HF_TOKEN
export GCP_PROJECT_NAME
export STUDENT_EMAIL
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/application_default_credentials.json"
SCRATCH="${SCRATCH:-/scratch/users/$USER}"
export SCRATCH
export HF_HOME="${SCRATCH}/hf_cache"
mkdir -p "$HF_HOME"

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start: $(date)"

echo "=== sweep_K variant K12 ==="
PYTHONUNBUFFERED=1 python run_experiment.py \
    --config configs/defan.yaml \
    --stage sweep_K \
    --k_variants K12

echo "=== DefAn accuracy eval K12 ==="
PYTHONUNBUFFERED=1 python scripts/eval_defan_accuracy.py \
    --config configs/defan.yaml \
    --k_tags K12 \
    --vanilla_acc 0.197 \
    --singlevec_acc 0.200

echo "Done. End: $(date)"
