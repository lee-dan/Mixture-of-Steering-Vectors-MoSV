#!/bin/bash
# Run this ONCE on FarmShare before submitting any SLURM jobs.
# Usage: bash jobs/farmshare_setup.sh
#
# This script creates the mosv-env virtual environment and installs all
# requirements. All SLURM job scripts assume this has already been run.

set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_DIR"
echo "Project dir: $PROJECT_DIR"

module purge
module load python/3.13.11

if [ ! -d "mosv-env" ]; then
    echo "Creating virtual environment..."
    python -m venv mosv-env
fi

source mosv-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete. mosv-env is ready."
echo "You can now submit jobs:"
echo "  sbatch jobs/farmshare_activations.sh"
echo "  sbatch jobs/farmshare_cluster_router.sh"
echo "  sbatch jobs/farmshare_eval.sh"
echo "  sbatch jobs/farmshare_sweep.sh"
