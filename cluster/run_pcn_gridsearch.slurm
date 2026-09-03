#!/bin/bash
#SBATCH --job-name=pcn-gridsearch
#SBATCH --partition=h200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=48:00:00

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_IMAGE="$PROJECT_ROOT/containers/pcn_uv.sif"

mkdir -p "$PROJECT_ROOT/out/gridsearch"

singularity run --nv \
    --bind "$PROJECT_ROOT:/workspace" \
    "$CONTAINER_IMAGE" \
    "$@"
