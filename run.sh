#!/bin/bash
# Usage:
#   sbatch run.sh train                       # train with config.yaml
#   sbatch run.sh train my_config.yaml        # train with custom config
#   sbatch run.sh infer                       # synthesize with config.yaml
#   sbatch run.sh infer my_config.yaml        # synthesize with custom config
#   sbatch run.sh nf_train                    # train normalizing flow
#   sbatch run.sh nf_infer                    # NF parameter inference

#SBATCH -J FM_v3
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --constraint=a100-80gb
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH -t 3-00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load python
source /mnt/home/mliu1/env/bin/activate
mkdir -p logs

MODE="${1:-train}"
CONFIG="${2:-config.yaml}"

echo "Node: $(hostname) | Mode: $MODE | Config: $CONFIG"
nvidia-smi

case "$MODE" in
  train)
    srun python train.py --config "$CONFIG"
    ;;
  infer)
    python infer.py --config "$CONFIG"
    ;;
  nf_train)
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    srun python nf_train.py --config "$CONFIG"
    ;;
  nf_infer)
    python nf_infer.py --config "$CONFIG"
    ;;
  *)
    echo "Unknown mode: $MODE (use 'train', 'infer', 'nf_train', or 'nf_infer')"
    exit 1
    ;;
esac
