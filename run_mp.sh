#!/bin/bash
# Model-parallel inference: UNet split across 2 GPUs.
# Usage:
#   sbatch run_mp.sh config/config_infer_l25_on_SB35.yaml
#   sbatch run_mp.sh config/config_infer_l25_on_SB35.yaml --compile
#   sbatch run_mp.sh config/config_infer_l25_on_SB35.yaml --compile --n-samples 1

#SBATCH -J FM_v3_infer_mp
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100-80gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH -t 3-00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load python
source /mnt/home/mliu1/env/bin/activate
mkdir -p logs

CONFIG="${1:-config/config_infer_l25_on_SB35.yaml}"
shift
EXTRA_ARGS="$@"

echo "Node: $(hostname) | Config: $CONFIG | Extra: $EXTRA_ARGS"
nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python infer_mp.py --config "$CONFIG" $EXTRA_ARGS
