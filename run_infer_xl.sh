#!/bin/bash
# Inference on gpuxl (H200 141GB) for large-memory jobs like full 256^3 SB35
# Usage: sbatch run_infer_xl.sh infer config/config_infer_l25_on_SB35.yaml

#SBATCH -J FM_v3_infer_xl
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH -t 1-00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load python
source /mnt/home/mliu1/env/bin/activate
mkdir -p logs

CONFIG="${2:-config/config_infer_l25_on_SB35.yaml}"

echo "Node: $(hostname) | Config: $CONFIG"
nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python infer.py --config "$CONFIG"
