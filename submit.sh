#!/bin/bash
# Thin wrapper that sets the SLURM job name from $MODE before calling sbatch.
# Usage:
#   ./submit.sh train                       # train with config.yaml
#   ./submit.sh train my_config.yaml        # train with custom config
#   ./submit.sh infer                       # synthesize with config.yaml
#   ./submit.sh nf_train                    # train normalizing flow
#   ./submit.sh nf_infer                    # NF parameter inference

MODE="${1:-train}"
shift
sbatch --job-name="FM_v3_${MODE}" run.sh "$MODE" "$@"
