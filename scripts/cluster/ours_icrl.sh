#!/bin/bash
# Our full ICRL loop (Maximum Entropy) on HalfCheetah-with-position, 10 seeds,
# to compare with Malik's own runs (malik_icrl.sh: wandb project icrl-hc-icrl,
# group malik_me). Same shape as his job script; one array task per seed (0-9).
# Setup (venv, clone, expert pile): docs/decisions.md, 2026-10-28.
# Submit from ~/scratch/icrl_benchmarking:
#   mkdir -p logs && sbatch scripts/cluster/ours_icrl.sh
#SBATCH --account=aip-s2ganapa
#SBATCH --job-name=ours_icrl
#SBATCH --array=0-9
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%A_%a.out

# Modules first, then the venv: loading modules resets some variables.
module load StdEnv/2023 python/3.13.2 mujoco/3.3.0
source $HOME/icrl_venv/bin/activate
# wandb offline (no internet on compute nodes); torch on the one CPU we asked for.
export WANDB_MODE=offline WANDB_SILENT=true OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Config paths (the expert pile) are relative to the repo root; runs go to
# runs/maximum_entropy_seed<k>/, wandb's offline files to ./wandb.
cd $HOME/scratch/icrl_benchmarking

python scripts/run_icrl.py --config configs/hc/maximum_entropy.yaml \
    --seed $SLURM_ARRAY_TASK_ID \
    --wandb-project icrl-hc-icrl --wandb-group ours_me
