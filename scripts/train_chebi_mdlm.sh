#!/bin/bash
#SBATCH --job-name=mdlm-chebi
#SBATCH --output=watch_folder/mdlm-chebi-%j.out
#SBATCH --gpus=1
#SBATCH --mem=16GB

export HYDRA_FULL_ERROR=1

python main.py \
  model=small_chebi \
  data=chebi \
  wandb.name=mdlm-chebi \
  parameterization=subs \
  backbone=dit \
  sampling.predictor=ddpm_cache \
  sampling.steps=1000 \
  +training.max_epochs=100