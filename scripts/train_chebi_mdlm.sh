#!/bin/bash
#SBATCH --job-name=mdlm-chebi
#SBATCH --output=watch_folder/mdlm-chebi-%j.out
#SBATCH --gpus=1
#SBATCH --mem=16GB  # Reduced memory requirement due to smaller model

export HYDRA_FULL_ERROR=1

python main.py \
  model=small_chebi \
  data=chebi \
  wandb.name=mdlm-chebi \
  parameterization=subs \
  backbone=dit \
  sampling.predictor=ddpm_cache \
  sampling.steps=1000 \
  +training.max_epochs=100 \        # Instead of steps, use epochs for small dataset
#   +training.val_check_interval=0.5 \ # Validate twice per epoch
#   loader.batch_size=16 \            # Smaller baxstch size
#   loader.eval_batch_size=16 \
#   +optim.lr=1e-4 \                 # Slightly lower learning rate
#   +optim.weight_decay=0.01 \       # Added weight decay for regularization
#   model.text_conditioning=True \
#   +training.early_stopping.patience=10 \ # Add early stopping
#   +training.early_stopping.monitor='val/loss'