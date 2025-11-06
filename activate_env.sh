#!/bin/bash
# Activation script for alex_train environment

echo "Activating alex_train conda environment..."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/alex_train

echo "✓ Environment activated"
echo "✓ Python: $(which python)"
echo "✓ Python version: $(python --version)"
echo "✓ Swift command: $(which swift)"
echo ""
echo "You can now run training scripts, e.g.:"
echo "  bash examples/train/lora_sft.sh"
echo "  swift sft --help"

