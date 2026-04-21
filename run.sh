#!/usr/bin/env bash
set -euo pipefail
echo "Generating synthetic data..."
python data/generate_data.py
echo "Executing notebook..."
jupyter nbconvert --to notebook --execute notebook.ipynb \
  --output notebook_executed.ipynb
echo "Pipeline complete. See notebook.ipynb and outputs/figures/."
