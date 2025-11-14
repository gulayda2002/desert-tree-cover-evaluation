#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root (this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$SCRIPT_DIR/.venv/bin/python"
MAIN="$SCRIPT_DIR/main.py"

# Create timestamped output directory
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="$SCRIPT_DIR/output/run-$STAMP"

# Ensure dependencies are installed (optional but convenient)
"$PY" -m pip install -r "$SCRIPT_DIR/requirements.txt"

# Baseline-like RF settings
RF_MAX_PIXELS=${RF_MAX_PIXELS:-20000}  # Moderate sampling
RF_N_EST=${RF_N_EST:-150}              # Reasonable ensemble
RF_MAX_DEPTH=${RF_MAX_DEPTH:-25}       # Not too deep
RF_MIN_SAMPLES=${RF_MIN_SAMPLES:-5}    # More conservative splits

# U-Net hyperparameters with augmentation
EPOCHS=${EPOCHS:-60}                   # More epochs with augmentation
BATCH_SIZE=${BATCH_SIZE:-8}            # Conservative batch

# Run both RF and U-Net, saving plots and samples
"$PY" "$MAIN" \
  --rf-max-pixels "$RF_MAX_PIXELS" \
  --rf-n-estimators "$RF_N_EST" \
  --rf-max-depth "$RF_MAX_DEPTH" \
  --rf-min-samples "$RF_MIN_SAMPLES" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --outdir "$OUTDIR" \
  --viz-samples 12

# Print where artifacts are stored
echo "Saved outputs to: $OUTDIR"
