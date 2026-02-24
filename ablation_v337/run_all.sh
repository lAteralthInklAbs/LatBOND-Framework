#!/bin/bash
# LatBOND Full Study - Local Sequential Runner
# Trains all 42 models on a single GPU.
# Estimated time: ~126 hours on T4 GPU.

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs_${TIMESTAMP}"
SEED=42

echo "============================================="
echo "LatBOND Full Study - All Architectures"
echo "42 models (3 arch x 5 budgets x 3 conditions - 3 redundant)"
echo "Start: $(date)"
echo "Output: ${OUTPUT_DIR}"
echo "============================================="

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "[GPU Check]"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "[WARN] No GPU detected! Training will be very slow."
fi

# Install dependencies
echo "[Installing dependencies...]"
pip install --quiet --break-system-packages -r requirements.txt 2>/dev/null || \
pip install --quiet -r requirements.txt

# Navigate to project directory
cd "$(dirname "$0")"

# Run full study locally
python run_full_study.py \
    --mode local \
    --output_dir ${OUTPUT_DIR} \
    --seed ${SEED} \
    2>&1 | tee "${OUTPUT_DIR}/full_study_log_${TIMESTAMP}.txt"

# Collect results and run analysis
python scripts/collect_results.py \
    --results_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR}/analysis

echo ""
echo "============================================="
echo "Full study complete!"
echo "End: $(date)"
echo "Results: ${OUTPUT_DIR}/"
echo "Analysis: ${OUTPUT_DIR}/analysis/"
echo "============================================="
