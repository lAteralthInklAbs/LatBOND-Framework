#!/bin/bash
# LatBOND Full Study - Single Architecture Runner
# Usage: ./run_single_arch.sh streaming_cnn
# Usage: ./run_single_arch.sh causal_transformer
# Usage: ./run_single_arch.sh tcn

set -e

ARCH=${1:?"Usage: $0 <architecture>"}
SEED=${2:-42}
OUTPUT_DIR=${3:-"outputs"}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================="
echo "LatBOND Full Study - ${ARCH}"
echo "Start: $(date)"
echo "Seed: ${SEED}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================="

# Install dependencies
pip install --quiet --break-system-packages -r requirements.txt 2>/dev/null || \
pip install --quiet -r requirements.txt

# Run training for this architecture (14 models)
python run_full_study.py \
    --mode single_arch \
    --architecture ${ARCH} \
    --output_dir ${OUTPUT_DIR} \
    --seed ${SEED} \
    2>&1 | tee "${OUTPUT_DIR}/${ARCH}_training_log_${TIMESTAMP}.txt"

echo ""
echo "============================================="
echo "Completed: ${ARCH}"
echo "End: $(date)"
echo "============================================="
