#!/bin/bash
# LatBOND Demo Lite v11 - One-Click Run Script
# =============================================
# Runs on GCP Compute Engine with T4 GPU
# Expected runtime: ~6-7 hours total
#
# v11 changes:
#   - Triple-head output: onset + offset + velocity prediction
#   - TripleHeadLoss combining focal + MSE losses
#   - Note-level evaluation ready (mir_eval)
#
# Usage:
#   bash run_all.sh                        # Full run
#   bash run_all.sh --skip_download        # Skip dataset download
#   bash run_all.sh --arch streaming_cnn   # Train only one architecture

set -e

echo "=============================================="
echo "LatBOND Demo Lite v11 - Starting..."
echo "=============================================="
echo "Time: $(date)"
echo ""

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
pip install --quiet --break-system-packages \
    torch torchaudio numpy scipy matplotlib pretty_midi mir_eval 2>/dev/null || \
pip install --quiet \
    torch torchaudio numpy scipy matplotlib pretty_midi mir_eval

echo ""

# Navigate to project directory
cd "$(dirname "$0")"

# Run
echo "[Starting training pipeline...]"
echo "Start time: $(date)"
echo ""

python run_demo.py "$@" 2>&1 | tee training_log_v11.txt

echo ""
echo "=============================================="
echo "Complete! Time: $(date)"
echo "=============================================="
echo ""
echo "Results in: outputs/"
echo "Log in: training_log_v11.txt"
echo ""
echo "To download results to your local machine:"
echo "  gcloud compute scp VM_NAME:~/latbond_demo_lite_v10/outputs/* . --zone=ZONE"
