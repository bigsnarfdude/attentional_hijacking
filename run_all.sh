#!/usr/bin/env bash
# Run all six attentional hijacking experiments for one model size.
#
# Usage:
#   bash run_all.sh --model 4b    # ~2 hours, 16GB VRAM
#   bash run_all.sh --model 12b   # ~4 hours, 40GB VRAM
#   bash run_all.sh --model 27b   # ~6 hours, 80GB VRAM
#
# Requires: HF_TOKEN environment variable set
# Results land in: results/{model}/

set -e

MODEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Usage: bash run_all.sh --model 4b|12b|27b"
    exit 1
fi

if [[ "$MODEL" != "4b" && "$MODEL" != "12b" && "$MODEL" != "27b" ]]; then
    echo "Error: --model must be 4b, 12b, or 27b"
    exit 1
fi

if [[ -z "$HF_TOKEN" ]]; then
    echo "Warning: HF_TOKEN not set. GemmaScope 2 SAE download may fail."
    echo "Set it with: export HF_TOKEN=hf_..."
    echo ""
fi

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/scripts" && pwd)"
LOG_DIR="results/${MODEL}/logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  Attentional Hijacking Experiments — model: ${MODEL}"
echo "  Logs: ${LOG_DIR}/"
echo "============================================================"
echo ""

run_script() {
    local name="$1"
    local script="$2"
    echo "--- [$(date +%H:%M:%S)] Starting: ${name} ---"
    python "${SCRIPTS_DIR}/${script}" --model "${MODEL}" 2>&1 | tee "${LOG_DIR}/${name}.log"
    echo "--- [$(date +%H:%M:%S)] Done: ${name} ---"
    echo ""
}

run_script "1_feature_swap"         "feature_swap.py"
run_script "2_attention_knockout"   "attention_knockout.py"
run_script "3_activation_patching"  "activation_patching.py"
run_script "4_held_out_validation"  "held_out_validation.py"
run_script "5_cross_domain_sae"     "cross_domain_sae.py"
run_script "6_statistical_rigor"    "statistical_rigor.py"

echo "============================================================"
echo "  All experiments complete. Results in: results/${MODEL}/"
echo "============================================================"
