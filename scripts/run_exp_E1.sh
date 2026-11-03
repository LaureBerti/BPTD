#!/usr/bin/env bash
# Author: Laure Berti-Equille (laure.berti@ird.fr)
# Paper:  Berti-Equille, L. (2026). Where Is the Truth When AI Agents Disagree?
#         Dependence-Aware Belief Propagation for Unsupervised Truth Discovery.
#         Paper accepted at IEEE BigData 2026.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

mkdir -p logs

echo "============================================================"
echo "[E1] START  $(date '+%Y-%m-%d %H:%M:%S')"
echo "[E1] root: ${ROOT}"
echo "[E1] data: cached real answers data/{mmlu,sciq,truthfulqa}/answers.csv"
echo "============================================================"

python3 scripts/analyze_E1_mv_sig.py "$@" 2>&1 | tee logs/E1.log

echo "============================================================"
echo "[E1] DONE   $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
