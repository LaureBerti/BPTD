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
echo "[E4] START  $(date '+%Y-%m-%d %H:%M:%S')"
echo "[E4] root: ${ROOT}"
echo "[E4] data: cached real answers data/mmlu/answers.csv + gold.csv"
echo "============================================================"

python3 scripts/analyze_E4_reliability_robustness.py "$@" 2>&1 | tee logs/E4.log

echo "============================================================"
echo "[E4] DONE   $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
