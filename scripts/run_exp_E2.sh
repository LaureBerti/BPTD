#!/usr/bin/env bash
# Author: Laure Berti-Equille (laure.berti@ird.fr)
# Paper:  Berti-Equille, L. (2026). Where Is the Truth When AI Agents Disagree?
#         Dependence-Aware Belief Propagation for Unsupervised Truth Discovery.
#         Paper accepted at IEEE BigData 2026.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

mkdir -p logs outputs

echo "================================================================"
echo "  E2  start : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  root      : ${ROOT}"
echo "  python    : $(which python3)"
echo "================================================================"

python3 scripts/analyze_E2_tuned_classical.py "$@" \
    2>&1 | tee logs/E2.log

echo "================================================================"
echo "  E2  done  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  log       : logs/E2.log"
echo "================================================================"
