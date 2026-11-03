#!/usr/bin/env bash
# Author: Laure Berti-Equille (laure.berti@ird.fr)
# Paper:  Berti-Equille, L. (2026). Where Is the Truth When AI Agents Disagree?
#         Dependence-Aware Belief Propagation for Unsupervised Truth Discovery.
#         Paper accepted at IEEE BigData 2026.
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

mkdir -p logs

stamp() { date "+%Y-%m-%d %H:%M:%S"; }

{
  echo "============================================================"
  echo "  E3 SCALABILITY  —  start $(stamp)"
  echo "  root: $ROOT"
  echo "  args: $*"
  echo "============================================================"

  echo "[$(stamp)] analyze_E3_scalability.py"
  python scripts/analyze_E3_scalability.py "$@"


  echo "============================================================"
  echo "  E3 SCALABILITY  —  done  $(stamp)"
  echo "============================================================"
} 2>&1 | tee logs/E3.log
