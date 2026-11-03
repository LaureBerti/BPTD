#!/usr/bin/env bash
# Author: Laure Berti-Equille (laure.berti@ird.fr)
# Paper:  Berti-Equille, L. (2026). Where Is the Truth When AI Agents Disagree?
#         Dependence-Aware Belief Propagation for Unsupervised Truth Discovery.
#         Paper accepted at IEEE BigData 2026.
set -euo pipefail
DEST="$(cd "$(dirname "$0")/.." && pwd)/experiments/third_party/CARE"
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/SprocketLab/CARE.git "$DEST"
fi
git -C "$DEST" checkout 72f5b29a822d9934d31777c10a5c38369884c9dc
echo "CARE ready at $DEST"
