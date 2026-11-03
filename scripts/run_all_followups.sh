#!/usr/bin/env bash
# Author: Laure Berti-Equille (laure.berti@ird.fr)
# Paper:  Berti-Equille, L. (2026). Where Is the Truth When AI Agents Disagree?
#         Dependence-Aware Belief Propagation for Unsupervised Truth Discovery.
#         Paper accepted at IEEE BigData 2026.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || source venv/bin/activate 2>/dev/null || true
mkdir -p logs

ts() { date '+%Y-%m-%d %H:%M %Z'; }

echo "[followups] $(ts) START E1–E4 (full runs)"

echo "[followups] $(ts) E1 — BP-TD vs MV significance (decisive)"
bash scripts/run_exp_E1.sh

echo "[followups] $(ts) E2 — tuned classical-TD baselines"
bash scripts/run_exp_E2.sh

echo "[followups] $(ts) E3 — scalability slope CI + agent-axis sweep"
bash scripts/run_exp_E3.sh

echo "[followups] $(ts) E4 — reliability ρ robustness"
bash scripts/run_exp_E4.sh

echo "[followups] $(ts) DONE. Results in outputs/e{1,2,3,4}_*.csv"
echo "[followups] Next: fill {{FILL_AFTER_RUN}} in descriptions/description_E*.md, then /checker full."
