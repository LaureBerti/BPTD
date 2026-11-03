#!/usr/bin/env python3
# Author: Laure Berti-Equille (laure.berti@ird.fr)
# Paper:  Berti-Equille, L. (2026). Where Is the Truth When AI Agents Disagree?
#         Dependence-Aware Belief Propagation for Unsupervised Truth Discovery.
#         Paper accepted at IEEE BigData 2026.
from __future__ import annotations

import csv
import os
import sys
import time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from bptd.data import load_td_data, TDData
from bptd.bp import run_bptd
from bptd.baselines import majority_vote
from bptd.config import BPConfig
from bptd import metrics as M

ARC_DIR   = os.path.join(ROOT, "data", "arc")
OUT_CSV   = os.path.join(ROOT, "outputs", "bloc_stress_arc.csv")
TIMING_CSV = os.path.join(ROOT, "outputs", "timing_W2B_stress.csv")
K_VALUES   = [0, 2, 4, 6, 8, 10, 12, 16, 20, 24]
CONDITIONS = ["shared", "distinct"]
RHO        = 1.0
N_SEEDS    = 20

VARIANTS = {
    "BP-TD-cooccur":      dict(dep_mode="cooccur",      dependence_aware=True),
    "BP-TD-family":       dict(dep_mode="family",       dependence_aware=True),
    "BP-TD-shared_error": dict(dep_mode="shared_error", dependence_aware=True),
    "BP-TD-none":         dict(dep_mode="none",         dependence_aware=False),
}


def _bpcfg(**kw) -> BPConfig:
    return BPConfig(mode="onepass", dep_threshold=0.5, alpha0=2.0, beta0=1.0, **kw)


def inject_bloc(base: TDData, k: int, condition: str, rho: float,
                rng: np.random.Generator) -> TDData:
    m, n = base.n_agents, base.n_questions
    if k == 0:
        return base
    bloc = np.full((k, n), -1, dtype=np.int64)
    active = rng.random(n) < rho
    for q in range(n):
        if not active[q]:
            continue
        dom = base.domain_size(q)
        g = int(base.gold[q])
        wrongs = [v for v in range(dom) if v != g]
        if not wrongs:
            continue
        shared_wrong = int(rng.choice(wrongs))
        bloc[:, q] = shared_wrong
    answers = np.vstack([base.answers, bloc])
    agents = list(base.agents) + [f"bloc_{i}" for i in range(k)]
    base_bm = list(base.base_model) if base.base_model else list(base.agents)
    if condition == "shared":
        bloc_bm = ["synthetic_bloc"] * k
    else:
        bloc_bm = [f"synthetic_bloc_{i}" for i in range(k)]
    return TDData(questions=list(base.questions), agents=agents,
                  domains=base.domains, answers=answers,
                  gold=base.gold.copy(), base_model=base_bm + bloc_bm)


def main() -> None:
    t_total = time.time()
    base = load_td_data(ARC_DIR, None)
    mv0 = M.accuracy(majority_vote(base).pred, base.gold)
    print(f"[W2B-STRESS] real ARC: {base.n_agents} agents x {base.n_questions} q | "
          f"MV={mv0:.4f}")

    rows, timing = [], []
    for cond in CONDITIONS:
        for k in K_VALUES:
            t_cond = time.time()
            accs = {name: [] for name in (["MajorityVote"] + list(VARIANTS))}
            for s in range(N_SEEDS):
                rng = np.random.default_rng(1000 * (CONDITIONS.index(cond) + 1)
                                            + 37 * k + s)
                data = inject_bloc(base, k, cond, RHO, rng)
                accs["MajorityVote"].append(M.accuracy(majority_vote(data).pred, data.gold))
                for name, kw in VARIANTS.items():
                    res = run_bptd(data, _bpcfg(**kw))
                    accs[name].append(M.accuracy(res.pred, data.gold))
                if k == 0:
                    break
            mv_mean = float(np.mean(accs["MajorityVote"]))
            for name in ["MajorityVote"] + list(VARIANTS):
                arr = np.array(accs[name])
                rows.append(dict(
                    k=k, rho=RHO, condition=cond, method=name,
                    acc_mean=round(float(arr.mean()), 4),
                    acc_sd=round(float(arr.std()), 4),
                    delta_vs_mv=round(float(arr.mean() - mv_mean), 4),
                    n_seeds=len(arr),
                    n_agents=base.n_agents + k,
                ))
            t_c = round(time.time() - t_cond, 3)
            timing.append(dict(condition=cond, k=k, n_seeds=N_SEEDS,
                               n_methods=1 + len(VARIANTS),
                               time_condition_s=t_c,
                               time_search_s=0.0, time_eval_s=t_c,
                               n_candidates=0))
            best_bp = max((r for r in rows if r["k"] == k and r["condition"] == cond
                           and r["method"].startswith("BP-TD")),
                          key=lambda r: r["acc_mean"])
            print(f"  {cond:8s} k={k:2d} | MV={mv_mean:.3f} "
                  f"cooccur={[r for r in rows if r['k']==k and r['condition']==cond and r['method']=='BP-TD-cooccur'][0]['acc_mean']:.3f} "
                  f"family={[r for r in rows if r['k']==k and r['condition']==cond and r['method']=='BP-TD-family'][0]['acc_mean']:.3f} "
                  f"none={[r for r in rows if r['k']==k and r['condition']==cond and r['method']=='BP-TD-none'][0]['acc_mean']:.3f} "
                  f"| best_bp={best_bp['method'].split('-')[-1]} ({t_c:.1f}s)")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    cols = ["k", "rho", "condition", "method", "acc_mean", "acc_sd",
            "delta_vs_mv", "n_seeds", "n_agents"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(TIMING_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "k", "n_seeds", "n_methods",
                                          "time_condition_s", "time_search_s",
                                          "time_eval_s", "n_candidates"])
        w.writeheader()
        w.writerows(timing)

    tt = time.time() - t_total
    print(f"\n{'-'*60}")
    print(f"[W2B-STRESS] wrote {OUT_CSV}  ({len(rows)} rows)")
    print(f"Timing: total {tt:.1f}s  ({tt/60:.1f} min) | "
          f"{len(timing)} conditions, mean {np.mean([t['time_condition_s'] for t in timing]):.2f}s each")


if __name__ == "__main__":
    main()
