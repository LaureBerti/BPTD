#!/usr/bin/env python3
# Author: Laure Berti-Equille (laure.berti@ird.fr)
# Paper:  Berti-Equille, L. (2026). Where Is the Truth When AI Agents Disagree?
#         Dependence-Aware Belief Propagation for Unsupervised Truth Discovery.
#         Paper accepted at IEEE BigData 2026.

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from copy import deepcopy

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bptd.config import default_config
from bptd.data import TDData, load_td_data
from bptd.bp import run_bptd
from bptd import metrics as M

DEFAULT_SEEDS = [42, 123, 7, 2024, 31]
DEFAULT_NQ = [160, 100]


def _bp_cooccur_cfg(cfg):
    bpc = deepcopy(cfg.bp)
    bpc.dep_mode = "cooccur"
    bpc.dependence_aware = True
    return bpc


def _stride_subset(base: TDData, n_keep: int) -> TDData:
    n = base.n_questions
    if n_keep >= n:
        return base
    idx = np.linspace(0, n - 1, num=n_keep).round().astype(int)
    idx = np.unique(idx)
    return _td_index(base, idx)


def _td_index(base: TDData, idx: np.ndarray) -> TDData:
    return TDData(
        questions=[base.questions[i] for i in idx],
        agents=base.agents, domains=base.domains,
        answers=base.answers[:, idx].copy(), gold=base.gold[idx].copy(),
        base_model=base.base_model,
    )


def _bootstrap_questions(base: TDData, seed: int) -> TDData:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, base.n_questions, size=base.n_questions)
    return _td_index(base, idx)


def _perseed_rho(base: TDData, bpc, seed: int) -> float:
    data = _bootstrap_questions(base, seed)
    true_rel = data.measured_agent_accuracy()
    est = run_bptd(data, bpc).reliability
    return float(M.spearman(est, true_rel))


def _boot_ci_agents(est, true, n_boot=1000, seed=0):
    e = np.asarray(est, float); t = np.asarray(true, float)
    valid = np.isfinite(e) & np.isfinite(t)
    e, t = e[valid], t[valid]
    m = len(e)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, m, size=m)
        r = M.spearman(e[idx], t[idx])
        if np.isfinite(r):
            vals.append(r)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def _boot_over_agents(data: TDData, bpc):
    true_rel = data.measured_agent_accuracy()
    est = run_bptd(data, bpc).reliability
    rho = float(M.spearman(est, true_rel))
    lo, hi = _boot_ci_agents(est, true_rel)
    return rho, lo, hi


def _write_csv(path, rows, tag="E4"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[{tag}] wrote {path}  ({len(rows)} rows)")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="E4 reliability-recovery robustness (per-seed + 100q vs 160q)")
    ap.add_argument("--datasets", default="mmlu",
                    help="comma-separated dataset names (default: mmlu)")
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                    help="comma-separated bootstrap seeds")
    ap.add_argument("--n-questions", default=",".join(str(n) for n in DEFAULT_NQ),
                    help="comma-separated question counts to compare (e.g. 160,100)")
    ap.add_argument("--smoke", action="store_true",
                    help="smoke run: mmlu, seeds 42,123, n=160 only")
    args = ap.parse_args(argv)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    nqs = [int(n) for n in args.n_questions.split(",") if n.strip()]

    if args.smoke:
        datasets = ["mmlu"]
        seeds = [42, 123]
        nqs = [160]
        print("[E4] SMOKE MODE: mmlu, seeds 42,123, n=160 only")

    cfg = default_config()
    bpc = _bp_cooccur_cfg(cfg)

    results_dir = os.path.join(_ROOT, "outputs")
    e4_dir = results_dir
    os.makedirs(e4_dir, exist_ok=True)

    perseed_rows = []
    summary_rows = []
    timing_rows = []

    for ds in datasets:
        print(f"\n[E4] === {ds} ===")
        full = load_td_data(os.path.join(cfg.data.folder, ds), cfg.data.max_questions)
        _ans = np.where((full.answers >= 0).any(axis=0))[0]
        if len(_ans) < full.n_questions:
            full = TDData(questions=[full.questions[i] for i in _ans],
                          agents=full.agents, domains=full.domains,
                          answers=full.answers[:, _ans].copy(),
                          gold=full.gold[_ans].copy(), base_model=full.base_model)
        print(f"[E4] {ds}: loaded {full.n_questions} answered questions, "
              f"{full.n_agents} agents")

        for nq in nqs:
            cond = f"E4_{ds}_n{nq}"
            base = _stride_subset(full, nq)
            actual_nq = base.n_questions
            if actual_nq != nq:
                print(f"[E4] {ds}: requested n={nq}, using actual n={actual_nq} "
                      f"(dataset has {full.n_questions} questions)")

            t_cond = time.time()

            seed_rhos = []
            for sd in seeds:
                t_sd = time.time()
                rho = _perseed_rho(base, bpc, sd)
                dt = round(time.time() - t_sd, 3)
                perseed_rows.append(dict(
                    dataset=ds, n_questions=actual_nq, seed=sd,
                    spearman=round(rho, 4), time_seed_s=dt,
                ))
                seed_rhos.append(rho)
                print(f"[E4] {ds} n={actual_nq} seed {sd:>4}: "
                      f"rho={rho:+.4f}  ({dt:.2f}s)")

            valid = np.array([r for r in seed_rhos if np.isfinite(r)], float)
            mean_rho = float(valid.mean()) if valid.size else float("nan")
            sd_rho = float(valid.std(ddof=1)) if valid.size > 1 else float("nan")

            boot_rho, boot_lo, boot_hi = _boot_over_agents(base, bpc)

            time_condition_s = round(time.time() - t_cond, 3)

            summary_rows.append(dict(
                dataset=ds, n_questions=actual_nq,
                mean_spearman=round(mean_rho, 4),
                sd_spearman=round(sd_rho, 4) if np.isfinite(sd_rho) else "",
                boot_rho=round(boot_rho, 4),
                boot_ci_low=round(boot_lo, 4),
                boot_ci_high=round(boot_hi, 4),
                time_condition_s=time_condition_s,
            ))
            timing_rows.append(dict(
                condition=cond, dataset=ds, n_questions=actual_nq,
                n_seeds=len(seeds), time_condition_s=time_condition_s,
                time_total_s=time_condition_s,
            ))

            print(f"[E4] {ds} n={actual_nq}: per-seed mean rho={mean_rho:+.4f} "
                  f"(SD={sd_rho:.4f})  |  boot-over-agents rho={boot_rho:+.4f} "
                  f"95%CI=[{boot_lo:+.4f},{boot_hi:+.4f}]")

    perseed_out = os.path.join(results_dir, "e4_reliability_perseed.csv")
    summary_out = os.path.join(results_dir, "e4_reliability_summary.csv")
    _write_csv(perseed_out, perseed_rows)
    _write_csv(summary_out, summary_rows)

    timing_out = os.path.join(e4_dir, "e4_timing_per_condition.csv")
    _write_csv(timing_out, timing_rows)

    for ds in datasets:
        rs = {r["n_questions"]: r for r in summary_rows if r["dataset"] == ds}
        ks = sorted(rs.keys())
        if len(ks) >= 2:
            lo_n, hi_n = ks[0], ks[-1]
            print(f"[E4] {ds.upper()} rho rises with question count: "
                  f"n={lo_n} mean={rs[lo_n]['mean_spearman']:+.3f} "
                  f"boot={rs[lo_n]['boot_rho']:+.3f}  ->  "
                  f"n={hi_n} mean={rs[hi_n]['mean_spearman']:+.3f} "
                  f"boot={rs[hi_n]['boot_rho']:+.3f}")

    total_t = sum(r["time_total_s"] for r in timing_rows)
    print("\n" + "=" * 60)
    print(f"[E4] TIMING BANNER: {len(timing_rows)} condition(s), "
          f"total {total_t:.1f}s")
    for r in timing_rows:
        print(f"[E4]   {r['condition']:<20} "
              f"condition={r['time_condition_s']:>8.3f}s "
              f"({r['n_seeds']} seeds)")
    print("=" * 60)


if __name__ == "__main__":
    main()
