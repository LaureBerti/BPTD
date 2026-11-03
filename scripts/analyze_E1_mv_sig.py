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
from bptd.data import load_td_data
from bptd.bp import run_bptd
from bptd import baselines as B
from bptd import metrics as M

N_SPLITS = 40
VAL_FRAC = 0.5
BP_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SPLIT_SEED_BASE = 7000
FIXED_SPLIT_SEED = 0


def _td_subset(base, idx):
    from bptd.data import TDData
    return TDData(
        questions=[base.questions[i] for i in idx], agents=base.agents,
        domains=base.domains, answers=base.answers[:, idx].copy(),
        gold=base.gold[idx].copy(), base_model=base.base_model,
    )


def _bp_cooccur_predictor(cfg, th):
    c = deepcopy(cfg.bp)
    c.dep_mode = "cooccur"
    c.dependence_aware = True
    c.dep_threshold = th
    return lambda d: run_bptd(d, c).pred


def _mv_predictor():
    return lambda d: B.majority_vote(d).pred


def _acc(d, pred):
    return M.accuracy(pred, d.gold)


def _tune_bp_on_val(cfg, val):
    return max(BP_GRID, key=lambda t: _acc(val, _bp_cooccur_predictor(cfg, t)(val)))


def mcnemar_fixed_split(cfg, base):
    rng0 = np.random.default_rng(FIXED_SPLIT_SEED)
    perm0 = rng0.permutation(base.n_questions)
    cut0 = int(round(VAL_FRAC * base.n_questions))
    val0 = _td_subset(base, perm0[:cut0])
    test0 = _td_subset(base, perm0[cut0:])

    tgold = test0.gold
    tvalid = tgold >= 0

    th = _tune_bp_on_val(cfg, val0)
    bp_pred = _bp_cooccur_predictor(cfg, th)(test0)
    mv_pred = _mv_predictor()(test0)

    bp_correct = (np.asarray(bp_pred)[tvalid] == tgold[tvalid])
    mv_correct = (np.asarray(mv_pred)[tvalid] == tgold[tvalid])

    b = int((mv_correct & ~bp_correct).sum())
    c = int((bp_correct & ~mv_correct).sum())
    nd = b + c

    stat = ((abs(b - c) - 1.0) ** 2) / nd if nd > 0 else 0.0

    if nd == 0:
        p = 1.0
    else:
        try:
            from scipy.stats import binomtest
            p = float(binomtest(min(b, c), nd, 0.5, alternative="two-sided").pvalue)
        except Exception:
            from scipy.stats import binom_test
            p = float(binom_test(min(b, c), nd, 0.5, alternative="two-sided"))

    return dict(
        b=b, c=c, n_discordant=nd, stat=float(stat), p=float(p),
        bp_acc=float(bp_correct.mean()) if tvalid.any() else float("nan"),
        mv_acc=float(mv_correct.mean()) if tvalid.any() else float("nan"),
        threshold=th, n_test=int(tvalid.sum()),
    )


def nadeau_bengio_t(cfg, base, n_splits):
    from scipy.stats import t as student_t

    diffs = []
    cut = int(round(VAL_FRAC * base.n_questions))
    n_train = cut
    n_test = base.n_questions - cut
    for s in range(n_splits):
        rng = np.random.default_rng(SPLIT_SEED_BASE + s)
        perm = rng.permutation(base.n_questions)
        val = _td_subset(base, perm[:cut])
        test = _td_subset(base, perm[cut:])
        th = _tune_bp_on_val(cfg, val)
        bp_acc = _acc(test, _bp_cooccur_predictor(cfg, th)(test))
        mv_acc = _acc(test, _mv_predictor()(test))
        diffs.append(bp_acc - mv_acc)

    diffs = np.asarray(diffs, dtype=float)
    J = len(diffs)
    mean_diff = float(diffs.mean())
    var = float(diffs.var(ddof=1)) if J > 1 else 0.0

    correction = (1.0 / J) + (n_test / n_train if n_train > 0 else 0.0)
    corrected_se = np.sqrt(var * correction) if var > 0 else 0.0

    df = J - 1
    if corrected_se > 0:
        tstat = mean_diff / corrected_se
        p = float(2.0 * student_t.sf(abs(tstat), df))
        tcrit = float(student_t.ppf(0.975, df))
        ci_low = mean_diff - tcrit * corrected_se
        ci_high = mean_diff + tcrit * corrected_se
    else:
        tstat = float("inf") if mean_diff != 0 else 0.0
        p = 0.0 if mean_diff != 0 else 1.0
        ci_low = ci_high = mean_diff

    return dict(
        t=float(tstat), df=int(df), p=float(p), mean_diff=mean_diff,
        ci_low=float(ci_low), ci_high=float(ci_high), n_splits=J,
    )


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[E1] wrote {path}  ({len(rows)} rows)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="E1 BP-TD vs MV significance")
    ap.add_argument("--datasets", default="mmlu,sciq,truthfulqa",
                    help="comma-separated dataset names")
    ap.add_argument("--smoke", action="store_true",
                    help="smoke run: mmlu only, 4 splits (writes a real CSV row)")
    args = ap.parse_args(argv)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    n_splits = N_SPLITS
    if args.smoke:
        datasets = ["mmlu"]
        n_splits = 4
        print("[E1] SMOKE MODE: mmlu only, 4 splits")

    cfg = default_config()

    results_dir = os.path.join(_ROOT, "outputs")
    e1_dir = results_dir
    os.makedirs(e1_dir, exist_ok=True)

    rows = []
    timing_rows = []
    for ds in datasets:
        print(f"\n[E1] === {ds} ===")
        base = load_td_data(os.path.join(cfg.data.folder, ds), cfg.data.max_questions)
        _ans = np.where((base.answers >= 0).any(axis=0))[0]
        if len(_ans) < base.n_questions:
            print(f"[E1] {ds}: restricting to {len(_ans)}/{base.n_questions} ANSWERED questions")
            base = _td_subset(base, _ans)

        t_cond0 = time.time()
        mc = mcnemar_fixed_split(cfg, base)
        nb = nadeau_bengio_t(cfg, base, n_splits)
        time_condition_s = round(time.time() - t_cond0, 3)

        t_eval = time.time()
        _ = _acc(base, B.majority_vote(base).pred)
        time_eval_s = round(time.time() - t_eval, 3)

        mc_sig = "significant" if mc["p"] < 0.05 else "not significant"
        nb_sig = "significant" if nb["p"] < 0.05 else "not significant"

        rows.append(dict(
            dataset=ds,
            mcnemar_b=mc["b"], mcnemar_c=mc["c"], n_discordant=mc["n_discordant"],
            mcnemar_stat=round(mc["stat"], 4), mcnemar_p=round(mc["p"], 6),
            nb_t=round(nb["t"], 4), nb_df=nb["df"], nb_p=round(nb["p"], 6),
            mean_diff=round(nb["mean_diff"], 4),
            nb_ci_low=round(nb["ci_low"], 4), nb_ci_high=round(nb["ci_high"], 4),
            time_eval_s=time_eval_s, time_condition_s=time_condition_s,
        ))
        timing_rows.append(dict(
            condition=f"E1_{ds}", dataset=ds, n_splits=n_splits,
            time_eval_s=time_eval_s, time_condition_s=time_condition_s,
            time_total_s=round(time_eval_s + time_condition_s, 3),
        ))

        print(f"[E1] {ds}: McNemar b={mc['b']} (MV-correct,BP-wrong) "
              f"c={mc['c']} (BP-correct,MV-wrong) n_disc={mc['n_discordant']} "
              f"stat={mc['stat']:.3f} p={mc['p']:.4g} | "
              f"BP acc={mc['bp_acc']:.4f} MV acc={mc['mv_acc']:.4f} (th={mc['threshold']})")
        print(f"[E1] {ds}: Nadeau-Bengio corrected-t  t={nb['t']:.3f} "
              f"df={nb['df']} p={nb['p']:.4g} meanΔ={nb['mean_diff']:+.4f} "
              f"95%CI=[{nb['ci_low']:+.4f},{nb['ci_high']:+.4f}]")
        print(f"[E1] {ds.upper()} BP-TD vs MV: McNemar p={mc['p']:.4g}, "
              f"corrected-t p={nb['p']:.4g} -> {mc_sig} / {nb_sig}")

    out = os.path.join(results_dir, "e1_mv_significance.csv")
    _write_csv(out, rows)

    timing_out = os.path.join(e1_dir, "e1_timing_per_condition.csv")
    _write_csv(timing_out, timing_rows)

    total_t = sum(r["time_total_s"] for r in timing_rows)
    print("\n" + "=" * 60)
    print(f"[E1] TIMING BANNER: {len(timing_rows)} condition(s), "
          f"total {total_t:.1f}s")
    for r in timing_rows:
        print(f"[E1]   {r['condition']:<18} eval={r['time_eval_s']:>7.3f}s "
              f"condition={r['time_condition_s']:>8.3f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
