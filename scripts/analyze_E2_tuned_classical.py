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

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bptd.config import default_config
from bptd.data import load_td_data
from bptd import baselines as B
from bptd import metrics as M
from bptd_main import _td_subset

N_SPLITS_DEFAULT = 40
VAL_FRAC = 0.5
SPLIT_SEED_BASE = 7000

DATASETS_DEFAULT = ["mmlu", "sciq", "truthfulqa"]

TUNED_METHODS = {
    "DawidSkene": dict(
        grid=[10, 30, 50, 100], default=50,
        predict=lambda hp: (lambda d: B.dawid_skene(d, n_iter=hp).pred),
    ),
    "CATD": dict(
        grid=[0.01, 0.05, 0.1, 0.5], default=0.05,
        predict=lambda hp: (lambda d: B.catd(d, alpha=hp).pred),
    ),
    "LTM": dict(
        grid=[20, 40, 80], default=40,
        predict=lambda hp: (lambda d: B.ltm(d, n_iter=hp).pred),
    ),
    "TruthFinder": dict(
        grid=[0.1, 0.3, 0.5], default=0.3,
        predict=lambda hp: (lambda d: B.truthfinder(d, gamma=hp).pred),
    ),
}

acc = lambda d, pred: M.accuracy(pred, d.gold)


def _splits(base, n_splits):
    for s in range(n_splits):
        rng = np.random.default_rng(SPLIT_SEED_BASE + s)
        perm = rng.permutation(base.n_questions)
        cut = int(round(VAL_FRAC * base.n_questions))
        yield _td_subset(base, perm[:cut]), _td_subset(base, perm[cut:])


def analyze_dataset_method(base, name, spec, n_splits):
    grid = spec["grid"]
    default_hp = spec["default"]
    make = spec["predict"]

    default_test = []
    tuned_test = []
    mv_test = []
    chosen_counts = {}

    t_search = 0.0
    t_eval = 0.0

    t_cond0 = time.perf_counter()
    for val, test in _splits(base, n_splits):
        t0 = time.perf_counter()
        val_acc = {hp: acc(val, make(hp)(val)) for hp in grid}
        best_hp = max(grid, key=lambda hp: val_acc[hp])
        t_search += time.perf_counter() - t0
        chosen_counts[best_hp] = chosen_counts.get(best_hp, 0) + 1

        t0 = time.perf_counter()
        tuned_test.append(acc(test, make(best_hp)(test)))
        default_test.append(acc(test, make(default_hp)(test)))
        t_eval += time.perf_counter() - t0

        mv_test.append(acc(test, B.majority_vote(test).pred))
    t_condition = time.perf_counter() - t_cond0

    default_test = np.asarray(default_test, float)
    tuned_test = np.asarray(tuned_test, float)
    mv_test = np.asarray(mv_test, float)

    chosen_hp = max(chosen_counts, key=chosen_counts.get)
    diff_vs_mv_tuned = float(np.mean(tuned_test - mv_test))
    below_mv = bool(diff_vs_mv_tuned < 0)

    res = dict(
        method=name,
        default_acc=round(float(default_test.mean()), 4),
        default_sd=round(float(default_test.std()), 4),
        tuned_acc=round(float(tuned_test.mean()), 4),
        tuned_sd=round(float(tuned_test.std()), 4),
        chosen_hp=str(chosen_hp),
        chosen_hp_share=f"{chosen_counts[chosen_hp]}/{n_splits}",
        mv_acc=round(float(mv_test.mean()), 4),
        diff_vs_mv_tuned=round(diff_vs_mv_tuned, 4),
        below_mv=below_mv,
        n_candidates=len(grid),
        time_search_s=round(t_search, 3),
        time_eval_s=round(t_eval, 3),
        time_condition_s=round(t_condition, 3),
    )
    return res


def _write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[E2] wrote {path}  ({len(rows)} rows)")


def run(datasets, methods, n_splits, folder, max_questions, out_dir):
    main_fields = [
        "dataset", "method", "default_acc", "default_sd", "tuned_acc", "tuned_sd",
        "chosen_hp", "diff_vs_mv_tuned", "below_mv",
        "time_search_s", "time_eval_s", "n_candidates", "time_condition_s",
    ]
    timing_fields = [
        "dataset", "method", "n_splits", "n_candidates",
        "time_search_s", "time_eval_s", "time_condition_s",
    ]
    main_rows, timing_rows = [], []
    t_all0 = time.perf_counter()

    for ds in datasets:
        base = load_td_data(os.path.join(folder, ds), max_questions)
        _ans = np.where((base.answers >= 0).any(axis=0))[0]
        if len(_ans) < base.n_questions:
            print(f"[E2] {ds}: restricting to {len(_ans)}/{base.n_questions} ANSWERED questions")
            base = _td_subset(base, _ans)
        print(f"\n[E2] dataset={ds}  n_questions={base.n_questions} "
              f"n_agents={base.n_agents}  splits={n_splits}")
        for mname in methods:
            spec = TUNED_METHODS[mname]
            r = analyze_dataset_method(base, mname, spec, n_splits)
            verdict = ("STILL < MV (collapse persists)"
                       if r["below_mv"] else "tuned >= MV")
            print(f"  {ds:<11} {mname:<12} "
                  f"default={r['default_acc']:.4f}±{r['default_sd']:.4f}  "
                  f"tuned={r['tuned_acc']:.4f}±{r['tuned_sd']:.4f}  "
                  f"hp={r['chosen_hp']} ({r['chosen_hp_share']})  "
                  f"MV={r['mv_acc']:.4f}  Δtuned-MV={r['diff_vs_mv_tuned']:+.4f}  "
                  f"[{verdict}]")
            print(f"      timing: search={r['time_search_s']:.3f}s "
                  f"eval={r['time_eval_s']:.3f}s "
                  f"condition={r['time_condition_s']:.3f}s "
                  f"({r['n_candidates']} candidates)")
            full = {"dataset": ds, **r}
            main_rows.append({k: full[k] for k in main_fields})
            timing_rows.append(dict(
                dataset=ds, method=mname, n_splits=n_splits,
                n_candidates=r["n_candidates"],
                time_search_s=r["time_search_s"], time_eval_s=r["time_eval_s"],
                time_condition_s=r["time_condition_s"],
            ))

    results_dir = out_dir
    _write_csv(os.path.join(results_dir, "e2_tuned_classical.csv"),
               main_rows, main_fields)
    _write_csv(os.path.join(results_dir, "e2_timing_per_condition.csv"),
               timing_rows, timing_fields)

    total = time.perf_counter() - t_all0
    grand_search = sum(r["time_search_s"] for r in timing_rows)
    grand_eval = sum(r["time_eval_s"] for r in timing_rows)
    print("\n" + "=" * 64)
    print("  E2 TIMING BANNER")
    print(f"  conditions      : {len(timing_rows)} "
          f"({len(datasets)} datasets × {len(methods)} methods)")
    print(f"  splits/cond     : {n_splits}")
    print(f"  total search    : {grand_search:.3f}s")
    print(f"  total eval      : {grand_eval:.3f}s")
    print(f"  wall clock      : {total:.3f}s")
    print("=" * 64)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E2 tuned-classical-baselines analysis")
    p.add_argument("--datasets", nargs="+", default=DATASETS_DEFAULT,
                   help="datasets under data/<name>/")
    p.add_argument("--methods", nargs="+", default=list(TUNED_METHODS.keys()),
                   choices=list(TUNED_METHODS.keys()))
    p.add_argument("--n-splits", type=int, default=N_SPLITS_DEFAULT,
                   help="number of val/test splits (run_tuned uses 40)")
    p.add_argument("--folder", default=None, help="data root (default cfg.data.folder)")
    p.add_argument("--max-questions", type=int, default=None,
                   help="cap questions (default cfg.data.max_questions)")
    p.add_argument("--out-dir", default=None,
                   help="output root (default cfg.output.dir)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = default_config()
    folder = args.folder or cfg.data.folder
    max_questions = args.max_questions if args.max_questions is not None else cfg.data.max_questions
    out_dir = args.out_dir or cfg.output.dir
    run(args.datasets, args.methods, args.n_splits, folder, max_questions, out_dir)


if __name__ == "__main__":
    main()
