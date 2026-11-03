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

from bptd.config import default_config, BPConfig
from bptd.data import make_synthetic
from bptd.bp import run_bptd
from bptd import metrics as M

SYNTH_SEED = 7
SYNTH_N_VALUES = 4
SCALE_NQ = (100, 500, 1000, 5000)
SCALE_NA = (4, 8, 16, 32)

N_FIXED_DEFAULT = 1000
M_LIST_DEFAULT = [4, 8, 16, 32, 64, 128]
REPEATS_DEFAULT = 3


def _read_scaling_csv(path):
    if not os.path.isfile(path):
        return None
    nm, rt = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            nm.append(float(row["nm"]))
            rt.append(float(row["runtime_s"]))
    if not nm:
        return None
    return np.asarray(nm, float), np.asarray(rt, float)


def _regenerate_scaling():
    cfg = default_config()
    nm, rt = [], []
    for nq in SCALE_NQ:
        for na in SCALE_NA:
            data = make_synthetic(n_questions=nq, n_agents=na,
                                  n_values=SYNTH_N_VALUES, seed=SYNTH_SEED)
            t0 = time.time()
            run_bptd(data, cfg.bp)
            nm.append(nq * na)
            rt.append(time.time() - t0)
            print(f"  [regen] n={nq:>5} m={na:>2} -> {rt[-1]:.3f}s")
    return np.asarray(nm, float), np.asarray(rt, float)


def _ols_slope(log_x, log_y):
    A = np.vstack([log_x, np.ones_like(log_x)]).T
    (k, b), *_ = np.linalg.lstsq(A, log_y, rcond=None)
    pred = k * log_x + b
    ss_res = float(np.sum((log_y - pred) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(k), float(b), float(r2)


def bootstrap_slope_ci(nm, runtime, n_boot=1000, seed=12345):
    log_x = np.log(nm)
    log_y = np.log(runtime)
    k_hat, b_hat, r2 = _ols_slope(log_x, log_y)

    rng = np.random.default_rng(seed)
    n = len(nm)
    ks = np.empty(n_boot, float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ks[i], _, _ = _ols_slope(log_x[idx], log_y[idx])
    ci_low, ci_high = np.percentile(ks, [2.5, 97.5])
    return dict(
        k_hat=round(k_hat, 4), ci_low=round(float(ci_low), 4),
        ci_high=round(float(ci_high), 4), r2=round(r2, 4),
        n_points=n, b=round(b_hat, 4), n_boot=n_boot,
    )


def _time_bptd(data, mode, repeats):
    bp = BPConfig()
    bp.mode = mode
    times, n_iter, acc = [], 0, float("nan")
    for _ in range(repeats):
        t0 = time.time()
        res = run_bptd(data, bp)
        times.append(time.time() - t0)
        n_iter = res.n_iter
        acc = M.accuracy(res.pred, data.gold)
    return float(np.mean(times)), float(np.std(times)), n_iter, float(acc)


def agent_axis_sweep(n_fixed, m_list, repeats):
    rows, timing = [], []
    for m in m_list:
        data = make_synthetic(n_questions=n_fixed, n_agents=m,
                              n_values=SYNTH_N_VALUES, seed=SYNTH_SEED)
        t_op, sd_op, it_op, acc_op = _time_bptd(data, "onepass", repeats)
        t_lp, sd_lp, it_lp, acc_lp = _time_bptd(data, "loopy", repeats)
        rows.append(dict(
            n_fixed=n_fixed, m=m,
            runtime_onepass_s=round(t_op, 4), runtime_onepass_sd_s=round(sd_op, 4),
            runtime_loopy_s=round(t_lp, 4), runtime_loopy_sd_s=round(sd_lp, 4),
            n_iter_loopy=it_lp, accuracy_loopy=round(acc_lp, 4), repeats=repeats,
        ))
        timing.append(dict(
            condition=f"n{n_fixed}_m{m}", n_fixed=n_fixed, m=m, repeats=repeats,
            time_onepass_mean_s=round(t_op, 4), time_onepass_sd_s=round(sd_op, 4),
            time_loopy_mean_s=round(t_lp, 4), time_loopy_sd_s=round(sd_lp, 4),
        ))
        print(f"  n={n_fixed} m={m:>4}  one-pass={t_op:.4f}s  "
              f"loopy={t_lp:.4f}s ({it_lp} it, acc={acc_lp:.3f})")
    return rows, timing


def fit_linear_quadratic(m_arr, rt_arr):
    if len(m_arr) < 2:
        return dict(a=float("nan"), c=float("nan"), r2=float("nan"),
                    quad_share_at_mmax=float("nan"), m_max=int(m_arr[-1]),
                    m2_nonnegligible=False)
    A = np.vstack([m_arr, m_arr ** 2]).T
    (a, c), *_ = np.linalg.lstsq(A, rt_arr, rcond=None)
    pred = a * m_arr + c * m_arr ** 2
    ss_res = float(np.sum((rt_arr - pred) ** 2))
    ss_tot = float(np.sum((rt_arr - rt_arr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    m_max = float(m_arr.max())
    quad_term = c * m_max ** 2
    total_term = a * m_max + c * m_max ** 2
    quad_share = quad_term / total_term if total_term != 0 else float("nan")
    nonneg = bool(np.isfinite(quad_share) and quad_share >= 0.10 and c > 0)
    return dict(a=round(float(a), 8), c=round(float(c), 10), r2=round(r2, 4),
                quad_share_at_mmax=round(float(quad_share), 4),
                m_max=int(m_max), m2_nonnegligible=nonneg)


def _write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[E3] wrote {path}  ({len(rows)} rows)")


def run(out_dir, n_fixed, m_list, repeats, n_boot, smoke):
    results_dir = out_dir
    csv_path = os.path.join(results_dir, "c5_scaling_synthetic.csv")
    t_all0 = time.perf_counter()

    data_pts = _read_scaling_csv(csv_path)
    if data_pts is None:
        print(f"[E3] {csv_path} not found -- regenerating run_c5 scaling grid")
        nm, runtime = _regenerate_scaling()
    else:
        nm, runtime = data_pts
        print(f"[E3] loaded {len(nm)} scaling points from c5_scaling_synthetic.csv")

    ci = bootstrap_slope_ci(nm, runtime, n_boot=n_boot)
    slope_fields = ["k_hat", "ci_low", "ci_high", "r2", "n_points", "b", "n_boot"]
    slope_row = {k: ci[k] for k in slope_fields}
    _write_csv(os.path.join(results_dir, "e3_slope_ci.csv"),
               [slope_row], slope_fields)
    print(f"[E3] log-log slope k_hat={ci['k_hat']}  "
          f"95% CI=[{ci['ci_low']}, {ci['ci_high']}]  "
          f"R2={ci['r2']}  (n={ci['n_points']}, {ci['n_boot']} boot)")

    print(f"\n[E3] agent-axis sweep  n_fixed={n_fixed}  m={m_list}  repeats={repeats}")
    rows, timing = agent_axis_sweep(n_fixed, m_list, repeats)
    axis_fields = [
        "n_fixed", "m", "runtime_onepass_s", "runtime_onepass_sd_s",
        "runtime_loopy_s", "runtime_loopy_sd_s",
        "n_iter_loopy", "accuracy_loopy", "repeats",
    ]
    _write_csv(os.path.join(results_dir, "e3_agent_axis.csv"), rows, axis_fields)

    m_arr = np.asarray([r["m"] for r in rows], float)
    rt_op = np.asarray([r["runtime_onepass_s"] for r in rows], float)
    rt_lp = np.asarray([r["runtime_loopy_s"] for r in rows], float)
    fit_op = fit_linear_quadratic(m_arr, rt_op)
    fit_lp = fit_linear_quadratic(m_arr, rt_lp)
    print(f"[E3] one-pass fit runtime = {fit_op['a']:.3e}*m + {fit_op['c']:.3e}*m^2 "
          f"(R2={fit_op['r2']})")
    print(f"     m^2 share at m={fit_op['m_max']}: {fit_op['quad_share_at_mmax']}  "
          f"-> m^2 {'NON-NEGLIGIBLE' if fit_op['m2_nonnegligible'] else 'negligible'}")
    print(f"[E3] loopy    fit runtime = {fit_lp['a']:.3e}*m + {fit_lp['c']:.3e}*m^2 "
          f"(R2={fit_lp['r2']})  m^2 share={fit_lp['quad_share_at_mmax']}")

    quad_fields = ["which", "a", "c", "r2", "quad_share_at_mmax",
                   "m_max", "m2_nonnegligible"]
    _write_csv(os.path.join(results_dir, "e3_agent_axis_fit.csv"),
               [dict(which="onepass", **fit_op), dict(which="loopy", **fit_lp)],
               quad_fields)

    timing_fields = ["condition", "n_fixed", "m", "repeats",
                     "time_onepass_mean_s", "time_onepass_sd_s",
                     "time_loopy_mean_s", "time_loopy_sd_s"]
    _write_csv(os.path.join(results_dir, "e3_timing_per_condition.csv"),
               timing, timing_fields)

    total = time.perf_counter() - t_all0
    grand = sum(r["time_onepass_mean_s"] + r["time_loopy_mean_s"] for r in timing)
    print("\n" + "=" * 64)
    print("  E3 TIMING BANNER" + ("  [SMOKE]" if smoke else ""))
    print(f"  conditions      : {len(timing)} (n={n_fixed} x {len(m_list)} m-values)")
    print(f"  repeats/cond    : {repeats}")
    print(f"  total timed run : {grand:.3f}s (sum of mean one-pass+loopy)")
    print(f"  wall clock      : {total:.3f}s")
    print("=" * 64)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E3 scalability: slope CI + agent-axis sweep")
    p.add_argument("--out-dir", default=None,
                   help="output root (default cfg.output.dir)")
    p.add_argument("--n-fixed", type=int, default=N_FIXED_DEFAULT,
                   help="fixed n_questions for the agent-axis sweep")
    p.add_argument("--m-list", type=int, nargs="+", default=None,
                   help="agent counts to sweep (default 4 8 16 32 64 128)")
    p.add_argument("--repeats", type=int, default=REPEATS_DEFAULT,
                   help="timing repeats per condition (mean reported)")
    p.add_argument("--n-boot", type=int, default=1000,
                   help="bootstrap resamples for the slope CI")
    p.add_argument("--smoke", action="store_true",
                   help="fast: slope CI on existing CSV + m in {4,8}, 1 repeat")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = default_config()
    out_dir = args.out_dir or cfg.output.dir
    if args.smoke:
        m_list = [4, 8]
        repeats = 1
        n_fixed = min(args.n_fixed, 200)
        n_boot = min(args.n_boot, 200)
    else:
        m_list = args.m_list or M_LIST_DEFAULT
        repeats = args.repeats
        n_fixed = args.n_fixed
        n_boot = args.n_boot
    run(out_dir, n_fixed, m_list, repeats, n_boot, args.smoke)


if __name__ == "__main__":
    main()
