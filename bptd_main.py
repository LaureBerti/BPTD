#!/usr/bin/env python3
# Author: Laure Berti-Equille (laure.berti@ird.fr)
# Paper:  Berti-Equille, L. (2026). Where Is the Truth When AI Agents Disagree?
#         Dependence-Aware Belief Propagation for Unsupervised Truth Discovery.
#         Paper accepted at IEEE BigData 2026.

from __future__ import annotations

import os
import sys
import csv
from typing import List, Optional

import numpy as np

from bptd.config import (
    BPTDConfig, default_config, apply_overrides, write_default_yaml,
)
from bptd.data import TDData, load_td_data, make_synthetic
from bptd.bp import run_bptd
from bptd import baselines as B
from bptd import metrics as M


def data_for_seed(cfg: BPTDConfig, seed: int) -> TDData:
    d = cfg.data
    if d.name == "synthetic":
        return make_synthetic(
            n_questions=d.syn_n_questions, n_agents=d.syn_n_agents,
            n_values=d.syn_n_values, reliability_low=d.syn_reliability_low,
            reliability_high=d.syn_reliability_high, missing_rate=d.syn_missing_rate,
            seed=seed,
        )
    base = load_td_data(os.path.join(d.folder, d.name), d.max_questions)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, base.n_questions, size=base.n_questions)
    return TDData(
        questions=[base.questions[i] for i in idx],
        agents=base.agents, domains=base.domains,
        answers=base.answers[:, idx].copy(), gold=base.gold[idx].copy(),
        base_model=base.base_model,
    )


def _out_dir(cfg: BPTDConfig) -> str:
    p = cfg.output.dir
    os.makedirs(p, exist_ok=True)
    return p


def run_demo(cfg: BPTDConfig) -> None:
    print("=== BP-TD demo (synthetic) ===")
    data = make_synthetic(n_questions=400, n_agents=8, n_values=4, seed=7)
    true_rel = data.__dict__.get("true_reliability")
    res = run_bptd(data, cfg.bp)
    acc = M.accuracy(res.pred, data.gold)
    f1 = M.macro_f1(res.pred, data.gold)
    rho = M.spearman(res.reliability, data.measured_agent_accuracy())
    print(f"BP-TD: acc={acc:.4f}  macroF1={f1:.4f}  "
          f"reliability rho={rho:.3f}  iters={res.n_iter}  conv={res.converged}")
    print(f"\n{'method':<18}{'acc':>8}{'macroF1':>9}")
    print(f"{'BP-TD':<18}{acc:>8.4f}{f1:>9.4f}")
    for name, fn in B.BASELINES.items():
        r = fn(data)
        a = M.accuracy(r.pred, data.gold)
        f = M.macro_f1(r.pred, data.gold)
        tag = " (oracle)" if r.oracle else ""
        print(f"{name:<18}{a:>8.4f}{f:>9.4f}{tag}")
    if true_rel is not None:
        print(f"\nplanted reliabilities : "
              f"{np.array2string(true_rel, precision=2)}")
        print(f"BP-TD estimated       : "
              f"{np.array2string(res.reliability, precision=2)}")


def run_compare(cfg: BPTDConfig) -> str:
    out = os.path.join(_out_dir(cfg), f"c1_results_{cfg.data.name}.csv")
    rows = []
    methods = ["BP-TD"] + list(B.BASELINES.keys())
    for seed in cfg.seeds:
        data = data_for_seed(cfg, seed)
        true_rel = data.measured_agent_accuracy()
        for method in methods:
            t = M.Timer()
            if method == "BP-TD":
                res = run_bptd(data, cfg.bp)
                pred, conf, rel = res.pred, res.confidence, res.reliability
                rt = res.runtime_s
            else:
                r = B.BASELINES[method](data)
                pred, conf, rel = r.pred, None, r.reliability
                rt = t.lap("method")
            te = t.lap("eval")
            acc = M.accuracy(pred, data.gold)
            f1 = M.macro_f1(pred, data.gold)
            rho = M.spearman(rel, true_rel) if rel is not None else float("nan")
            ece_v = M.ece(conf, pred, data.gold) if conf is not None else float("nan")
            rows.append(dict(
                dataset=cfg.data.name, method=method, seed=seed,
                accuracy=round(acc, 4), macro_f1=round(f1, 4),
                ece=round(ece_v, 4), spearman_rel=round(rho, 4),
                time_search_s=round(rt, 3), time_eval_s=round(te, 3),
                n_candidates=data.n_questions, time_condition_s=round(rt + te, 3),
            ))
            print(f"  {cfg.data.name} | seed {seed:>4} | {method:<16} "
                  f"acc={acc:.4f} F1={f1:.4f} rho={rho:.3f} ({rt:.2f}s)")
    _write_csv(out, rows)
    _print_summary(rows)
    return out


def _td_subset(base: TDData, idx) -> TDData:
    return TDData(
        questions=[base.questions[i] for i in idx], agents=base.agents,
        domains=base.domains, answers=base.answers[:, idx].copy(),
        gold=base.gold[idx].copy(), base_model=base.base_model,
    )


def _mcnemar_exact_p(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    from scipy.stats import binomtest

    a = np.asarray(correct_a, dtype=bool)
    bb = np.asarray(correct_b, dtype=bool)
    b = int((a & ~bb).sum())
    c = int((~a & bb).sum())
    nd = b + c
    if nd == 0:
        return 1.0
    return float(binomtest(min(b, c), nd, 0.5, alternative="two-sided").pvalue)


def _paired_bootstrap_ci(correct_a: np.ndarray, correct_b: np.ndarray,
                         n_boot: int = 1000, seed: int = 0):
    a = np.asarray(correct_a, dtype=float)
    bb = np.asarray(correct_b, dtype=float)
    n = len(a)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = a[idx].mean() - bb[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def _holm_bonferroni(pairs):
    order = sorted(pairs, key=lambda kv: kv[1])
    m = len(order)
    adj = {}
    prev = 0.0
    for i, (k, p) in enumerate(order):
        a = min(1.0, (m - i) * p)
        a = max(a, prev)
        adj[k] = a
        prev = a
    return adj


def _nadeau_bengio_t(diffs, n_train: int, n_test: int):
    from scipy.stats import t as student_t
    diffs = np.asarray(diffs, dtype=float)
    J = len(diffs)
    if J < 2:
        return float("nan"), float("nan")
    mean_d = float(diffs.mean())
    var = float(diffs.var(ddof=1))
    corr = (1.0 / J) + (n_test / n_train if n_train > 0 else 0.0)
    se = np.sqrt(var * corr) if var > 0 else 0.0
    if se == 0:
        return float("nan"), (0.0 if abs(mean_d) > 0 else 1.0)
    tstat = mean_d / se
    p = float(2.0 * student_t.sf(abs(tstat), J - 1))
    return float(tstat), p


def run_tuned(cfg: BPTDConfig) -> str:
    from copy import deepcopy
    from scipy.stats import wilcoxon

    base = load_td_data(os.path.join(cfg.data.folder, cfg.data.name), cfg.data.max_questions)
    n_splits, val_frac = 40, 0.5
    BP_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    CV_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]
    TF_GRID = [0.1, 0.3, 0.5, 1.0]
    acc = lambda d, p: M.accuracy(p, d.gold)

    def bp(mode, th=0.5):
        c = deepcopy(cfg.bp); c.dep_mode = mode; c.dependence_aware = (mode != "none")
        c.dep_threshold = th
        return lambda d: run_bptd(d, c).pred

    methods = {
        "MajorityVote":   {"-": lambda d: B.majority_vote(d).pred},
        "WeightedMV":     {"-": lambda d: B.weighted_mv(d).pred},
        "DawidSkene":     {"-": lambda d: B.dawid_skene(d).pred},
        "SSTF-VI":        {"-": lambda d: B.sstf_vi(d).pred},
        "SSTF-IV":        {"-": lambda d: B.sstf_iv(d).pred},
        "CRH":            {"-": lambda d: B.crh(d).pred},
        "CATD":           {"-": lambda d: B.catd(d).pred},
        "LTM":            {"-": lambda d: B.ltm(d).pred},
        "Investment":     {"-": lambda d: B.investment(d).pred},
        "Sums":           {"-": lambda d: B.sums(d).pred},
        "ACCU":           {"-": lambda d: B.accu_vote(d).pred},
        "DEPEN":          {"-": lambda d: B.depen_accu(d).pred},
        "PochampallyCorr":{"-": lambda d: B.pochampally_corr(d).pred},
        "IsingMF":        {"-": lambda d: B.ising_meanfield(d).pred},
        "TruthFinder":    {g: (lambda gg: (lambda d: B.truthfinder(d, gamma=gg).pred))(g) for g in TF_GRID},
        "ClusterVote":    {t: (lambda tt: (lambda d: B.cluster_vote(d, threshold=tt).pred))(t) for t in CV_GRID},
        "BP-TD-none":     {"-": bp("none")},
        "BP-TD-cooccur":  {t: bp("cooccur", t) for t in BP_GRID},
        "BP-TD-family":   {"-": bp("family")},
    }
    res = {k: [] for k in methods}
    res["BestAgent-valsel"] = []
    res["BestAgent-oracle"] = []

    for s in range(n_splits):
        rng = np.random.default_rng(7000 + s)
        perm = rng.permutation(base.n_questions)
        cut = int(round(val_frac * base.n_questions))
        val, test = _td_subset(base, perm[:cut]), _td_subset(base, perm[cut:])
        for name, grid in methods.items():
            if len(grid) == 1:
                pred = next(iter(grid.values()))(test)
            else:
                best = max(grid, key=lambda k: acc(val, grid[k](val)))
                pred = grid[best](test)
            res[name].append(acc(test, pred))
        va = val.measured_agent_accuracy()
        va = np.where(np.isnan(va), -1.0, va)
        best_a = int(np.random.default_rng(9500 + s).choice(np.flatnonzero(va == va.max())))
        mv_test = B.majority_vote(test).pred
        p = test.answers[best_a].copy(); p[p < 0] = mv_test[p < 0]
        res["BestAgent-valsel"].append(acc(test, p))
        res["BestAgent-oracle"].append(acc(test, B.best_single_agent(test).pred))

    rng0 = np.random.default_rng(0)
    perm0 = rng0.permutation(base.n_questions)
    cut0 = int(round(val_frac * base.n_questions))
    val0, test0 = _td_subset(base, perm0[:cut0]), _td_subset(base, perm0[cut0:])
    tgold = test0.gold
    tvalid = tgold >= 0
    correct = {}
    chosen = {}
    for name, grid in methods.items():
        if len(grid) == 1:
            best = "-"
            pred = next(iter(grid.values()))(test0)
        else:
            best = max(grid, key=lambda k: acc(val0, grid[k](val0)))
            pred = grid[best](test0)
        chosen[name] = best
        correct[name] = (np.asarray(pred)[tvalid] == tgold[tvalid])

    bp_variants = ["BP-TD-cooccur", "BP-TD-family", "BP-TD-none"]
    best_bp = max(bp_variants, key=lambda k: correct[k].mean())

    comp_methods = [k for k in methods if k != best_bp]
    raw_p = {}
    boot_ci = {}
    for k in comp_methods:
        raw_p[k] = _mcnemar_exact_p(correct[best_bp], correct[k])
        boot_ci[k] = _paired_bootstrap_ci(correct[best_bp], correct[k],
                                           n_boot=1000, seed=0)
    holm = _holm_bonferroni(list(raw_p.items()))

    mv = np.array(res["MajorityVote"])
    rows = []
    for k in sorted(res, key=lambda k: -np.mean(res[k])):
        v = np.array(res[k])
        nb_t, nb_p = ((float("nan"), float("nan")) if k == "MajorityVote"
                      else _nadeau_bengio_t(v - mv, cut0, base.n_questions - cut0))
        try:
            _, pval = wilcoxon(v, mv) if k != "MajorityVote" else (0.0, 1.0)
        except ValueError:
            pval = 1.0
        is_bp_ref = (k == best_bp)
        mc = raw_p.get(k, float("nan"))
        hp = holm.get(k, float("nan"))
        ci = boot_ci.get(k, (float("nan"), float("nan")))
        rows.append(dict(
            method=k, mean_acc=round(float(v.mean()), 4),
            sd_acc=round(float(v.std()), 4),
            diff_vs_mv=round(float((v - mv).mean()), 4),
            nb_t_vs_mv=(round(nb_t, 3) if nb_t == nb_t else float("nan")),
            nb_p_vs_mv=(round(nb_p, 4) if nb_p == nb_p else float("nan")),
            wilcoxon_p_vs_mv=round(float(pval), 4),
            tuned=("yes" if len(methods.get(k, {"-": 0})) > 1 else "no"),
            chosen_hp=str(chosen.get(k, "-")),
            split_acc=round(float(correct[k].mean()), 4) if k in correct else float("nan"),
            mcnemar_p=("ref" if is_bp_ref else round(float(mc), 4)),
            holm_p=("ref" if is_bp_ref else round(float(hp), 4)),
            boot_ci_low=("ref" if is_bp_ref else round(float(ci[0]), 4)),
            boot_ci_high=("ref" if is_bp_ref else round(float(ci[1]), 4)),
            sig_ref=best_bp,
            n_splits=n_splits))
    out = os.path.join(_out_dir(cfg), f"tuned_compare_{cfg.data.name}.csv")
    _write_csv(out, rows)
    print(f"\n  === Tuned val/test comparison on {cfg.data.name} "
          f"({n_splits} splits, val_frac={val_frac}; sig vs {best_bp} on fixed split) ===")
    print(f"  {'method':<18}{'acc':>7}{'±sd':>7}{'ΔvsMV':>8}{'wilc':>7}"
          f"{'McN':>8}{'Holm':>8}{'CI':>16}")
    for r in rows:
        ci = (f"[{r['boot_ci_low']},{r['boot_ci_high']}]"
              if r['boot_ci_low'] != "ref" else "ref")
        mc = r['mcnemar_p'] if r['mcnemar_p'] == "ref" else f"{r['mcnemar_p']:.3f}"
        hp = r['holm_p'] if r['holm_p'] == "ref" else f"{r['holm_p']:.3f}"
        print(f"  {r['method']:<18}{r['mean_acc']:>7.3f}{r['sd_acc']:>7.3f}"
              f"{r['diff_vs_mv']:>+8.3f}{r['wilcoxon_p_vs_mv']:>7.3f}"
              f"{mc:>8}{hp:>8}{ci:>16}")
    return out


def run_reliability(cfg: BPTDConfig) -> str:
    from copy import deepcopy

    base = load_td_data(os.path.join(cfg.data.folder, cfg.data.name), cfg.data.max_questions)
    true_rel = base.measured_agent_accuracy()

    bpc = deepcopy(cfg.bp)
    bpc.dep_mode = "cooccur"; bpc.dependence_aware = True
    rel_methods = {
        "BP-TD-cooccur": lambda d: run_bptd(d, bpc).reliability,
        "DawidSkene":    lambda d: B.dawid_skene(d).reliability,
        "CRH":           lambda d: B.crh(d).reliability,
        "AgreementFreq": lambda d: B.majority_vote(d).reliability,
    }

    def boot_ci_agents(est, true, n_boot=1000, seed=0):
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

    rows = []
    for name, fn in rel_methods.items():
        est = fn(base)
        rho = M.spearman(est, true_rel)
        lo, hi = boot_ci_agents(est, true_rel)
        rows.append(dict(
            dataset=cfg.data.name, method=name,
            spearman=round(float(rho), 4),
            ci_low=round(lo, 4), ci_high=round(hi, 4),
            n_agents=int(np.isfinite(true_rel).sum()),
        ))
        print(f"  {cfg.data.name} | {name:<14} rho={rho:.3f} "
              f"95%CI=[{lo:.3f},{hi:.3f}]")
    out = os.path.join(_out_dir(cfg), f"reliability_{cfg.data.name}.csv")
    _write_csv(out, rows)
    return out


def _disagreement_per_question(data: TDData) -> np.ndarray:
    n = data.n_questions
    out = np.zeros(n)
    for q in range(n):
        col = data.answers[:, q]
        v = col[col >= 0]
        if len(v) == 0:
            out[q] = float("nan"); continue
        _, counts = np.unique(v, return_counts=True)
        out[q] = 1.0 - counts.max() / len(v)
    return out


def run_disagreement(cfg: BPTDConfig) -> str:
    from copy import deepcopy

    base = load_td_data(os.path.join(cfg.data.folder, cfg.data.name), cfg.data.max_questions)
    gold = base.gold
    bpc = deepcopy(cfg.bp); bpc.dep_mode = "cooccur"; bpc.dependence_aware = True

    bp_pred = run_bptd(base, bpc).pred
    mv_pred = B.majority_vote(base).pred
    bp_corr = (bp_pred == gold).astype(float)
    mv_corr = (mv_pred == gold).astype(float)
    disag = _disagreement_per_question(base)

    rows = []
    valid = gold >= 0
    rows.append(dict(
        label=cfg.data.name,
        disagreement=round(float(np.nanmean(disag[valid])), 4),
        bp_td_delta_vs_mv=round(float((bp_corr[valid] - mv_corr[valid]).mean()), 4),
        n_questions=int(valid.sum()),
    ))

    dv = disag[valid]; bc = bp_corr[valid]; mc = mv_corr[valid]
    order = np.argsort(dv, kind="stable")
    nq = len(dv)
    bin_id = np.empty(nq, dtype=int)
    for rank, idx in enumerate(order):
        bin_id[idx] = min(3, rank * 4 // nq)
    for b in range(4):
        mask = bin_id == b
        if mask.sum() == 0:
            continue
        rows.append(dict(
            label=f"{cfg.data.name}-disag-Q{b+1}",
            disagreement=round(float(dv[mask].mean()), 4),
            bp_td_delta_vs_mv=round(float((bc[mask] - mc[mask]).mean()), 4),
            n_questions=int(mask.sum()),
        ))

    out = os.path.join(_out_dir(cfg), "gain_vs_disagreement.csv")
    fields = set(rows[0].keys())
    existing = []
    if os.path.isfile(out):
        with open(out, newline="") as f:
            for r in csv.DictReader(f):
                if set(r.keys()) == fields:
                    existing.append(r)
    new_labels = {r["label"] for r in rows}
    keep = [r for r in existing if r["label"] not in new_labels
            and not r["label"].startswith(f"{cfg.data.name}-disag")]
    merged = keep + rows
    _write_csv(out, merged)
    for r in rows:
        print(f"  {r['label']:<18} disag={r['disagreement']:.3f} "
              f"BP-TD Δ vs MV={r['bp_td_delta_vs_mv']:+.4f}")
    return out


def run_c2(cfg: BPTDConfig) -> str:
    out = os.path.join(_out_dir(cfg), f"c2_reliability_{cfg.data.name}.csv")
    rows = []
    rel_methods = {
        "BP-TD": lambda d: run_bptd(d, cfg.bp).reliability,
        "AgreementFreq": lambda d: B.majority_vote(d).reliability,
        "DawidSkene": lambda d: B.dawid_skene(d).reliability,
        "CRH": lambda d: B.crh(d).reliability,
    }
    for seed in cfg.seeds:
        data = data_for_seed(cfg, seed)
        true_rel = data.measured_agent_accuracy()
        for name, fn in rel_methods.items():
            est = fn(data)
            rows.append(dict(
                dataset=cfg.data.name, method=name, seed=seed,
                spearman=round(M.spearman(est, true_rel), 4),
                kendall=round(M.kendall(est, true_rel), 4),
                ndcg=round(M.ndcg_at_k(est, true_rel), 4),
            ))
            print(f"  {cfg.data.name} | seed {seed:>4} | {name:<14} "
                  f"rho={rows[-1]['spearman']} tau={rows[-1]['kendall']}")
    _write_csv(out, rows)
    return out


def run_c3(cfg: BPTDConfig) -> str:
    out = os.path.join(_out_dir(cfg), f"c3_signed_ablation_{cfg.data.name}.csv")
    rows = []
    from copy import deepcopy
    for seed in cfg.seeds:
        data = data_for_seed(cfg, seed)
        true_rel = data.measured_agent_accuracy()
        for signed in (True, False):
            bp = deepcopy(cfg.bp); bp.signed = signed
            res = run_bptd(data, bp)
            rows.append(dict(
                dataset=cfg.data.name, seed=seed, signed=signed,
                accuracy=round(M.accuracy(res.pred, data.gold), 4),
                macro_f1=round(M.macro_f1(res.pred, data.gold), 4),
                spearman_rel=round(M.spearman(res.reliability, true_rel), 4),
            ))
            print(f"  {cfg.data.name} | seed {seed:>4} | signed={signed!s:<5} "
                  f"acc={rows[-1]['accuracy']} rho={rows[-1]['spearman_rel']}")
    _write_csv(out, rows)
    return out


def run_c4(cfg: BPTDConfig) -> str:
    out = os.path.join(_out_dir(cfg), f"c4_seeding_{cfg.data.name}.csv")
    rows = []
    from copy import deepcopy
    for seed in cfg.seeds:
        data = data_for_seed(cfg, seed)
        gmask = np.where(data.gold_mask())[0]
        rng = np.random.default_rng(cfg.bp.seed_rng + seed)
        for k in (0.0, 0.01, 0.05, 0.10, 0.20):
            n_seed = int(round(k * len(gmask)))
            seed_idx = rng.choice(gmask, size=n_seed, replace=False) if n_seed else np.array([], int)
            eval_mask = np.ones(data.n_questions, bool)
            eval_mask[seed_idx] = False
            bp = deepcopy(cfg.bp)
            res = run_bptd(data, bp, seed_idx=seed_idx)
            rows.append(dict(
                dataset=cfg.data.name, seed=seed, seed_fraction=k,
                n_seeded=n_seed,
                accuracy=round(M.accuracy(res.pred, data.gold, eval_mask), 4),
            ))
            print(f"  {cfg.data.name} | seed {seed:>4} | k={k:<5} "
                  f"acc(unseeded)={rows[-1]['accuracy']}")
    _write_csv(out, rows)
    return out


def run_c5(cfg: BPTDConfig) -> str:
    from copy import deepcopy
    import time
    out_conv = os.path.join(_out_dir(cfg), f"c5_convergence_{cfg.data.name}.csv")
    out_scale = os.path.join(_out_dir(cfg), "c5_scaling_synthetic.csv")
    conv_rows = []
    for seed in cfg.seeds:
        data = data_for_seed(cfg, seed)
        for mode in ("loopy", "onepass"):
            bp = deepcopy(cfg.bp); bp.mode = mode
            res = run_bptd(data, bp)
            conv_rows.append(dict(
                dataset=cfg.data.name, seed=seed, mode=mode,
                n_iter=res.n_iter, converged=res.converged,
                final_residual=round(res.extra.get("final_residual", float("nan")), 6),
                accuracy=round(M.accuracy(res.pred, data.gold), 4),
                runtime_s=round(res.runtime_s, 3),
            ))
    _write_csv(out_conv, conv_rows)
    scale_rows = []
    for nq in (100, 500, 1000, 5000):
        for na in (4, 8, 16, 32):
            data = make_synthetic(n_questions=nq, n_agents=na, n_values=4, seed=7)
            t0 = time.time()
            res = run_bptd(data, cfg.bp)
            scale_rows.append(dict(
                n_questions=nq, n_agents=na, nm=nq * na,
                runtime_s=round(time.time() - t0, 3), n_iter=res.n_iter,
                accuracy=round(M.accuracy(res.pred, data.gold), 4),
            ))
            print(f"  scaling n={nq:>5} m={na:>2} -> {scale_rows[-1]['runtime_s']}s")
    _write_csv(out_scale, scale_rows)
    return out_conv


def _write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[bptd] wrote {path}  ({len(rows)} rows)")


def _print_summary(rows: List[dict]) -> None:
    by_method: dict = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r["accuracy"])
    print(f"\n  {'method':<18}{'mean acc':>10}{'± sd':>8}")
    for method, accs in sorted(by_method.items(), key=lambda kv: -np.nanmean(kv[1])):
        print(f"  {method:<18}{np.nanmean(accs):>10.4f}{np.nanstd(accs):>8.4f}")


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--write-config" in argv:
        path = write_default_yaml("conf/bptd.yaml")
        print(f"Wrote default config to {path}")
        return
    overrides = [a for a in argv if "=" in a and not a.startswith("--")]
    cfg = apply_overrides(default_config(), overrides)

    dispatch = {
        "demo": run_demo, "compare": run_compare, "tuned": run_tuned, "c2": run_c2,
        "c3": run_c3, "c4": run_c4, "c5": run_c5,
        "reliability": run_reliability, "disagreement": run_disagreement,
    }
    if cfg.mode == "harvest":
        if getattr(cfg.agents, "backend", "ollama") == "vllm":
            from bptd.harvest_vllm import harvest_vllm
            harvest_vllm(cfg.data, cfg.agents)
        else:
            from bptd.harvest import harvest
            harvest(cfg.data, cfg.agents)
        return
    fn = dispatch.get(cfg.mode)
    if fn is None:
        print(f"Unknown mode {cfg.mode!r}. Choose from: harvest, {', '.join(dispatch)}")
        sys.exit(2)
    fn(cfg)


if __name__ == "__main__":
    main()
