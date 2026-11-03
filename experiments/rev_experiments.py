# Author: Laure Berti-Equille (laure.berti@ird.fr)
# Paper:  Berti-Equille, L. (2026). Where Is the Truth When AI Agents Disagree?
#         Dependence-Aware Belief Propagation for Unsupervised Truth Discovery.
#         Paper accepted at IEEE BigData 2026.
from __future__ import annotations

import itertools
import os
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from bptd import baselines as B
from bptd import metrics as M
from bptd.bp import run_bptd
from bptd.config import BPConfig
from bptd.data import TDData, load_td_data, make_synthetic, signed_agreement_matrix


@dataclass
class RevConfig:
    exp: str = "main"
    datasets: List[str] = field(default_factory=lambda: ["mmlu", "sciq", "truthfulqa", "arc", "boolq"])
    data_dir: str = os.path.join(ROOT, "data")
    out_dir: str = os.path.join(ROOT, "outputs")
    n_splits: int = 40
    val_frac: float = 0.5
    split_seed0: int = 7000
    theta_grid: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    theta_sweep: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    fixed_theta: float = 0.5
    pool_sizes: List[int] = field(default_factory=lambda: [5, 7, 9, 11])
    pools_per_size: int = 100
    n_seeds: int = 20
    budgets: List[int] = field(default_factory=lambda: [0, 2, 5, 10, 20, 40, 80])
    n_boot: int = 1000
    scale_datasets: List[str] = field(default_factory=lambda: ["mmlu_vllm_5k", "mmlu_vllm_10k"])
    scale_splits: int = 10


ConfigStore.instance().store(name="rev", node=RevConfig)

BASE_OF = {
    "direct:qwen2.5:14b": "qwen2.5", "direct:qwen2.5:7b": "qwen2.5", "sc:qwen2.5:7b": "qwen2.5",
    "rag:qwen2.5:7b": "qwen2.5", "direct:mistral:7b": "mistral", "direct:phi3.5:latest": "phi3.5",
    "direct:gemma2:2b": "gemma2", "direct:llama3.2:latest": "llama3.2", "sc:llama3.2:latest": "llama3.2",
    "rag:llama3.2:latest": "llama3.2", "persona:expert": "llama3.2", "persona:skeptic": "llama3.2",
    "persona:cautious": "llama3.2", "openclaw": "openclaw",
}


def load(cfg: RevConfig, name: str) -> TDData:
    d = load_td_data(os.path.join(cfg.data_dir, name))
    if not d.base_model or d.base_model == list(d.agents):
        d.base_model = [BASE_OF.get(a, a) for a in d.agents]
    return d


def answering(d: TDData) -> List[str]:
    return [a for i, a in enumerate(d.agents) if (d.answers[i] >= 0).any()]


def subset_q(d: TDData, idx) -> TDData:
    idx = np.asarray(idx)
    return TDData(questions=[d.questions[i] for i in idx], agents=list(d.agents), domains=d.domains,
                  answers=d.answers[:, idx].copy(), gold=d.gold[idx].copy(), base_model=d.base_model)


def otsu_theta(d: TDData) -> float:
    W = signed_agreement_matrix(d)
    act = np.where((d.answers >= 0).any(axis=1))[0]
    v = np.sort(W[np.ix_(act, act)][np.triu_indices(len(act), 1)])
    best_t, best_s = 0.5, -1.0
    for t in np.unique(v)[1:]:
        lo, hi = v[v < t], v[v >= t]
        s = len(lo) * len(hi) * (lo.mean() - hi.mean()) ** 2
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t


def bp(mode: str, dep: str, theta: float = 0.5, **kw) -> Callable[[TDData], np.ndarray]:
    c = BPConfig()
    c.mode, c.dep_mode, c.dependence_aware, c.dep_threshold = mode, dep, dep != "none", theta
    for k, v in kw.items():
        setattr(c, k, v)
    return lambda d: run_bptd(d, c).pred


def bp_otsu(mode: str) -> Callable[[TDData], np.ndarray]:
    return lambda d: bp(mode, "cooccur", otsu_theta(d))(d)


def acc(d: TDData, pred) -> float:
    return M.accuracy(np.asarray(pred), d.gold)


def nb_t(diffs, n_train, n_test):
    from scipy.stats import t as student_t
    diffs = np.asarray(diffs, float)
    J = len(diffs)
    var = diffs.var(ddof=1)
    se = np.sqrt(var * (1.0 / J + n_test / n_train)) if var > 0 else 0.0
    if se == 0:
        return float("nan"), (0.0 if abs(diffs.mean()) > 0 else 1.0)
    t = diffs.mean() / se
    return float(t), float(2 * student_t.sf(abs(t), J - 1))


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted((p, k) for k, p in pvals.items() if p == p)
    out, run, m = {}, 0.0, len(items)
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def write(cfg: RevConfig, name: str, rows: List[dict]) -> str:
    import csv
    os.makedirs(cfg.out_dir, exist_ok=True)
    path = os.path.join(cfg.out_dir, name)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[rev] wrote {path} ({len(rows)} rows)")
    return path


def label_free_methods(theta: float) -> Dict[str, Callable]:
    return {
        "MajorityVote": lambda d: B.majority_vote(d).pred,
        "WVote-family": bp("onepass", "family"),
        f"WVote-cooccur@{theta}": bp("onepass", "cooccur", theta),
        "WVote-cooccur-otsu": bp_otsu("onepass"),
        "BP-TD-family": bp("loopy", "family"),
        f"BP-TD-cooccur@{theta}": bp("loopy", "cooccur", theta),
        "BP-TD-cooccur-otsu": bp_otsu("loopy"),
        "BP-TD-none": bp("loopy", "none"),
        f"ClusterVote@{theta}": lambda d: B.cluster_vote(d, threshold=theta).pred,
    }


def exp_main(cfg: RevConfig, datasets=None, n_splits=None, reduced=False) -> None:
    th = cfg.fixed_theta
    grid = list(cfg.theta_grid)
    for name in datasets or cfg.datasets:
        base = load(cfg, name)
        tuned = {
            "BP-TD-cooccur": {t: bp("loopy", "cooccur", t) for t in grid},
            "WVote-cooccur": {t: bp("onepass", "cooccur", t) for t in grid},
            "ClusterVote": {t: (lambda tt: (lambda d: B.cluster_vote(d, threshold=tt).pred))(t)
                            for t in [0.3, 0.4, 0.5, 0.6, 0.7]},
        }
        fixed = label_free_methods(th)
        if not reduced:
            tuned["TruthFinder"] = {g: (lambda gg: (lambda d: B.truthfinder(d, gamma=gg).pred))(g)
                                    for g in [0.1, 0.3, 0.5, 1.0]}
            fixed.update({
                "WeightedMV": lambda d: B.weighted_mv(d).pred,
                "DawidSkene": lambda d: B.dawid_skene(d).pred,
                "LTM": lambda d: B.ltm(d).pred,
                "CATD": lambda d: B.catd(d).pred,
                "CRH": lambda d: B.crh(d).pred,
                "Investment": lambda d: B.investment(d).pred,
                "Sums": lambda d: B.sums(d).pred,
                "SSTF-IV": lambda d: B.sstf_iv(d).pred,
                "SSTF-VI": lambda d: B.sstf_vi(d).pred,
                "ACCU": lambda d: B.accu_vote(d).pred,
                "DEPEN": lambda d: B.depen_accu(d).pred,
                "PochampallyCorr": lambda d: B.pochampally_corr(d).pred,
                "IsingMF": lambda d: B.ising_meanfield(d).pred,
            })
        res: Dict[str, List[float]] = {k: [] for k in list(tuned) + list(fixed)}
        res["BestAgent-valsel"] = []
        chosen: Dict[str, List] = {k: [] for k in tuned}
        otsu_vals, niters = [], {"BP-TD-cooccur": [], "BP-TD-family": [], "BP-TD-none": []}
        J = n_splits or cfg.n_splits
        cut = int(round(cfg.val_frac * base.n_questions))
        for s in range(J):
            perm = np.random.default_rng(cfg.split_seed0 + s).permutation(base.n_questions)
            val, test = subset_q(base, perm[:cut]), subset_q(base, perm[cut:])
            for k, g in tuned.items():
                best = max(g, key=lambda t: acc(val, g[t](val)))
                chosen[k].append(best)
                res[k].append(acc(test, g[best](test)))
            for k, f in fixed.items():
                res[k].append(acc(test, f(test)))
            otsu_vals.append(otsu_theta(test))
            for mode_name, dep in [("BP-TD-cooccur", "cooccur"), ("BP-TD-family", "family"),
                                   ("BP-TD-none", "none")]:
                c = BPConfig(); c.dep_mode = dep; c.dependence_aware = dep != "none"
                c.dep_threshold = chosen["BP-TD-cooccur"][-1]
                niters[mode_name].append(run_bptd(test, c).n_iter)
            va = val.measured_agent_accuracy()
            va = np.where(np.isnan(va), -1, va)
            a = int(np.random.default_rng(9500 + s).choice(np.flatnonzero(va == va.max())))
            p = test.answers[a].copy(); mv = B.majority_vote(test).pred; p[p < 0] = mv[p < 0]
            res["BestAgent-valsel"].append(acc(test, p))
        mv = np.array(res["MajorityVote"])
        ref = np.array(res["BP-TD-cooccur"])
        p_mv = {k: nb_t(np.array(v) - mv, cut, base.n_questions - cut)[1] for k, v in res.items()
                if k != "MajorityVote"}
        p_ref = {k: nb_t(ref - np.array(v), cut, base.n_questions - cut)[1] for k, v in res.items()
                 if k != "BP-TD-cooccur"}
        h_mv, h_ref = holm(p_mv), holm(p_ref)
        rows = []
        for k in sorted(res, key=lambda k: -np.mean(res[k])):
            v = np.array(res[k])
            rows.append(dict(
                dataset=name, method=k, mean_acc=round(v.mean(), 4), sd_acc=round(v.std(ddof=1), 4),
                diff_vs_mv=round((v - mv).mean(), 4),
                nb_t_vs_mv=round(nb_t(v - mv, cut, base.n_questions - cut)[0], 3) if k != "MajorityVote" else "",
                nb_p_vs_mv=round(p_mv.get(k, float("nan")), 4), holm_p_vs_mv=round(h_mv.get(k, float("nan")), 4),
                diff_vs_bptd=round((ref - v).mean(), 4),
                nb_p_bptd_vs=round(p_ref.get(k, float("nan")), 4),
                holm_p_bptd_vs=round(h_ref.get(k, float("nan")), 4),
                mode=("one-pass (T=1)" if k.startswith("WVote") else
                      "loopy" if k.startswith("BP-TD") else ""),
                mean_n_iter=(round(float(np.mean(niters[k])), 1) if k in niters else ""),
                chosen_theta=(dict(Counter(chosen[k]).most_common(3)) if k in chosen else ""),
                n_splits=J, n_test=base.n_questions - cut))
        rows.append(dict(dataset=name, method="_otsu_theta", mean_acc=round(float(np.mean(otsu_vals)), 3),
                         sd_acc=round(float(np.std(otsu_vals)), 3)))
        write(cfg, f"E1_main_{name}.csv", rows)
        for r in rows[:30]:
            print(f"  {name:<11}{r['method']:<24}{r['mean_acc']:>7}  Δmv={r.get('diff_vs_mv','')}"
                  f"  p_mv={r.get('holm_p_vs_mv','')}  p_vsBP={r.get('holm_p_bptd_vs','')}"
                  f"  T={r.get('mean_n_iter','')}")


def boot_ci(correct: np.ndarray, n_boot: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(correct)
    vals = [correct[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    return np.percentile(vals, [2.5, 97.5])


def exp_theta(cfg: RevConfig) -> None:
    rows = []
    for name in cfg.datasets:
        d = load(cfg, name)
        gm = d.gold >= 0
        mv = B.majority_vote(d).pred
        ot = otsu_theta(d)
        for meth, f in [("MajorityVote", lambda dd, t: mv), ("BP-TD-family", lambda dd, t: bp("loopy", "family")(dd)),
                        ("WVote-family", lambda dd, t: bp("onepass", "family")(dd))]:
            c = (np.asarray(f(d, None))[gm] == d.gold[gm]).astype(float)
            lo, hi = boot_ci(c, cfg.n_boot)
            rows.append(dict(dataset=name, method=meth, theta="", acc=round(c.mean(), 4),
                             ci_lo=round(lo, 4), ci_hi=round(hi, 4), otsu_theta=round(ot, 3)))
        for t in cfg.theta_sweep:
            for meth, f in [("BP-TD-cooccur", bp("loopy", "cooccur", t)), ("WVote-cooccur", bp("onepass", "cooccur", t)),
                            ("ClusterVote", lambda dd, tt=t: B.cluster_vote(dd, threshold=tt).pred)]:
                c = (np.asarray(f(d))[gm] == d.gold[gm]).astype(float)
                lo, hi = boot_ci(c, cfg.n_boot)
                rows.append(dict(dataset=name, method=meth, theta=t, acc=round(c.mean(), 4),
                                 ci_lo=round(lo, 4), ci_hi=round(hi, 4), otsu_theta=round(ot, 3)))
        for meth, f in [("BP-TD-cooccur-otsu", bp("loopy", "cooccur", ot)),
                        ("WVote-cooccur-otsu", bp("onepass", "cooccur", ot))]:
            c = (np.asarray(f(d))[gm] == d.gold[gm]).astype(float)
            lo, hi = boot_ci(c, cfg.n_boot)
            rows.append(dict(dataset=name, method=meth, theta=round(ot, 3), acc=round(c.mean(), 4),
                             ci_lo=round(lo, 4), ci_hi=round(hi, 4), otsu_theta=round(ot, 3)))
        print(f"  {name}: otsu θ={ot:.3f}")
    write(cfg, "E2_theta.csv", rows)


def pool_stats(d: TDData) -> dict:
    fam = Counter(d.base_model)
    big = max(fam, key=lambda f: (fam[f], f))
    inb = np.array([b == big for b in d.base_model])
    agree, flips = [], []
    for q in range(d.n_questions):
        a_in, a_out = d.answers[inb, q], d.answers[~inb, q]
        a_in, a_out = a_in[a_in >= 0], a_out[a_out >= 0]
        if len(a_in) and len(a_out):
            agree.append(Counter(a_in).most_common(1)[0][0] == Counter(a_out).most_common(1)[0][0])
    mv = B.majority_vote(d).pred
    fam_pred = bp("onepass", "family")(d)
    return dict(S_bloc_vs_rest=float(np.mean(agree)) if agree else float("nan"),
                flip_rate=float(np.mean(mv != fam_pred)), largest_family=big, largest_size=fam[big],
                n_families=len(fam))


def eval_pool(d: TDData, th: float) -> dict:
    out = {}
    for k, f in label_free_methods(th).items():
        out[k] = acc(d, f(d))
    return out


def exp_subpools(cfg: RevConfig) -> None:
    rows = []
    th = cfg.fixed_theta
    for name in cfg.datasets:
        full = load(cfg, name)
        ag = answering(full)
        full = full.subset_agents(ag)
        fam_of = dict(zip(full.agents, full.base_model))
        true_acc = dict(zip(full.agents, full.measured_agent_accuracy()))
        fam_acc = {f: np.mean([true_acc[a] for a in ag if fam_of[a] == f]) for f in set(fam_of.values())}
        rng = np.random.default_rng(123)
        pools = []
        from math import comb
        for s in cfg.pool_sizes:
            seen = set()
            target = min(cfg.pools_per_size, comb(len(ag), s))
            while len(seen) < target:
                p = tuple(sorted(rng.choice(len(ag), size=s, replace=False)))
                if p not in seen:
                    seen.add(p); pools.append(("random", [ag[i] for i in p]))
        qwen = [a for a in ag if fam_of[a] == "qwen2.5"]
        llama = [a for a in ag if fam_of[a] == "llama3.2"]
        singles = [a for a in ag if fam_of[a] not in ("qwen2.5", "llama3.2")]
        for j in range(0, min(3, len(llama)) + 1):
            for L in itertools.combinations(llama, j):
                for r in range(0, len(singles) + 1):
                    for S in itertools.combinations(singles, r):
                        if len(qwen) + j + r >= 5:
                            pools.append(("strong-bloc-largest", qwen + list(L) + list(S)))
        print(f"  {name}: {len(pools)} pools", flush=True)
        for i, (kind, members) in enumerate(pools):
            if i % 100 == 0:
                print(f"  {name}: pool {i}/{len(pools)}", flush=True)
            d = full.subset_agents(members)
            fam = Counter(d.base_model)
            big = max(fam, key=lambda f: (fam[f], f))
            strongest = max(fam, key=lambda f: fam_acc[f])
            r = dict(dataset=name, kind=kind, size=len(members), members="|".join(members),
                     largest_is_strongest=bool(fam[big] > 1 and big == strongest and
                                               sum(1 for f in fam if fam[f] == fam[big]) == 1))
            r.update(eval_pool(d, th))
            r.update(pool_stats(d))
            rows.append(r)
    write(cfg, "E3_subpools.csv", rows)


def exp_strongbloc(cfg: RevConfig, eps: float = 0.1, ks=(0, 2, 4, 6, 8, 10)) -> None:
    rows = []
    th = cfg.fixed_theta
    for name in [x for x in cfg.datasets if x in ("mmlu", "arc", "truthfulqa")]:
        base = load(cfg, name)
        base = base.subset_agents(answering(base))
        src = base.agents.index("direct:qwen2.5:14b")
        for seed in range(cfg.n_seeds):
            rng = np.random.default_rng(seed)
            for k in ks:
                extra = []
                for j in range(k):
                    a = base.answers[src].copy()
                    for q in range(base.n_questions):
                        if a[q] >= 0 and rng.random() < eps:
                            dsz = base.domain_size(q)
                            w = rng.integers(0, dsz - 1); a[q] = w + (w >= a[q])
                    extra.append(a)
                d = TDData(questions=base.questions, agents=base.agents + [f"copy{j}" for j in range(k)],
                           domains=base.domains,
                           answers=np.vstack([base.answers] + extra) if extra else base.answers.copy(),
                           gold=base.gold, base_model=base.base_model + ["qwen2.5"] * k)
                r = dict(dataset=name, seed=seed, k=k, eps=eps)
                r.update(eval_pool(d, th))
                rows.append(r)
        print(f"  {name} done")
    write(cfg, "E4_strongbloc.csv", rows)


def exp_ds_sanity(cfg: RevConfig) -> None:
    rows = []
    meths = {
        "MajorityVote": lambda d: B.majority_vote(d).pred, "DawidSkene": lambda d: B.dawid_skene(d).pred,
        "LTM": lambda d: B.ltm(d).pred, "CATD": lambda d: B.catd(d).pred, "CRH": lambda d: B.crh(d).pred,
        "BP-TD-none": bp("loopy", "none"), "BP-TD-cooccur@0.5": bp("loopy", "cooccur", 0.5),
        "BP-TD-family": bp("loopy", "family"),
    }
    for n in (160, 1000):
        for seed in range(cfg.n_seeds):
            d = make_synthetic(n_questions=n, n_agents=13, n_values=4, seed=seed)
            d.base_model = list(d.agents)
            r = dict(setting="independent", n=n, seed=seed)
            r.update({k: acc(d, f(d)) for k, f in meths.items()})
            rows.append(r)
    d = load(cfg, "mmlu")
    inb = np.array([b == "llama3.2" for b in d.base_model])
    bloc = np.array([Counter(x[x >= 0]).most_common(1)[0][0] if (x >= 0).any() else -1
                     for x in d.answers[inb].T])
    for k in ("DawidSkene", "LTM", "MajorityVote", "BP-TD-none"):
        p = np.asarray(meths[k](d))
        rows.append(dict(setting="mmlu_full", n=d.n_questions, seed="", method=k, acc=round(acc(d, p), 4),
                         agree_with_llama_bloc=round(float((p == bloc).mean()), 4),
                         llama_bloc_acc=round(acc(d, bloc), 4)))
    write(cfg, "E5_ds_sanity.csv", rows)


def exp_budget(cfg: RevConfig) -> None:
    rows = []
    grid = list(cfg.theta_grid)
    for name in cfg.datasets:
        base = load(cfg, name)
        n = base.n_questions
        for k in cfg.budgets:
            for s in range(cfg.n_splits):
                perm = np.random.default_rng(9000 + s).permutation(n)
                lab, test = subset_q(base, perm[:k]), subset_q(base, perm[k:])
                r = dict(dataset=name, k=k, draw=s)
                r["MajorityVote"] = acc(test, B.majority_vote(test).pred)
                r["BP-TD-family"] = acc(test, bp("loopy", "family")(test))
                if k == 0:
                    r["BP-TD-cooccur(k-tuned)"] = acc(test, bp_otsu("loopy")(test))
                    r["BestAgent(k)"] = float("nan")
                else:
                    best = max(grid, key=lambda t: acc(lab, bp("loopy", "cooccur", t)(lab)))
                    r["BP-TD-cooccur(k-tuned)"] = acc(test, bp("loopy", "cooccur", best)(test))
                    va = lab.measured_agent_accuracy()
                    va = np.where(np.isnan(va), -1, va)
                    a = int(np.random.default_rng(9500 + s).choice(np.flatnonzero(va == va.max())))
                    p = test.answers[a].copy(); mv = B.majority_vote(test).pred; p[p < 0] = mv[p < 0]
                    r["BestAgent(k)"] = acc(test, p)
                rows.append(r)
        print(f"  {name} done")
    write(cfg, "E6_budget.csv", rows)


def exp_seeding(cfg: RevConfig) -> None:
    rows = []
    for name in cfg.datasets:
        base = load(cfg, name)
        gm = np.where(base.gold >= 0)[0]
        for frac in (0.0, 0.01, 0.05, 0.10, 0.20):
            for s in range(cfg.n_splits):
                rng = np.random.default_rng(5000 + s)
                ns = int(round(frac * len(gm)))
                seed_idx = rng.choice(gm, size=ns, replace=False) if ns else np.array([], int)
                mask = np.ones(base.n_questions, bool); mask[seed_idx] = False
                for meth, mode, dep in [("BP-TD-cooccur@0.5", "loopy", "cooccur"), ("BP-TD-family", "loopy", "family")]:
                    c = BPConfig(); c.mode = mode; c.dep_mode = dep; c.dep_threshold = 0.5
                    res = run_bptd(base, c, seed_idx=seed_idx)
                    rows.append(dict(dataset=name, method=meth, seed_fraction=frac, n_seeded=ns, draw=s,
                                     acc_unseeded=M.accuracy(res.pred, base.gold, mask),
                                     mv_unseeded=M.accuracy(B.majority_vote(base).pred, base.gold, mask)))
        print(f"  {name} done")
    write(cfg, "E8_seeding.csv", rows)


def exp_scale(cfg: RevConfig) -> None:
    exp_main(cfg, datasets=cfg.scale_datasets, n_splits=cfg.scale_splits, reduced=True)


def exp_agents(cfg: RevConfig) -> None:
    rows = []
    for name in cfg.datasets + list(cfg.scale_datasets):
        d = load(cfg, name)
        acc_a = d.measured_agent_accuracy()
        parsed = (d.answers >= 0).mean(axis=1)
        for a, b, x, p in zip(d.agents, d.base_model, acc_a, parsed):
            rows.append(dict(dataset=name, agent=a, base_model=b, accuracy=round(float(x), 4) if x == x else "",
                             answered_frac=round(float(p), 4), n_questions=d.n_questions))
    write(cfg, "E10_agents.csv", rows)


CARE_GRID = [0.1, 0.2, 0.25, 0.5, 0.75, 1, 2, 3, 5, 7, 10]


def exp_concurrent(cfg: RevConfig) -> None:
    import concurrent_methods as C
    for name in cfg.datasets:
        base = load(cfg, name)
        cut = int(round(cfg.val_frac * base.n_questions))
        res = {k: [] for k in ["MajorityVote", "BP-TD-cooccur", "CARE-SVD(pub,γ=1)", "CARE-SVD(pub,γ-val)",
                               "CARE-SVD(ridge,γ=1)", "CARE-SVD(ridge,γ-val)", "ISP", "OW-I"]}
        chosen = {"BP": [], "pub": [], "ridge": []}
        fail0 = len(C.CARE_FAILURES)
        for s in range(cfg.n_splits):
            perm = np.random.default_rng(cfg.split_seed0 + s).permutation(base.n_questions)
            val, test = subset_q(base, perm[:cut]), subset_q(base, perm[cut:])
            res["MajorityVote"].append(acc(test, B.majority_vote(test).pred))
            g = {t: bp("loopy", "cooccur", t) for t in cfg.theta_grid}
            bt = max(g, key=lambda t: acc(val, g[t](val))); chosen["BP"].append(bt)
            res["BP-TD-cooccur"].append(acc(test, g[bt](test)))
            for tag, rdg in (("pub", 0.0), ("ridge", 1e-3)):
                res[f"CARE-SVD({tag},γ=1)"].append(acc(test, C.care_svd(test, 1.0, rdg)))
                bg = max(CARE_GRID, key=lambda gm: acc(val, C.care_svd(val, gm, rdg)))
                chosen[tag].append(bg)
                res[f"CARE-SVD({tag},γ-val)"].append(acc(test, C.care_svd(test, bg, rdg)))
            res["ISP"].append(acc(test, C.isp(test)))
            res["OW-I"].append(acc(test, C.ow_i(test)))
        mv, ref = np.array(res["MajorityVote"]), np.array(res["BP-TD-cooccur"])
        p_mv = {k: nb_t(np.array(v) - mv, cut, base.n_questions - cut)[1] for k, v in res.items() if k != "MajorityVote"}
        p_ref = {k: nb_t(ref - np.array(v), cut, base.n_questions - cut)[1] for k, v in res.items() if k != "BP-TD-cooccur"}
        h_mv, h_ref = holm(p_mv), holm(p_ref)
        fails = C.CARE_FAILURES[fail0:]
        rows = []
        for k, v in res.items():
            v = np.array(v)
            rows.append(dict(dataset=name, method=k, mean_acc=round(v.mean(), 4), sd_acc=round(v.std(ddof=1), 4),
                             diff_vs_mv=round((v - mv).mean(), 4), holm_p_vs_mv=round(h_mv.get(k, float("nan")), 4),
                             diff_vs_bptd=round((ref - v).mean(), 4), holm_p_bptd_vs=round(h_ref.get(k, float("nan")), 4),
                             n_splits=cfg.n_splits))
        rows.append(dict(dataset=name, method="_care_failures(pub)", mean_acc=sum(1 for f in fails if f[1] == 0.0)))
        rows.append(dict(dataset=name, method="_care_failures(ridge)", mean_acc=sum(1 for f in fails if f[1] > 0)))
        rows.append(dict(dataset=name, method="_chosen", mean_acc=str({k: dict(Counter(v).most_common(3)) for k, v in chosen.items()})))
        write(cfg, f"E11_concurrent_{name}.csv", rows)


def exp_rho(cfg: RevConfig) -> None:
    rows = []
    for name in cfg.datasets:
        base = load(cfg, name)
        base = base.subset_agents(answering(base))
        cut = int(round(cfg.val_frac * base.n_questions))
        for s in range(cfg.n_splits):
            perm = np.random.default_rng(cfg.split_seed0 + s).permutation(base.n_questions)
            test = subset_q(base, perm[cut:])
            true = test.measured_agent_accuracy()
            r = dict(dataset=name, split=s)
            for k, (mode, dep, th) in {"BP-TD-cooccur@0.5": ("loopy", "cooccur", 0.5),
                                        "BP-TD-cooccur@0.7": ("loopy", "cooccur", 0.7),
                                        "BP-TD-family": ("loopy", "family", 0.5),
                                        "BP-TD-none": ("loopy", "none", 0.5)}.items():
                c = BPConfig(); c.mode, c.dep_mode, c.dependence_aware, c.dep_threshold = mode, dep, dep != "none", th
                r[k] = M.spearman(run_bptd(test, c).reliability, true)
            r["DawidSkene"] = M.spearman(B.dawid_skene(test).reliability, true)
            rows.append(r)
    write(cfg, "E12_rho_splits.csv", rows)


EXPS = dict(main=exp_main, theta=exp_theta, subpools=exp_subpools, strongbloc=exp_strongbloc,
            ds_sanity=exp_ds_sanity, budget=exp_budget, seeding=exp_seeding, scale=exp_scale, agents=exp_agents,
            concurrent=exp_concurrent, rho=exp_rho)


@hydra.main(version_base=None, config_name="rev")
def main(cfg) -> None:
    c = RevConfig(**OmegaConf.to_container(cfg, resolve=True))
    print(f"[rev] exp={c.exp} datasets={c.datasets}")
    EXPS[c.exp](c)


if __name__ == "__main__":
    if "--write-config" in sys.argv:
        os.makedirs(os.path.join(ROOT, "conf"), exist_ok=True)
        p = os.path.join(ROOT, "conf", "rev_experiments.yaml")
        with open(p, "w") as f:
            f.write(OmegaConf.to_yaml(OmegaConf.structured(RevConfig)))
        print(f"wrote {p}")
        sys.exit(0)
    sys.argv += ["hydra.run.dir=" + os.path.join(ROOT, "outputs", "hydra_logs"),
                 "hydra.output_subdir=null"]
    main()
