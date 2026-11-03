# Author: Laure Berti-Equille (laure.berti@ird.fr)
# Paper:  Berti-Equille, L. (2026). Where Is the Truth When AI Agents Disagree?
#         Dependence-Aware Belief Propagation for Unsupervised Truth Discovery.
#         Paper accepted at IEEE BigData 2026.
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
LF = ["WVote-family", "BP-TD-family", "BP-TD-cooccur@0.5", "WVote-cooccur@0.5", "BP-TD-cooccur-otsu",
      "BP-TD-none", "ClusterVote@0.5"]


def auc(score, label):
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def subpools():
    d = pd.read_csv(os.path.join(RES, "E3_subpools.csv"))
    rows = []
    for (ds, kind), g in d.groupby(["dataset", "kind"]):
        for strat, h in [("all", g), ("largest_is_strongest", g[g.largest_is_strongest]),
                         ("other", g[~g.largest_is_strongest])]:
            if len(h) == 0:
                continue
            for m in LF:
                gain = h[m] - h["MajorityVote"]
                rows.append(dict(dataset=ds, kind=kind, stratum=strat, method=m, n_pools=len(h),
                                 median_gain=round(gain.median(), 4), q25=round(gain.quantile(.25), 4),
                                 q75=round(gain.quantile(.75), 4), frac_pos=round((gain > 0).mean(), 3),
                                 frac_neg=round((gain < 0).mean(), 3)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "summary_E3_subpools.csv"), index=False)
    print(out[out.method.isin(["BP-TD-family", "BP-TD-cooccur@0.5", "BP-TD-cooccur-otsu"])].to_string(index=False))
    rows = []
    for ds, g in list(d.groupby("dataset")) + [("pooled", d)]:
        gain = (g["BP-TD-family"] - g["MajorityVote"]).to_numpy()
        lab = gain > 0
        for stat in ("S_bloc_vs_rest", "flip_rate"):
            x = g[stat].to_numpy()
            rho = spearmanr(x, gain).correlation if np.std(x) > 0 and np.std(gain) > 0 else float("nan")
            rows.append(dict(dataset=ds, statistic=stat, n_pools=len(g), spearman_with_gain=round(rho, 3),
                             auc_low_predicts_gain=round(auc(-x, lab), 3) if stat == "S_bloc_vs_rest" else "",
                             auc_high_predicts_gain=round(auc(x, lab), 3) if stat == "flip_rate" else "",
                             frac_gain_pos=round(lab.mean(), 3)))
    out7 = pd.DataFrame(rows)
    out7.to_csv(os.path.join(RES, "summary_E7_diagnostic.csv"), index=False)
    print(out7.to_string(index=False))


def budget():
    d = pd.read_csv(os.path.join(RES, "E6_budget.csv"))
    cols = ["MajorityVote", "BP-TD-family", "BP-TD-cooccur(k-tuned)", "BestAgent(k)"]
    g = d.groupby(["dataset", "k"])[cols].agg(["mean", "std"]).round(4)
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    g = g.reset_index()
    g.to_csv(os.path.join(RES, "summary_E6_budget.csv"), index=False)
    print(g[["dataset", "k"] + [c for c in g.columns if c.endswith("_mean")]].to_string(index=False))
    for ds, h in g.groupby("dataset"):
        best_unsup = h[["BP-TD-family_mean", "BP-TD-cooccur(k-tuned)_mean"]].max(axis=1)
        cross = h[h["BestAgent(k)_mean"] > best_unsup]
        print(f"  {ds}: smallest k where BestAgent(k) > best BP-TD = {cross.k.min() if len(cross) else 'none'}")


def concurrent():
    fs = [f for f in os.listdir(RES) if f.startswith("E11_concurrent_")]
    if not fs:
        return
    d = pd.concat([pd.read_csv(os.path.join(RES, f)) for f in sorted(fs)])
    print(d.to_string(index=False))


if __name__ == "__main__":
    for f in (subpools, budget, concurrent):
        try:
            f()
        except FileNotFoundError as e:
            print(f"[skip] {e}")
