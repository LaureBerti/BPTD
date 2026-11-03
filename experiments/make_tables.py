# Author: Laure Berti-Equille (laure.berti@ird.fr)
# Paper:  Berti-Equille, L. (2026). Where Is the Truth When AI Agents Disagree?
#         Dependence-Aware Belief Propagation for Unsupervised Truth Discovery.
#         Paper accepted at IEEE BigData 2026.
import os

import pandas as pd

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
DS = ["mmlu", "sciq", "truthfulqa", "arc"]

ROWS = [
    ("Majority vote (MV)", "E1", "MajorityVote", False),
    ("BP-TD cooccur, $\\theta$ val.-tuned", "E1", "BP-TD-cooccur", True),
    ("\\quad one-pass = weighted vote", "E1", "WVote-cooccur", False),
    ("BP-TD cooccur, $\\theta{=}0.5$ (no labels)", "E1", "BP-TD-cooccur@0.5", False),
    ("BP-TD cooccur, Otsu $\\theta$ (no labels)", "E1", "BP-TD-cooccur-otsu", False),
    ("BP-TD family (no labels)", "E1", "BP-TD-family", True),
    ("\\quad one-pass = weighted vote", "E1", "WVote-family", False),
    ("BP-TD none ($w_a{\\equiv}1$)", "E1", "BP-TD-none", True),
    ("ClusterVote, $\\theta$ val.-tuned", "E1", "ClusterVote", False),
    ("ClusterVote, $\\theta{=}0.5$", "E1", "ClusterVote@0.5", False),
    ("PochampallyCorr", "E1", "PochampallyCorr", False),
    ("ACCU", "E1", "ACCU", False),
    ("DEPEN", "E1", "DEPEN", False),
    ("IsingMF (surrogate)", "E1", "IsingMF", False),
    ("CARE-SVD (ridge, $\\gamma$ val.-tuned)", "E11", "CARE-SVD(ridge,γ-val)", False),
    ("ISP", "E11", "ISP", False),
    ("OW-I", "E11", "OW-I", False),
    ("Dawid--Skene ($=$ LTM here)", "E1", "DawidSkene", False),
    ("CRH", "E1", "CRH", False),
    ("CATD", "E1", "CATD", False),
    ("TruthFinder", "E1", "TruthFinder", False),
    ("Investment", "E1", "Investment", False),
    ("Sums", "E1", "Sums", False),
    ("SSTF-IV", "E1", "SSTF-IV", False),
    ("SSTF-VI", "E1", "SSTF-VI", False),
    ("WeightedMV", "E1", "WeightedMV", False),
    ("Best agent, val.-selected (uses labels)", "E1", "BestAgent-valsel", False),
]


def load():
    t = {}
    for ds in DS:
        t[("E1", ds)] = pd.read_csv(os.path.join(RES, f"E1_main_{ds}.csv")).set_index("method")
        t[("E11", ds)] = pd.read_csv(os.path.join(RES, f"E11_concurrent_{ds}.csv")).set_index("method")
    return t


def cell(df, m, show_T):
    r = df.loc[m]
    acc, d = float(r.mean_acc), float(r.diff_vs_mv)
    p = r.get("holm_p_vs_mv")
    mark = ""
    if m != "MajorityVote" and p == p and float(p) < 0.05:
        mark = "$^{+}$" if d > 0 else "$^{-}$"
    s = f"{acc:.3f}{mark}"
    if show_T and "mean_n_iter" in r and r.mean_n_iter == r.mean_n_iter and str(r.mean_n_iter) != "":
        s += f" {{\\scriptsize[{float(r.mean_n_iter):.0f}]}}"
    return s


def main_table():
    t = load()
    lines = []
    for lab, src, m, showT in ROWS:
        cells = [cell(t[(src, ds)], m, showT) for ds in DS]
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
        if m in ("MajorityVote", "BP-TD-none", "IsingMF", "OW-I", "WeightedMV"):
            lines.append("\\midrule")
    open(os.path.join(RES, "tab_main.tex"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def scale_table():
    rows = [("Majority vote", "MajorityVote"), ("BP-TD cooccur, $\\theta$ val.-tuned", "BP-TD-cooccur"),
            ("\\quad one-pass = weighted vote", "WVote-cooccur"), ("BP-TD cooccur, $\\theta{=}0.5$", "BP-TD-cooccur@0.5"),
            ("BP-TD cooccur, Otsu $\\theta$", "BP-TD-cooccur-otsu"), ("BP-TD family", "BP-TD-family"),
            ("BP-TD none", "BP-TD-none"), ("ClusterVote, $\\theta$ val.-tuned", "ClusterVote"),
            ("Best agent, val.-selected", "BestAgent-valsel")]
    t = {k: pd.read_csv(os.path.join(RES, f"E1_main_mmlu_vllm_{k}.csv")).set_index("method") for k in ("5k", "10k")}
    lines = []
    for lab, m in rows:
        cells = []
        for k in ("5k", "10k"):
            r = t[k].loc[m]
            p = r.holm_p_vs_mv
            mark = "$^{+}$" if (m != "MajorityVote" and p == p and p < 0.05 and r.diff_vs_mv > 0) else \
                   "$^{-}$" if (m != "MajorityVote" and p == p and p < 0.05) else ""
            d = "" if m == "MajorityVote" else f" ({r.diff_vs_mv:+.3f})"
            cells.append(f"{r.mean_acc:.3f}{mark}{d}")
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    open(os.path.join(RES, "tab_scale.tex"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main_table()
    scale_table()
