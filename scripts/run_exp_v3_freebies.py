#!/usr/bin/env python3
# Author: Laure Berti-Equille (laure.berti@ird.fr)
# Paper:  Berti-Equille, L. (2026). Where Is the Truth When AI Agents Disagree?
#         Dependence-Aware Belief Propagation for Unsupervised Truth Discovery.
#         Paper accepted at IEEE BigData 2026.
import os, sys, time, csv
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from bptd.data import load_td_data
from bptd.bp import run_bptd
from bptd import baselines as B
from bptd import metrics as M
from bptd.config import BPConfig

RES = os.path.join(ROOT, "outputs")
bpc = BPConfig(mode="onepass", dep_mode="cooccur", dep_threshold=0.5, alpha0=2.0, beta0=1.0)

d = load_td_data(os.path.join(ROOT, "data", "mmlu"), None)
true_acc = d.measured_agent_accuracy()
est_bptd = run_bptd(d, bpc).reliability
est_ds = B.dawid_skene(d).reliability
rho_bptd = M.spearman(est_bptd, true_acc)
rho_ds = M.spearman(est_ds, true_acc)
with open(os.path.join(RES, "c2_scatter_mmlu.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["agent", "true_acc", "est_bptd", "est_ds"])
    for i, a in enumerate(d.agents):
        w.writerow([a, round(float(true_acc[i]), 4), round(float(est_bptd[i]), 4),
                    round(float(est_ds[i]), 4)])
print(f"[C2 scatter] MMLU-160: BP-TD rho={rho_bptd:.4f}  DawidSkene rho={rho_ds:.4f}  "
      f"({d.n_agents} agents)")

rows = []
for tag, folder in [("ollama-160", "data/mmlu"),
                    ("vllm-5k", "data/mmlu_vllm_5k"),
                    ("vllm-10k", "data/mmlu_vllm_10k")]:
    dd = load_td_data(os.path.join(ROOT, folder), None)
    ta = dd.measured_agent_accuracy()
    ts = []
    rel = None
    for _ in range(5):
        t0 = time.time(); res = run_bptd(dd, bpc); ts.append(time.time() - t0); rel = res.reliability
    rho = M.spearman(rel, ta)
    nm = dd.n_questions * dd.n_agents
    rt = float(np.median(ts))
    rows.append(dict(tag=tag, n_questions=dd.n_questions, n_agents=dd.n_agents, nm=nm,
                     bptd_rho=round(float(rho), 4), runtime_s=round(rt, 4),
                     ms_per_1k_obs=round(1000 * rt / (nm / 1000), 4)))
    print(f"[scale] {tag:11s} n={dd.n_questions:5d} m={dd.n_agents} nm={nm:7d} | "
          f"BP-TD rho={rho:.4f} | runtime={rt*1000:.1f}ms")
with open(os.path.join(RES, "scale_reliability_runtime.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

r5, r10 = rows[1], rows[2]
print(f"[runtime] 5k->10k: nm x{r10['nm']/r5['nm']:.2f}, runtime x{r10['runtime_s']/max(r5['runtime_s'],1e-9):.2f} "
      f"(=linear if ~equal)")
print("wrote outputs/c2_scatter_mmlu.csv + outputs/scale_reliability_runtime.csv")
