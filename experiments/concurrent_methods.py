# Author: Laure Berti-Equille (laure.berti@ird.fr)
# Paper:  Berti-Equille, L. (2026). Where Is the Truth When AI Agents Disagree?
#         Dependence-Aware Belief Propagation for Unsupervised Truth Discovery.
#         Paper accepted at IEEE BigData 2026.
from __future__ import annotations

import os
import sys
import types

import numpy as np

CARE_FAILURES: list = []
CARE_DIR = os.path.join(os.path.dirname(__file__), "third_party", "CARE", "src")


def _import_care():
    if "snorkel.labeling.model" not in sys.modules:
        for name in ("snorkel", "snorkel.labeling", "snorkel.labeling.model"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["snorkel.labeling.model"].LabelModel = None
    if CARE_DIR not in sys.path:
        sys.path.insert(0, CARE_DIR)
    import pgm_tools
    return pgm_tools


def _indicator_frame(d):
    import pandas as pd
    act = np.where((d.answers >= 0).any(axis=1))[0]
    rows, owner = [], []
    for q in range(d.n_questions):
        k = d.domain_size(q)
        for v in range(k):
            rows.append([(1.0 if d.answers[a, q] == v else 0.0) for a in act])
            owner.append((q, v))
    return pd.DataFrame(rows, columns=[d.agents[a] for a in act]), owner


def care_svd(d, gamma: float = 1.0, ridge: float = 0.0) -> np.ndarray:
    pgm = _import_care()
    df, owner = _indicator_frame(d)
    df = df.loc[:, df.std() > 0]
    corr = df.corr().replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(corr, 1.0)
    corr = corr + ridge * np.eye(corr.shape[0])
    scores = None
    for solver in (pgm.cp.SCS, pgm.cp.CLARABEL):
        try:
            scores = np.asarray(pgm.caresl_aggregate(df, gamma=float(gamma), verbose=False,
                                                     corr_matrix=corr, solver=solver))
            if np.all(np.isfinite(scores)):
                break
            scores = None
        except Exception:
            scores = None
    if scores is None:
        CARE_FAILURES.append((gamma, ridge))
        from bptd import baselines as B
        return B.majority_vote(d).pred
    pred = np.full(d.n_questions, -1, dtype=np.int64)
    best = np.full(d.n_questions, -np.inf)
    for s, (q, v) in zip(scores, owner):
        if s > best[q]:
            best[q], pred[q] = s, v
    return pred


def _second_order(d):
    m = d.n_agents
    K = max(d.domain_size(q) for q in range(d.n_questions))
    cnt = np.zeros((m, m, K, K))
    for q in range(d.n_questions):
        a = d.answers[:, q]
        idx = np.where(a >= 0)[0]
        for i in idx:
            cnt[i, idx, a[i], a[idx]] += 1.0
    denom = cnt.sum(axis=2, keepdims=True)
    return np.divide(cnt, denom, out=np.zeros_like(cnt), where=denom > 0), K


def isp(d) -> np.ndarray:
    P, _ = _second_order(d)
    pred = np.full(d.n_questions, -1, dtype=np.int64)
    for q in range(d.n_questions):
        a = d.answers[:, q]
        idx = np.where(a >= 0)[0]
        if len(idx) == 0:
            continue
        K = d.domain_size(q)
        votes = np.bincount(a[idx], minlength=K)[:K].astype(float)
        expect = np.zeros(K)
        for i in idx:
            others = [j for j in idx if j != i]
            if not others:
                continue
            for s in range(K):
                tot = 0.0
                for j in others:
                    alt = [x for x in range(K) if x != a[j]]
                    tot += np.mean([P[i, j, s, x] for x in alt]) if alt else 0.0
                expect[s] += tot / len(others)
        pred[q] = int(np.argmax(votes - expect))
    return pred


def ow_i(d, eps: float = 1e-3) -> np.ndarray:
    pseudo = isp(d)
    pred = np.full(d.n_questions, -1, dtype=np.int64)
    m = d.n_agents
    x = np.full(m, np.nan)
    for a in range(m):
        ok = d.answers[a] >= 0
        if ok.any():
            x[a] = float((d.answers[a, ok] == pseudo[ok]).mean())
    for q in range(d.n_questions):
        K = d.domain_size(q)
        score = np.zeros(K)
        for a in np.where(d.answers[:, q] >= 0)[0]:
            xa = np.clip(x[a], eps, 1 - eps)
            score[d.answers[a, q]] += np.log((K - 1) * xa / (1 - xa))
        if np.any(d.answers[:, q] >= 0):
            pred[q] = int(np.argmax(score))
    return pred
