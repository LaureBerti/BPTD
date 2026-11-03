# Author: Laure Berti-Equille (laure.berti@ird.fr)
# Paper:  Berti-Equille, L. (2026). Where Is the Truth When AI Agents Disagree?
#         Dependence-Aware Belief Propagation for Unsupervised Truth Discovery.
#         Paper accepted at IEEE BigData 2026.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .data import TDData, signed_agreement_matrix


@dataclass
class BaselineResult:
    pred: np.ndarray
    reliability: Optional[np.ndarray] = None
    oracle: bool = False


def _answered(data: TDData) -> np.ndarray:
    return data.answers >= 0


def _vote_counts(data: TDData, q: int, weights: Optional[np.ndarray] = None) -> np.ndarray:
    d = data.domain_size(q)
    counts = np.zeros(d)
    col = data.answers[:, q]
    for a in range(data.n_agents):
        v = col[a]
        if v < 0:
            continue
        counts[v] += 1.0 if weights is None else weights[a]
    return counts


def majority_vote(data: TDData) -> BaselineResult:
    n = data.n_questions
    pred = np.zeros(n, dtype=np.int64)
    for q in range(n):
        pred[q] = int(np.argmax(_vote_counts(data, q)))
    rel = _agreement_with_pred(data, pred)
    return BaselineResult(pred=pred, reliability=rel)


def _agreement_with_pred(data: TDData, pred: np.ndarray) -> np.ndarray:
    m = data.n_agents
    rel = np.full(m, np.nan)
    ans = data.answers
    for a in range(m):
        mask = ans[a] >= 0
        if mask.sum() == 0:
            continue
        rel[a] = float((ans[a][mask] == pred[mask]).mean())
    return rel


def best_single_agent(data: TDData) -> BaselineResult:
    acc = data.measured_agent_accuracy()
    best = int(np.nanargmax(acc))
    mv = majority_vote(data).pred
    pred = data.answers[best].copy()
    pred[pred < 0] = mv[pred < 0]
    rel = np.zeros(data.n_agents)
    rel[best] = 1.0
    return BaselineResult(pred=pred, reliability=acc, oracle=True)


def weighted_mv(data: TDData, n_iter: int = 1) -> BaselineResult:
    pred = majority_vote(data).pred
    weights = np.ones(data.n_agents)
    for _ in range(n_iter):
        weights = np.nan_to_num(_agreement_with_pred(data, pred), nan=0.0)
        pred = np.array(
            [int(np.argmax(_vote_counts(data, q, weights))) for q in range(data.n_questions)],
            dtype=np.int64,
        )
    return BaselineResult(pred=pred, reliability=weights)


def dawid_skene(data: TDData, n_iter: int = 50, tol: float = 1e-4) -> BaselineResult:
    m, n = data.n_agents, data.n_questions
    ans, answered = data.answers, _answered(data)
    dom = [data.domain_size(q) for q in range(n)]
    beliefs = [_vote_counts(data, q) for q in range(n)]
    beliefs = [b / b.sum() if b.sum() > 0 else np.ones(dom[q]) / dom[q]
               for q, b in enumerate(beliefs)]
    rel = np.full(m, 0.7)
    for _ in range(n_iter):
        sc = np.zeros(m); tot = np.zeros(m)
        for q in range(n):
            for a in np.where(answered[:, q])[0]:
                sc[a] += beliefs[q][ans[a, q]]; tot[a] += 1.0
        new_rel = np.where(tot > 0, sc / np.maximum(tot, 1), 0.7)
        new_rel = np.clip(new_rel, 1e-3, 1 - 1e-3)
        delta = 0.0
        for q in range(n):
            d = dom[q]
            logb = np.zeros(d)
            for a in np.where(answered[:, q])[0]:
                y = ans[a, q]
                c = np.full(d, np.log((1 - new_rel[a]) / max(d - 1, 1)))
                c[y] = np.log(new_rel[a])
                logb += c
            logb -= logb.max()
            b = np.exp(logb); b /= b.sum()
            delta = max(delta, float(np.abs(b - beliefs[q]).max()))
            beliefs[q] = b
        rel = new_rel
        if delta < tol:
            break
    pred = np.array([int(np.argmax(beliefs[q])) for q in range(n)], dtype=np.int64)
    return BaselineResult(pred=pred, reliability=rel)


def truthfinder(data: TDData, n_iter: int = 20, gamma: float = 0.3) -> BaselineResult:
    m, n = data.n_agents, data.n_questions
    ans, answered = data.answers, _answered(data)
    trust = np.full(m, 0.7)
    pred = majority_vote(data).pred
    for _ in range(n_iter):
        tau = -np.log(np.clip(1 - trust, 1e-6, 1.0))
        for q in range(n):
            d = data.domain_size(q)
            sigma = np.zeros(d)
            for a in np.where(answered[:, q])[0]:
                sigma[ans[a, q]] += tau[a]
            conf = 1.0 / (1.0 + np.exp(-gamma * sigma))
            pred[q] = int(np.argmax(conf))
        trust = np.nan_to_num(_agreement_with_pred(data, pred), nan=0.5)
        trust = np.clip(trust, 1e-3, 1 - 1e-3)
    return BaselineResult(pred=pred, reliability=trust)


def investment(data: TDData, n_iter: int = 20, g_exp: float = 1.2) -> BaselineResult:
    m, n = data.n_agents, data.n_questions
    ans, answered = data.answers, _answered(data)
    n_claims_of = answered.sum(axis=1).astype(float)
    n_claims_of[n_claims_of == 0] = 1.0
    trust = np.full(m, 1.0)
    pred = majority_vote(data).pred
    for _ in range(n_iter):
        claim_val: Dict[tuple, float] = {}
        for q in range(n):
            for a in np.where(answered[:, q])[0]:
                key = (q, int(ans[a, q]))
                claim_val[key] = claim_val.get(key, 0.0) + trust[a] / n_claims_of[a]
        for q in range(n):
            d = data.domain_size(q)
            score = np.array([claim_val.get((q, v), 0.0) ** g_exp for v in range(d)])
            pred[q] = int(np.argmax(score))
        new_trust = np.zeros(m)
        for q in range(n):
            for a in np.where(answered[:, q])[0]:
                key = (q, int(ans[a, q]))
                total = claim_val.get(key, 1e-9)
                grown = total ** g_exp
                new_trust[a] += grown * (trust[a] / n_claims_of[a]) / max(total, 1e-9)
        trust = new_trust / max(new_trust.max(), 1e-9)
    return BaselineResult(pred=pred, reliability=trust)


def sums(data: TDData, n_iter: int = 20) -> BaselineResult:
    m, n = data.n_agents, data.n_questions
    ans, answered = data.answers, _answered(data)
    trust = np.full(m, 1.0)
    pred = np.zeros(n, dtype=np.int64)
    for _ in range(n_iter):
        belief: Dict[tuple, float] = {}
        for q in range(n):
            for a in np.where(answered[:, q])[0]:
                key = (q, int(ans[a, q]))
                belief[key] = belief.get(key, 0.0) + trust[a]
        if belief:
            bmax = max(belief.values()) or 1.0
            for k in belief:
                belief[k] /= bmax
        new_trust = np.zeros(m)
        for q in range(n):
            for a in np.where(answered[:, q])[0]:
                new_trust[a] += belief.get((q, int(ans[a, q])), 0.0)
        trust = new_trust / max(new_trust.max(), 1e-9)
        for q in range(n):
            d = data.domain_size(q)
            score = np.array([belief.get((q, v), 0.0) for v in range(d)])
            pred[q] = int(np.argmax(score))
    return BaselineResult(pred=pred, reliability=trust)


def crh(data: TDData, n_iter: int = 20) -> BaselineResult:
    m, n = data.n_agents, data.n_questions
    ans, answered = data.answers, _answered(data)
    pred = majority_vote(data).pred
    weights = np.ones(m)
    for _ in range(n_iter):
        err = np.zeros(m)
        for a in range(m):
            mask = answered[a]
            err[a] = float((ans[a][mask] != pred[mask]).sum())
        total_err = err.sum() + 1e-9
        weights = -np.log(np.clip(err / total_err, 1e-9, 1.0))
        weights = np.clip(weights, 1e-6, None)
        pred = np.array(
            [int(np.argmax(_vote_counts(data, q, weights))) for q in range(n)],
            dtype=np.int64,
        )
    rel = weights / max(weights.max(), 1e-9)
    return BaselineResult(pred=pred, reliability=rel)


def catd(data: TDData, n_iter: int = 20, alpha: float = 0.05) -> BaselineResult:
    from scipy.stats import chi2

    m, n = data.n_agents, data.n_questions
    ans, answered = data.answers, _answered(data)
    n_ans = answered.sum(axis=1).astype(float)
    pred = majority_vote(data).pred
    weights = np.ones(m)
    for _ in range(n_iter):
        err = np.zeros(m)
        for a in range(m):
            mask = answered[a]
            err[a] = float((ans[a][mask] != pred[mask]).sum())
        chi_q = np.array([chi2.ppf(1 - alpha, max(int(k), 1)) for k in n_ans])
        weights = chi_q / np.clip(err, 1e-6, None)
        pred = np.array(
            [int(np.argmax(_vote_counts(data, q, weights))) for q in range(n)],
            dtype=np.int64,
        )
    rel = weights / max(weights.max(), 1e-9)
    return BaselineResult(pred=pred, reliability=rel)


def ltm(data: TDData, n_iter: int = 40, tol: float = 1e-4) -> BaselineResult:
    m, n = data.n_agents, data.n_questions
    ans, answered = data.answers, _answered(data)
    dom = [data.domain_size(q) for q in range(n)]
    beliefs = [_vote_counts(data, q) for q in range(n)]
    beliefs = [b / b.sum() if b.sum() > 0 else np.ones(dom[q]) / dom[q]
               for q, b in enumerate(beliefs)]
    phi1 = np.full(m, 0.7)
    phi0 = np.full(m, 0.3 / 3)
    for _ in range(n_iter):
        s1 = np.zeros(m); n1 = np.zeros(m)
        s0 = np.zeros(m); n0 = np.zeros(m)
        for q in range(n):
            d = dom[q]; bq = beliefs[q]
            for a in np.where(answered[:, q])[0]:
                y = ans[a, q]
                s1[a] += bq[y]; n1[a] += 1.0
                s0[a] += (1 - bq[y]); n0[a] += (d - 1)
        phi1 = np.clip(np.where(n1 > 0, s1 / np.maximum(n1, 1), 0.7), 1e-3, 1 - 1e-3)
        phi0 = np.clip(np.where(n0 > 0, s0 / np.maximum(n0, 1), 0.1), 1e-4, 0.9)
        delta = 0.0
        for q in range(n):
            d = dom[q]
            logb = np.zeros(d)
            for a in np.where(answered[:, q])[0]:
                y = ans[a, q]
                c = np.full(d, np.log(phi0[a]))
                c[y] = np.log(phi1[a])
                logb += c
            logb -= logb.max()
            b = np.exp(logb); b /= b.sum()
            delta = max(delta, float(np.abs(b - beliefs[q]).max()))
            beliefs[q] = b
        if delta < tol:
            break
    pred = np.array([int(np.argmax(beliefs[q])) for q in range(n)], dtype=np.int64)
    return BaselineResult(pred=pred, reliability=phi1)


def sstf_iv(data: TDData, n_iter: int = 50, init_cred: float = 0.90) -> BaselineResult:
    m, n = data.n_agents, data.n_questions
    ans, answered = data.answers, _answered(data)
    cred = np.full(m, init_cred)
    pred = majority_vote(data).pred
    for _ in range(n_iter):
        prev = pred.copy()
        pred = np.array(
            [int(np.argmax(_vote_counts(data, q, cred))) for q in range(n)],
            dtype=np.int64,
        )
        for s in range(m):
            mask = answered[s]
            cred[s] = float((ans[s][mask] == pred[mask]).mean()) if mask.sum() else 0.0
        if np.array_equal(pred, prev):
            break
    return BaselineResult(pred=pred, reliability=cred)


def sstf_vi(data: TDData, n_iter: int = 100, alpha0: float = 9.0,
            beta0: float = 1.0) -> BaselineResult:
    m, n = data.n_agents, data.n_questions
    ans, answered = data.answers, _answered(data)
    dom = [data.domain_size(q) for q in range(n)]
    alpha = np.full(m, alpha0)
    beta = np.full(m, beta0)
    beliefs = []
    for q in range(n):
        c = _vote_counts(data, q)
        b = np.full(dom[q], 1.0 / dom[q])
        if c.sum() > 0:
            b[int(np.argmax(c))] += 0.02
        beliefs.append(b / b.sum())
    for _ in range(n_iter):
        ec = np.clip(alpha / (alpha + beta), 1e-9, 1 - 1e-9)
        log_ec, log_1mec = np.log(ec), np.log(1 - ec)
        delta = 0.0
        for q in range(n):
            d = dom[q]
            logb = np.zeros(d)
            log_norm = np.log(max(d - 1, 1))
            for s in np.where(answered[:, q])[0]:
                y = ans[s, q]
                contrib = np.full(d, log_1mec[s] - log_norm)
                contrib[y] = log_ec[s]
                logb += contrib
            logb -= logb.max()
            b = np.exp(logb); b /= b.sum()
            delta = max(delta, float(np.abs(b - beliefs[q]).max()))
            beliefs[q] = b
        alpha = np.ones(m); beta = np.ones(m)
        for q in range(n):
            bq = beliefs[q]
            for s in np.where(answered[:, q])[0]:
                p_correct = bq[ans[s, q]]
                alpha[s] += p_correct
                beta[s] += (1 - p_correct)
        if delta < 1e-4:
            break
    pred = np.array([int(np.argmax(beliefs[q])) for q in range(n)], dtype=np.int64)
    return BaselineResult(pred=pred, reliability=alpha / (alpha + beta))


def cluster_vote(data: TDData, threshold: float = 0.5) -> BaselineResult:
    m, n = data.n_agents, data.n_questions
    ans = data.answers

    W = signed_agreement_matrix(data)
    adj = W >= threshold
    np.fill_diagonal(adj, False)
    comp = np.full(m, -1, dtype=np.int64)
    cid = 0
    for a in range(m):
        if comp[a] != -1:
            continue
        stack = [a]
        comp[a] = cid
        while stack:
            u = stack.pop()
            for v in np.where(adj[u])[0]:
                if comp[v] == -1:
                    comp[v] = cid
                    stack.append(v)
        cid += 1
    n_clusters = cid

    pred = np.zeros(n, dtype=np.int64)
    for q in range(n):
        d = data.domain_size(q)
        col = ans[:, q]
        group_votes = []
        for g in range(n_clusters):
            members = np.where((comp == g) & (col >= 0))[0]
            if len(members) == 0:
                continue
            counts = np.zeros(d)
            for a in members:
                counts[col[a]] += 1.0
            group_votes.append(int(np.argmax(counts)))
        if not group_votes:
            pred[q] = 0
            continue
        final = np.zeros(d)
        for v in group_votes:
            final[v] += 1.0
        pred[q] = int(np.argmax(final))

    rel = _agreement_with_pred(data, pred)
    return BaselineResult(pred=pred, reliability=rel)


def accu_vote(data: TDData, n_iter: int = 20, init_acc: float = 0.8) -> BaselineResult:
    m, n = data.n_agents, data.n_questions
    ans, answered = data.answers, _answered(data)
    A = np.full(m, init_acc)
    pred = majority_vote(data).pred
    for _ in range(n_iter):
        A = np.clip(A, 1e-3, 1 - 1e-3)
        for q in range(n):
            d = data.domain_size(q)
            w_lo = np.log(A * (d - 1) / (1 - A))
            score = np.zeros(d)
            for a in np.where(answered[:, q])[0]:
                score[ans[a, q]] += w_lo[a]
            pred[q] = int(np.argmax(score))
        A = np.nan_to_num(_agreement_with_pred(data, pred), nan=init_acc)
    return BaselineResult(pred=pred, reliability=A)


def depen_accu(data: TDData, n_iter: int = 20, init_acc: float = 0.8,
               alpha: float = 0.5) -> BaselineResult:
    m, n = data.n_agents, data.n_questions
    ans, answered = data.answers, _answered(data)
    A = np.full(m, init_acc)
    pred = majority_vote(data).pred
    indep = np.ones(m)
    for _ in range(n_iter):
        A = np.clip(A, 1e-3, 1 - 1e-3)
        dep = np.zeros((m, m))
        for a in range(m):
            aa = ans[a]
            for b in range(a + 1, m):
                bb = ans[b]
                both = (aa >= 0) & (bb >= 0)
                co = int(both.sum())
                if co == 0:
                    continue
                shared_err = np.sum((aa[both] == bb[both]) & (aa[both] != pred[both]))
                dij = shared_err / co
                dep[a, b] = dep[b, a] = dij
        indep = np.clip(np.prod(1.0 - alpha * dep, axis=1), 1e-3, 1.0)
        w_base = np.log(A * 2.0 / (1 - A))
        for q in range(n):
            d = data.domain_size(q)
            lo = np.log(A * (d - 1) / (1 - A)) * indep
            score = np.zeros(d)
            for a in np.where(answered[:, q])[0]:
                score[ans[a, q]] += lo[a]
            pred[q] = int(np.argmax(score))
        A = np.nan_to_num(_agreement_with_pred(data, pred), nan=init_acc)
    return BaselineResult(pred=pred, reliability=A * indep)


def pochampally_corr(data: TDData, n_iter: int = 20) -> BaselineResult:
    m, n = data.n_agents, data.n_questions
    ans, answered = data.answers, _answered(data)
    corr = np.clip(signed_agreement_matrix(data), 0.0, 1.0)
    p = np.full(m, 0.8)
    pred = majority_vote(data).pred
    for _ in range(n_iter):
        p = np.clip(np.nan_to_num(_agreement_with_pred(data, pred), nan=0.5), 1e-3, 1 - 1e-3)
        for q in range(n):
            d = data.domain_size(q)
            score = np.full(d, -np.inf)
            for v in range(d):
                S = [a for a in np.where(answered[:, q])[0] if ans[a, q] == v]
                if not S:
                    continue
                k = len(S)
                if k > 1:
                    pc = [corr[a, b] for i, a in enumerate(S) for b in S[i + 1:]]
                    avg_corr = float(np.mean(pc)) if pc else 0.0
                else:
                    avg_corr = 0.0
                eff = k / (1.0 + avg_corr * (k - 1))
                mean_lo = float(np.mean([np.log(p[a] * (d - 1) / (1 - p[a])) for a in S]))
                score[v] = mean_lo * eff
            pred[q] = int(np.argmax(score))
    return BaselineResult(pred=pred, reliability=p)


def ising_meanfield(data: TDData, n_iter: int = 40, beta: float = 1.0,
                    mf_iter: int = 15) -> BaselineResult:
    m, n = data.n_agents, data.n_questions
    J = signed_agreement_matrix(data)
    deg = np.maximum(np.abs(J).sum(axis=1), 1.0)
    pred = majority_vote(data).pred
    mag = np.zeros(m)
    w = np.full(m, 0.5)
    for _ in range(n_iter):
        h = 2.0 * np.nan_to_num(_agreement_with_pred(data, pred), nan=0.5) - 1.0
        for _mf in range(mf_iter):
            mag = np.tanh(beta * (J @ mag) / deg + h)
        w = np.clip((mag + 1.0) / 2.0, 1e-3, 1.0)
        pred = np.array(
            [int(np.argmax(_vote_counts(data, q, w))) for q in range(n)],
            dtype=np.int64,
        )
    return BaselineResult(pred=pred, reliability=w)


BASELINES: Dict[str, Callable[[TDData], BaselineResult]] = {
    "MajorityVote": majority_vote,
    "BestSingleAgent": best_single_agent,
    "WeightedMV": weighted_mv,
    "DawidSkene": dawid_skene,
    "TruthFinder": truthfinder,
    "Investment": investment,
    "Sums": sums,
    "CRH": crh,
    "CATD": catd,
    "LTM": ltm,
    "SSTF-IV": sstf_iv,
    "SSTF-VI": sstf_vi,
    "ClusterVote": cluster_vote,
    "ACCU": accu_vote,
    "DEPEN": depen_accu,
    "PochampallyCorr": pochampally_corr,
    "IsingMF": ising_meanfield,
}
