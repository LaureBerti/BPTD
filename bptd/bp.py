# Author: Laure Berti-Equille (laure.berti@ird.fr)
# Paper:  Berti-Equille, L. (2026). Where Is the Truth When AI Agents Disagree?
#         Dependence-Aware Belief Propagation for Unsupervised Truth Discovery.
#         Paper accepted at IEEE BigData 2026.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.special import digamma

from .config import BPConfig
from .data import TDData, signed_agreement_matrix


@dataclass
class BPTDResult:
    pred: np.ndarray
    confidence: np.ndarray
    reliability: np.ndarray
    beliefs: List[np.ndarray]
    n_iter: int
    residuals: List[float]
    converged: bool
    runtime_s: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)


def _expected_logs(alpha: np.ndarray, beta: np.ndarray):
    psi_sum = digamma(alpha + beta)
    e_log_r = digamma(alpha) - psi_sum
    e_log_1mr = digamma(beta) - psi_sum
    return e_log_r, e_log_1mr


def _cooccur_weights(W: np.ndarray, threshold: float) -> np.ndarray:
    cluster_size = 1.0 + (W >= threshold).sum(axis=1)
    return 1.0 / cluster_size


def _family_weights(data: TDData) -> np.ndarray:
    base = getattr(data, "base_model", None)
    if not base:
        return np.ones(data.n_agents)
    from collections import Counter
    sizes = Counter(base)
    return np.array([1.0 / sizes[b] for b in base])


def _shared_error_weights(data: TDData, beliefs, threshold: float) -> np.ndarray:
    m, n = data.n_agents, data.n_questions
    ans = data.answers
    answered = ans >= 0
    shared_wrong = np.zeros((m, m))
    co = np.zeros((m, m))
    for q in range(n):
        voters = np.where(answered[:, q])[0]
        if len(voters) < 2:
            continue
        co[np.ix_(voters, voters)] += 1.0
        bq = beliefs[q]
        vals = ans[voters, q]
        for v in np.unique(vals):
            grp = voters[vals == v]
            if len(grp) < 2:
                continue
            shared_wrong[np.ix_(grp, grp)] += (1.0 - bq[v])
    np.fill_diagonal(co, 0.0)
    np.fill_diagonal(shared_wrong, 0.0)
    D = shared_wrong / np.maximum(co, 1.0)
    cluster_size = 1.0 + (D >= threshold).sum(axis=1)
    return 1.0 / cluster_size


def _signed_smooth(rel: np.ndarray, W: np.ndarray, lam: float, signed: bool) -> np.ndarray:
    A = W if signed else np.abs(W)
    deg = np.abs(A).sum(axis=1)
    deg[deg == 0] = 1.0
    centered = rel - 0.5
    neighbor = (A @ centered) / deg
    smoothed = 0.5 + (1.0 - lam) * centered + lam * neighbor
    return np.clip(smoothed, 1e-3, 1.0 - 1e-3)


def run_bptd(
    data: TDData,
    cfg: Optional[BPConfig] = None,
    seed_idx: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> BPTDResult:
    import time

    cfg = cfg or BPConfig()
    t0 = time.time()
    m, n = data.n_agents, data.n_questions
    ans = data.answers

    dom_sizes = np.array([data.domain_size(q) for q in range(n)], dtype=np.int64)
    answered = ans >= 0

    seed_set = set(int(i) for i in seed_idx) if seed_idx is not None else set()

    beliefs = [np.full(dom_sizes[q], 1.0 / dom_sizes[q]) for q in range(n)]
    for q in seed_set:
        g = int(data.gold[q])
        if g >= 0:
            b = np.full(dom_sizes[q], 1e-9)
            b[g] = 1.0
            beliefs[q] = b / b.sum()

    alpha = np.full(m, cfg.alpha0, dtype=np.float64)
    beta = np.full(m, cfg.beta0, dtype=np.float64)
    rel = alpha / (alpha + beta)

    W = signed_agreement_matrix(data)
    indep = np.ones(m)
    dyn_dep = False
    if cfg.dependence_aware and cfg.dep_mode != "none":
        if cfg.dep_mode == "cooccur":
            indep = _cooccur_weights(W, cfg.dep_threshold)
        elif cfg.dep_mode == "family":
            indep = _family_weights(data)
        elif cfg.dep_mode == "shared_error":
            dyn_dep = True
        else:
            raise ValueError(f"unknown dep_mode {cfg.dep_mode!r}")

    max_iter = 1 if cfg.mode == "onepass" else cfg.max_iter
    residuals: List[float] = []
    converged = False

    for it in range(max_iter):
        e_log_r, e_log_1mr = _expected_logs(alpha, beta)
        if dyn_dep:
            indep = _shared_error_weights(data, beliefs, cfg.dep_threshold)

        max_resid = 0.0
        for q in range(n):
            if q in seed_set:
                continue
            d = dom_sizes[q]
            logb = np.zeros(d)
            log_wrong_norm = np.log(max(d - 1, 1))
            voters = np.where(answered[:, q])[0]
            for a in voters:
                y = ans[a, q]
                contrib = np.full(d, e_log_1mr[a] - log_wrong_norm)
                contrib[y] = e_log_r[a]
                logb += indep[a] * contrib
            logb -= logb.max()
            b = np.exp(logb)
            b /= b.sum()
            if cfg.damping > 0 and it > 0:
                b = (1 - cfg.damping) * b + cfg.damping * beliefs[q]
                b /= b.sum()
            max_resid = max(max_resid, float(np.abs(b - beliefs[q]).max()))
            beliefs[q] = b

        soft_correct = np.full(m, 0.0)
        soft_total = np.full(m, 0.0)
        for q in range(n):
            voters = np.where(answered[:, q])[0]
            bq = beliefs[q]
            for a in voters:
                soft_correct[a] += bq[ans[a, q]]
                soft_total[a] += 1.0
        alpha = cfg.alpha0 + soft_correct
        beta = cfg.beta0 + (soft_total - soft_correct)
        rel = alpha / (alpha + beta)

        if cfg.sign_lambda > 0 and W is not None:
            rel = _signed_smooth(rel, W, cfg.sign_lambda, cfg.signed)
            strength = soft_total + cfg.alpha0 + cfg.beta0
            alpha = rel * strength
            beta = (1.0 - rel) * strength

        residuals.append(max_resid)
        if cfg.mode != "onepass" and max_resid < cfg.eps:
            converged = True
            break

    pred = np.array([int(np.argmax(beliefs[q])) for q in range(n)], dtype=np.int64)
    confidence = np.array([float(beliefs[q].max()) for q in range(n)])

    return BPTDResult(
        pred=pred,
        confidence=confidence,
        reliability=rel,
        beliefs=beliefs,
        n_iter=len(residuals),
        residuals=residuals,
        converged=converged,
        runtime_s=time.time() - t0,
        extra={"final_residual": residuals[-1] if residuals else float("nan")},
    )
