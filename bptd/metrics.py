# Author: Laure Berti-Equille (laure.berti@ird.fr)
# Paper:  Berti-Equille, L. (2026). Where Is the Truth When AI Agents Disagree?
#         Dependence-Aware Belief Propagation for Unsupervised Truth Discovery.
#         Paper accepted at IEEE BigData 2026.

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


def accuracy(pred: np.ndarray, gold: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    pred = np.asarray(pred); gold = np.asarray(gold)
    valid = gold >= 0
    if mask is not None:
        valid = valid & mask
    if valid.sum() == 0:
        return float("nan")
    return float((pred[valid] == gold[valid]).mean())


def macro_f1(pred: np.ndarray, gold: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    pred = np.asarray(pred); gold = np.asarray(gold)
    valid = gold >= 0
    if mask is not None:
        valid = valid & mask
    if valid.sum() == 0:
        return float("nan")
    p, g = pred[valid], gold[valid]
    labels = np.unique(g)
    f1s = []
    for c in labels:
        tp = float(((p == c) & (g == c)).sum())
        fp = float(((p == c) & (g != c)).sum())
        fn = float(((p != c) & (g == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


def _valid_pairs(est: np.ndarray, true: np.ndarray):
    est = np.asarray(est, dtype=float); true = np.asarray(true, dtype=float)
    m = np.isfinite(est) & np.isfinite(true)
    return est[m], true[m]


def spearman(est: np.ndarray, true: np.ndarray) -> float:
    from scipy.stats import spearmanr

    e, t = _valid_pairs(est, true)
    if len(e) < 3 or np.std(e) == 0 or np.std(t) == 0:
        return float("nan")
    return float(spearmanr(e, t).correlation)


def kendall(est: np.ndarray, true: np.ndarray) -> float:
    from scipy.stats import kendalltau

    e, t = _valid_pairs(est, true)
    if len(e) < 3:
        return float("nan")
    return float(kendalltau(e, t).correlation)


def ndcg_at_k(est: np.ndarray, true: np.ndarray, k: Optional[int] = None) -> float:
    e, t = _valid_pairs(est, true)
    if len(e) == 0:
        return float("nan")
    k = k or len(e)
    order = np.argsort(-e)[:k]
    gains = t[order]
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float((gains * discounts).sum())
    ideal = np.sort(t)[::-1][:k]
    idcg = float((ideal * discounts).sum())
    return dcg / idcg if idcg > 0 else float("nan")


def ece(confidence: np.ndarray, pred: np.ndarray, gold: np.ndarray, n_bins: int = 10) -> float:
    conf = np.asarray(confidence); pred = np.asarray(pred); gold = np.asarray(gold)
    valid = gold >= 0
    conf, correct = conf[valid], (pred[valid] == gold[valid]).astype(float)
    if len(conf) == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        e += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


@dataclass
class Timer:

    _t0: float = field(default_factory=time.time)
    _laps: Dict[str, float] = field(default_factory=dict)

    def lap(self, label: str) -> float:
        now = time.time()
        elapsed = now - self._t0
        self._t0 = now
        self._laps[label] = self._laps.get(label, 0.0) + elapsed
        return round(elapsed, 3)

    def total(self, label: str) -> float:
        return round(self._laps.get(label, 0.0), 3)

    def reset(self) -> None:
        self._t0 = time.time()
