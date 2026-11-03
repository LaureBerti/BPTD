# Author: Laure Berti-Equille (laure.berti@ird.fr)
# Paper:  Berti-Equille, L. (2026). Where Is the Truth When AI Agents Disagree?
#         Dependence-Aware Belief Propagation for Unsupervised Truth Discovery.
#         Paper accepted at IEEE BigData 2026.

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

ABSTAIN = "<ABSTAIN>"


@dataclass
class TDData:

    questions: List[str]
    agents: List[str]
    domains: Dict[str, List[str]]
    answers: np.ndarray
    gold: np.ndarray
    base_model: Optional[List[str]] = None

    _agent_index: Dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.answers = np.asarray(self.answers, dtype=np.int64)
        self.gold = np.asarray(self.gold, dtype=np.int64)
        self._agent_index = {a: i for i, a in enumerate(self.agents)}

    @property
    def n_questions(self) -> int:
        return len(self.questions)

    @property
    def n_agents(self) -> int:
        return len(self.agents)

    def domain_size(self, q_idx: int) -> int:
        return len(self.domains[self.questions[q_idx]])

    def has_answer(self, a_idx: int, q_idx: int) -> bool:
        return self.answers[a_idx, q_idx] >= 0

    def gold_mask(self) -> np.ndarray:
        return self.gold >= 0

    def measured_agent_accuracy(self) -> np.ndarray:
        acc = np.full(self.n_agents, np.nan)
        gmask = self.gold_mask()
        for a in range(self.n_agents):
            ans = self.answers[a]
            valid = gmask & (ans >= 0)
            if valid.sum() == 0:
                continue
            acc[a] = float((ans[valid] == self.gold[valid]).mean())
        return acc

    def subset_agents(self, agent_ids: List[str]) -> "TDData":
        idx = [self._agent_index[a] for a in agent_ids]
        return TDData(
            questions=list(self.questions),
            agents=list(agent_ids),
            domains=self.domains,
            answers=self.answers[idx, :].copy(),
            gold=self.gold.copy(),
            base_model=([self.base_model[i] for i in idx] if self.base_model else None),
        )


def signed_agreement_matrix(data: TDData) -> np.ndarray:
    m = data.n_agents
    W = np.zeros((m, m), dtype=np.float64)
    ans = data.answers
    for a in range(m):
        aa = ans[a]
        for b in range(a + 1, m):
            bb = ans[b]
            both = (aa >= 0) & (bb >= 0)
            co = int(both.sum())
            if co == 0:
                continue
            agree = int((aa[both] == bb[both]).sum())
            w = (2 * agree - co) / co
            W[a, b] = w
            W[b, a] = w
    return W


def load_td_data(folder: str, max_questions: Optional[int] = None) -> TDData:
    import csv

    gold_path = os.path.join(folder, "gold.csv")
    ans_path = os.path.join(folder, "answers.csv")
    if not os.path.isfile(gold_path) or not os.path.isfile(ans_path):
        raise FileNotFoundError(f"Expected gold.csv and answers.csv in {folder!r}")

    domains: Dict[str, List[str]] = {}
    gold_value: Dict[str, str] = {}
    questions: List[str] = []
    with open(gold_path, newline="") as f:
        for row in csv.DictReader(f):
            qid = row["qid"]
            vs = [v for v in row["value_set"].split("|") if v != ""]
            domains[qid] = vs
            gold_value[qid] = row.get("gold_value", "")
            questions.append(qid)
    if max_questions is not None:
        questions = questions[:max_questions]
    qset = set(questions)

    raw: Dict[str, Dict[str, str]] = {}
    agents_seen: List[str] = []
    agent_base: Dict[str, str] = {}
    with open(ans_path, newline="") as f:
        for row in csv.DictReader(f):
            qid = row["qid"]
            if qid not in qset:
                continue
            agent = row["agent"]
            if agent not in raw.setdefault(qid, {}):
                raw[qid][agent] = row["value"]
            if agent not in agents_seen:
                agents_seen.append(agent)
            bm = row.get("base_model")
            if bm and agent not in agent_base:
                agent_base[agent] = bm

    agents = agents_seen
    base_model = [agent_base.get(a, a) for a in agents] if agent_base else None
    a_idx = {a: i for i, a in enumerate(agents)}
    q_idx = {q: i for i, q in enumerate(questions)}
    m, n = len(agents), len(questions)
    answers = np.full((m, n), -1, dtype=np.int64)
    gold = np.full(n, -1, dtype=np.int64)

    for q in questions:
        dom = domains[q]
        dom_idx = {v: i for i, v in enumerate(dom)}
        qi = q_idx[q]
        gv = gold_value.get(q, "")
        if gv in dom_idx:
            gold[qi] = dom_idx[gv]
        for agent, val in raw.get(q, {}).items():
            if val == ABSTAIN or val not in dom_idx:
                continue
            answers[a_idx[agent], qi] = dom_idx[val]

    td = TDData(questions=questions, agents=agents, domains=domains,
                answers=answers, gold=gold, base_model=base_model)
    answered = (td.answers >= 0).any(axis=0)
    if not answered.all():
        keep = list(np.where(answered)[0])
        td = TDData(questions=[td.questions[i] for i in keep], agents=td.agents,
                    domains=td.domains, answers=td.answers[:, keep].copy(),
                    gold=td.gold[keep].copy(), base_model=td.base_model)
    return td


def make_synthetic(
    n_questions: int = 500,
    n_agents: int = 8,
    n_values: int = 4,
    reliability_low: float = 0.30,
    reliability_high: float = 0.92,
    missing_rate: float = 0.05,
    seed: int = 7,
) -> TDData:
    rng = np.random.default_rng(seed)
    agents = [f"agent_{i:02d}" for i in range(n_agents)]
    reliabilities = np.linspace(reliability_low, reliability_high, n_agents)

    questions = [f"q{j:05d}" for j in range(n_questions)]
    values = [f"v{k}" for k in range(n_values)]
    domains = {q: list(values) for q in questions}

    gold = rng.integers(0, n_values, size=n_questions)
    answers = np.full((n_agents, n_questions), -1, dtype=np.int64)
    for a in range(n_agents):
        r = reliabilities[a]
        for q in range(n_questions):
            if rng.random() < missing_rate:
                continue
            if rng.random() < r:
                answers[a, q] = gold[q]
            else:
                wrong = rng.integers(0, n_values - 1)
                if wrong >= gold[q]:
                    wrong += 1
                answers[a, q] = wrong

    data = TDData(questions=questions, agents=agents, domains=domains,
                  answers=answers, gold=gold.astype(np.int64))
    data.__dict__["true_reliability"] = reliabilities
    return data
