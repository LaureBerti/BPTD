# Author: Laure Berti-Equille (laure.berti@ird.fr)
# Paper:  Berti-Equille, L. (2026). Where Is the Truth When AI Agents Disagree?
#         Dependence-Aware Belief Propagation for Unsupervised Truth Discovery.
#         Paper accepted at IEEE BigData 2026.

from __future__ import annotations

import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from .data import ABSTAIN
from .qa import QARecord, build_prompt, parse_answer

ChatFn = Callable[[str, List[dict], float, int], str]


@dataclass
class AgentAnswer:
    raw: str
    value: str
    latency_s: float
    meta: Dict[str, str] = field(default_factory=dict)


class Agent:

    name: str = "agent"

    def answer(self, rec: QARecord) -> AgentAnswer:
        raise NotImplementedError


class _LLMAgent(Agent):
    def __init__(self, name: str, model: str, chat_fn: ChatFn, max_tokens: int = 64,
                 include_context: bool = True):
        self.name = name
        self.model = model
        self.base_model = model.split(":")[0]
        self.chat_fn = chat_fn
        self.max_tokens = max_tokens
        self.include_context = include_context

    def _call(self, messages: List[dict], temperature: float) -> str:
        try:
            return self.chat_fn(self.model, messages, temperature, self.max_tokens) or ""
        except Exception as exc:
            return f"<ERROR:{type(exc).__name__}>"


class DirectAgent(_LLMAgent):

    def answer(self, rec: QARecord) -> AgentAnswer:
        t0 = time.time()
        ctx = rec.context if self.include_context else ""
        raw = self._call([{"role": "user", "content": build_prompt(rec, ctx)}], 0.0)
        val = parse_answer(raw, rec.value_set)
        return AgentAnswer(raw[:300], val, time.time() - t0, {"type": "direct"})


class SelfConsistencyAgent(_LLMAgent):

    def __init__(self, name, model, chat_fn, k: int = 5,
                 temperature: float = 0.7, max_tokens: int = 64):
        super().__init__(name, model, chat_fn, max_tokens)
        self.k = k
        self.temperature = temperature

    def answer(self, rec: QARecord) -> AgentAnswer:
        t0 = time.time()
        prompt = build_prompt(rec, rec.context if self.include_context else "")
        votes: List[str] = []
        for _ in range(self.k):
            raw = self._call([{"role": "user", "content": prompt}], self.temperature)
            v = parse_answer(raw, rec.value_set)
            if v != ABSTAIN:
                votes.append(v)
        if not votes:
            return AgentAnswer("", ABSTAIN, time.time() - t0,
                               {"type": "self_consistency", "k": str(self.k)})
        top, cnt = Counter(votes).most_common(1)[0]
        return AgentAnswer(f"vote {top} {cnt}/{self.k}", top, time.time() - t0,
                           {"type": "self_consistency", "agree": f"{cnt}/{self.k}"})


class PersonaAgent(_LLMAgent):

    def __init__(self, name, model, chat_fn, persona: str, max_tokens: int = 64):
        super().__init__(name, model, chat_fn, max_tokens)
        self.persona = persona

    def answer(self, rec: QARecord) -> AgentAnswer:
        t0 = time.time()
        ctx = rec.context if self.include_context else ""
        msgs = [{"role": "system", "content": self.persona},
                {"role": "user", "content": build_prompt(rec, ctx)}]
        raw = self._call(msgs, 0.0)
        val = parse_answer(raw, rec.value_set)
        return AgentAnswer(raw[:300], val, time.time() - t0,
                           {"type": "persona", "persona": self.persona[:40]})


class RagAgent(_LLMAgent):

    def __init__(self, name, model, chat_fn, retriever: "TfidfRetriever",
                 top_k: int = 3, max_tokens: int = 64):
        super().__init__(name, model, chat_fn, max_tokens)
        self.retriever = retriever
        self.top_k = top_k

    def answer(self, rec: QARecord) -> AgentAnswer:
        t0 = time.time()
        query = (rec.context + " " + rec.question).strip()
        passages = self.retriever.retrieve(query, self.top_k)
        context = "\n".join(f"- {p}" for p in passages)
        raw = self._call([{"role": "user", "content": build_prompt(rec, context)}], 0.0)
        val = parse_answer(raw, rec.value_set)
        return AgentAnswer(raw[:300], val, time.time() - t0,
                           {"type": "rag", "n_ctx": str(len(passages))})


class SubprocessAgent(Agent):

    def __init__(self, name: str, command: Sequence[str], timeout_s: float = 120.0):
        self.name = name
        self.base_model = name
        self.command = list(command)
        self.timeout_s = timeout_s

    def answer(self, rec: QARecord) -> AgentAnswer:
        t0 = time.time()
        if not self.command:
            return AgentAnswer("", ABSTAIN, 0.0, {"type": "cli", "status": "unconfigured"})
        prompt = build_prompt(rec)
        argv = [tok.replace("{prompt}", prompt) for tok in self.command]
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=self.timeout_s, check=False,
            )
            raw = (proc.stdout or proc.stderr or "").strip()
        except Exception as exc:
            raw = f"<ERROR:{type(exc).__name__}>"
        val = parse_answer(raw, rec.value_set)
        return AgentAnswer(raw[:300], val, time.time() - t0, {"type": "cli"})


class TfidfRetriever:
    def __init__(self, corpus: List[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.corpus = [c for c in corpus if c and c.strip()]
        if not self.corpus:
            self.corpus = ["(empty corpus)"]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=50_000)
        self.matrix = self.vectorizer.fit_transform(self.corpus)

    def retrieve(self, query: str, top_k: int) -> List[str]:
        import numpy as np

        qv = self.vectorizer.transform([query])
        sims = (self.matrix @ qv.T).toarray().ravel()
        idx = np.argsort(-sims)[:top_k]
        return [self.corpus[i] for i in idx if sims[i] > 0]


def build_corpus_from_records(records: List[QARecord]) -> List[str]:
    corpus = []
    for r in records:
        if r.context:
            corpus.extend(s.strip() for s in r.context.split(". ") if len(s.strip()) > 20)
    return corpus


def build_agents(specs, chat_fn: ChatFn, records: Optional[List[QARecord]] = None,
                 max_tokens: int = 64) -> List[Agent]:
    retriever = None
    if any(getattr(s, "type", "") == "rag" for s in specs) and records is not None:
        retriever = TfidfRetriever(build_corpus_from_records(records))

    agents: List[Agent] = []
    for s in specs:
        t = s.type
        if t == "direct":
            agents.append(DirectAgent(s.name, s.model, chat_fn, max_tokens))
        elif t == "self_consistency":
            agents.append(SelfConsistencyAgent(s.name, s.model, chat_fn,
                                               k=s.k, temperature=s.temperature,
                                               max_tokens=max_tokens))
        elif t == "persona":
            agents.append(PersonaAgent(s.name, s.model, chat_fn, s.persona, max_tokens))
        elif t == "rag":
            if retriever is None:
                retriever = TfidfRetriever(["(no corpus)"])
            agents.append(RagAgent(s.name, s.model, chat_fn, retriever,
                                   top_k=s.rag_top_k, max_tokens=max_tokens))
        elif t == "cli":
            agents.append(SubprocessAgent(s.name, s.command, s.request_timeout_s))
        else:
            raise ValueError(f"Unknown agent type {t!r} for agent {s.name!r}")
    return agents
