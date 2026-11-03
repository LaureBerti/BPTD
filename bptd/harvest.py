# Author: Laure Berti-Equille (laure.berti@ird.fr)
# Paper:  Berti-Equille, L. (2026). Where Is the Truth When AI Agents Disagree?
#         Dependence-Aware Belief Propagation for Unsupervised Truth Discovery.
#         Paper accepted at IEEE BigData 2026.

from __future__ import annotations

import csv
import os
import time
from typing import List

from .config import AgentsConfig, AgentSpec, DataConfig
from .data import ABSTAIN
from .qa import QARecord, load_dataset_records
from .agents import ChatFn, build_agents, DirectAgent


def make_ollama_chat_fn(cfg: AgentsConfig) -> ChatFn:
    from openai import OpenAI

    client = OpenAI(api_key=cfg.api_key, base_url=cfg.api_base)

    def chat_fn(model, messages, temperature, max_tokens):
        resp = client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
            timeout=cfg.request_timeout_s,
        )
        return resp.choices[0].message.content or ""

    return chat_fn


def _resolve_specs(cfg: AgentsConfig) -> List[AgentSpec]:
    if cfg.specs:
        return cfg.specs
    return [AgentSpec(name=f"direct:{m}", type="direct", model=m) for m in cfg.models]


def harvest(data_cfg: DataConfig, agents_cfg: AgentsConfig) -> str:
    folder = os.path.join(data_cfg.folder, data_cfg.name)
    os.makedirs(folder, exist_ok=True)
    gold_path = os.path.join(folder, "gold.csv")
    ans_path = os.path.join(folder, "answers.csv")

    recs = load_dataset_records(data_cfg.name, data_cfg.max_questions or 500)

    with open(gold_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "gold_value", "value_set"])
        for r in recs:
            w.writerow([r.qid, r.gold, "|".join(r.value_set)])

    chat_fn = make_ollama_chat_fn(agents_cfg)
    agents = build_agents(_resolve_specs(agents_cfg), chat_fn, records=recs,
                          max_tokens=agents_cfg.max_tokens)
    print(f"[harvest] {data_cfg.name}: {len(recs)} questions x {len(agents)} agents "
          f"({', '.join(a.name for a in agents)})")

    done = set()
    if os.path.isfile(ans_path):
        with open(ans_path, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["qid"], row["agent"]))
        out = open(ans_path, "a", newline="")
        writer = csv.writer(out)
    else:
        out = open(ans_path, "w", newline="")
        writer = csv.writer(out)
        writer.writerow(["qid", "agent", "raw", "value", "parse_ok", "latency_s",
                         "agent_type", "base_model"])

    total = len(recs) * len(agents)
    n = 0
    t_start = time.time()
    try:
        for r in recs:
            for agent in agents:
                n += 1
                if (r.qid, agent.name) in done:
                    continue
                ans = agent.answer(r)
                writer.writerow([r.qid, agent.name, ans.raw.replace("\n", " ")[:300],
                                 ans.value, int(ans.value != ABSTAIN),
                                 round(ans.latency_s, 3), ans.meta.get("type", ""),
                                 getattr(agent, "base_model", "")])
                out.flush()
                if n % 50 == 0:
                    el = time.time() - t_start
                    print(f"  [{n}/{total}] {r.qid} | {agent.name} -> {ans.value} "
                          f"({ans.latency_s:.1f}s, elapsed {el/60:.1f} min)")
    finally:
        out.close()
    print(f"[harvest] {data_cfg.name}: wrote {ans_path}")
    return folder
