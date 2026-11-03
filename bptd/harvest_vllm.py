# Author: Laure Berti-Equille (laure.berti@ird.fr)
# Paper:  Berti-Equille, L. (2026). Where Is the Truth When AI Agents Disagree?
#         Dependence-Aware Belief Propagation for Unsupervised Truth Discovery.
#         Paper accepted at IEEE BigData 2026.

from __future__ import annotations

import csv
import os
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

from .config import AgentsConfig, AgentSpec, DataConfig
from .data import ABSTAIN
from .qa import QARecord, build_prompt, parse_answer, load_dataset_records
from .agents import (
    TfidfRetriever,
    build_corpus_from_records,
    build_agents,
)
from .harvest import _resolve_specs


DEFAULT_VLLM_MODEL_MAP: Dict[str, str] = {
    "qwen2.5:14b":       "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5:7b":        "Qwen/Qwen2.5-7B-Instruct",
    "mistral:7b":        "mistralai/Mistral-7B-Instruct-v0.3",
    "llama3.2:latest":   "meta-llama/Llama-3.2-3B-Instruct",
    "phi3.5:latest":     "microsoft/Phi-3.5-mini-instruct",
    "gemma2:2b":         "google/gemma-2-2b-it",
    "deepseek-r1:latest": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-r1:1.5b":   "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
}

_LLM_TYPES = {"direct", "self_consistency", "persona", "rag"}


def _hf_repo(ollama_tag: str, cfg: AgentsConfig) -> str:
    mapping = dict(DEFAULT_VLLM_MODEL_MAP)
    mapping.update(cfg.vllm_model_map or {})
    _env_map = os.environ.get("BPTD_VLLM_MODEL_MAP")
    if _env_map:
        import json
        mapping.update(json.loads(_env_map))
    if ollama_tag not in mapping:
        raise ValueError(
            f"No HF repo mapping for Ollama model {ollama_tag!r}. Add it to "
            f"`agents.vllm_model_map` (e.g. agents.vllm_model_map.'{ollama_tag}'=org/repo)."
        )
    return mapping[ollama_tag]


def _messages_for(spec: AgentSpec, rec: QARecord,
                  retriever: Optional[TfidfRetriever]) -> List[dict]:
    t = spec.type
    if t == "rag":
        assert retriever is not None
        query = (rec.context + " " + rec.question).strip()
        passages = retriever.retrieve(query, spec.rag_top_k)
        context = "\n".join(f"- {p}" for p in passages)
        return [{"role": "user", "content": build_prompt(rec, context)}]
    ctx = rec.context
    prompt = build_prompt(rec, ctx)
    if t == "persona":
        return [{"role": "system", "content": spec.persona},
                {"role": "user", "content": prompt}]
    return [{"role": "user", "content": prompt}]


def _base_model(spec: AgentSpec) -> str:
    return spec.model.split(":")[0]


def harvest_vllm(data_cfg: DataConfig, agents_cfg: AgentsConfig) -> str:
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        raise ImportError(
            "The vLLM backend requires the `vllm` package, which is not installed.\n"
            "Install it on a CUDA GPU machine with:\n"
            "    pip install vllm\n"
            "(vLLM needs an NVIDIA GPU + CUDA; it will not run on this Mac.)\n"
            f"Original import error: {exc}"
        ) from exc

    folder = os.path.join(data_cfg.folder, data_cfg.name)
    os.makedirs(folder, exist_ok=True)
    gold_path = os.path.join(folder, "gold.csv")
    ans_path = os.path.join(folder, "answers.csv")

    recs = load_dataset_records(data_cfg.name, data_cfg.max_questions or 500)
    rec_by_qid: Dict[str, QARecord] = {r.qid: r for r in recs}

    with open(gold_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "gold_value", "value_set"])
        for r in recs:
            w.writerow([r.qid, r.gold, "|".join(r.value_set)])

    specs = _resolve_specs(agents_cfg)
    llm_specs = [s for s in specs if s.type in _LLM_TYPES]
    other_specs = [s for s in specs if s.type not in _LLM_TYPES]

    retriever: Optional[TfidfRetriever] = None
    if any(s.type == "rag" for s in llm_specs):
        retriever = TfidfRetriever(build_corpus_from_records(recs))

    print(f"[harvest-vllm] {data_cfg.name}: {len(recs)} questions x {len(specs)} agents "
          f"({', '.join(s.name for s in specs)})")

    done: set = set()
    new_file = not os.path.isfile(ans_path)
    if not new_file:
        with open(ans_path, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["qid"], row["agent"]))
    out = open(ans_path, "a", newline="")
    writer = csv.writer(out)
    if new_file:
        writer.writerow(["qid", "agent", "raw", "value", "parse_ok", "latency_s",
                         "agent_type", "base_model"])

    def _write_row(qid: str, agent_name: str, raw: str, value: str,
                   latency_s: float, agent_type: str, base_model: str) -> None:
        writer.writerow([qid, agent_name, raw.replace("\n", " ")[:300],
                         value, int(value != ABSTAIN), round(latency_s, 3),
                         agent_type, base_model])
        out.flush()

    by_model: Dict[str, List[AgentSpec]] = {}
    for s in llm_specs:
        by_model.setdefault(s.model, []).append(s)

    only = os.environ.get("BPTD_VLLM_ONLY_MODEL") or None

    try:
        for model_tag, model_specs in by_model.items():
            if only is not None and only != model_tag:
                continue
            conversations: List[List[dict]] = []
            sp_list: List["SamplingParams"] = []
            meta: List[Tuple[str, str, str, str, str, int]] = []

            for spec in model_specs:
                bm = _base_model(spec)
                agent_type = "self_consistency" if spec.type == "self_consistency" else spec.type
                for r in recs:
                    if (r.qid, spec.name) in done:
                        continue
                    msgs = _messages_for(spec, r, retriever)
                    if spec.type == "self_consistency":
                        sp = SamplingParams(temperature=spec.temperature,
                                            max_tokens=agents_cfg.max_tokens,
                                            n=spec.k, seed=agents_cfg.vllm_seed)
                    else:
                        sp = SamplingParams(temperature=0.0,
                                            max_tokens=agents_cfg.max_tokens, n=1)
                    conversations.append(msgs)
                    sp_list.append(sp)
                    meta.append((r.qid, spec.name, agent_type, bm, spec.type, spec.k))

            if not conversations:
                print(f"[harvest-vllm]   {model_tag}: nothing to do (all cached).")
                continue

            hf_repo = _hf_repo(model_tag, agents_cfg)
            print(f"[harvest-vllm]   loading {model_tag} -> {hf_repo} "
                  f"({len(conversations)} prompts across {len(model_specs)} agents) ...")
            llm = LLM(
                model=hf_repo,
                dtype=agents_cfg.vllm_dtype,
                gpu_memory_utilization=agents_cfg.vllm_gpu_mem_util,
                max_model_len=agents_cfg.vllm_max_model_len,
                tensor_parallel_size=agents_cfg.vllm_tensor_parallel,
                seed=agents_cfg.vllm_seed,
            )

            t0 = time.time()
            outputs = llm.chat(conversations, sp_list)
            batch_s = time.time() - t0
            per_item = batch_s / max(1, len(outputs))
            print(f"[harvest-vllm]   {model_tag}: generated {len(outputs)} items "
                  f"in {batch_s/60:.1f} min ({per_item:.2f}s/item)")

            for out_i, (qid, agent_name, agent_type, bm, spec_type, k) in zip(outputs, meta):
                rec = rec_by_qid[qid]
                if spec_type == "self_consistency":
                    votes: List[str] = []
                    for comp in out_i.outputs:
                        v = parse_answer(comp.text, rec.value_set)
                        if v != ABSTAIN:
                            votes.append(v)
                    if not votes:
                        _write_row(qid, agent_name, "", ABSTAIN, per_item, agent_type, bm)
                    else:
                        top, cnt = Counter(votes).most_common(1)[0]
                        _write_row(qid, agent_name, f"vote {top} {cnt}/{k}", top,
                                   per_item, agent_type, bm)
                else:
                    raw = out_i.outputs[0].text if out_i.outputs else ""
                    val = parse_answer(raw, rec.value_set)
                    _write_row(qid, agent_name, raw[:300], val, per_item, agent_type, bm)

            del llm
            try:
                import gc
                import torch
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass

        if other_specs and only in (None, "__cli__"):
            other_agents = build_agents(other_specs, chat_fn=lambda *a, **k: "",
                                        records=recs, max_tokens=agents_cfg.max_tokens)
            for agent in other_agents:
                for r in recs:
                    if (r.qid, agent.name) in done:
                        continue
                    ans = agent.answer(r)
                    _write_row(r.qid, agent.name, ans.raw, ans.value,
                               ans.latency_s, ans.meta.get("type", ""),
                               getattr(agent, "base_model", ""))
    finally:
        out.close()

    print(f"[harvest-vllm] {data_cfg.name}: wrote {ans_path}")
    return folder
