# Author: Laure Berti-Equille (laure.berti@ird.fr)
# Paper:  Berti-Equille, L. (2026). Where Is the Truth When AI Agents Disagree?
#         Dependence-Aware Belief Propagation for Unsupervised Truth Discovery.
#         Paper accepted at IEEE BigData 2026.

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from .data import ABSTAIN


@dataclass
class QARecord:
    qid: str
    question: str
    choices: Dict[str, str]
    value_set: List[str]
    gold: str
    context: str = ""


def _mc_record(qid, question, options: List[str], gold_idx: int, context: str = "") -> QARecord:
    labels = [chr(ord("A") + i) for i in range(len(options))]
    choices = {lab: opt for lab, opt in zip(labels, options)}
    return QARecord(qid=qid, question=question, choices=choices,
                    value_set=labels, gold=labels[gold_idx], context=context)


def build_prompt(rec: QARecord, context: str = "") -> str:
    lines = []
    if context:
        lines += ["Context:", context.strip(), ""]
    lines += [rec.question, ""]
    for lab in rec.value_set:
        lines.append(f"{lab}. {rec.choices[lab]}")
    lines.append("")
    lines.append(
        "Answer with ONLY the single letter/label of the correct option "
        f"(one of: {', '.join(rec.value_set)}). No explanation."
    )
    return "\n".join(lines)


def parse_answer(text: str, value_set: List[str]) -> str:
    if not text:
        return ABSTAIN
    t = text.strip()
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL | re.IGNORECASE).strip()
    upper_set = {v.upper(): v for v in value_set}
    m = re.match(r"^\W*([A-Za-z]+|yes|no|supports|refutes)\b", t, flags=re.IGNORECASE)
    if m and m.group(1).upper() in upper_set:
        return upper_set[m.group(1).upper()]
    for v in value_set:
        if re.search(rf"\b{re.escape(v)}\b", t, flags=re.IGNORECASE):
            return v
    return ABSTAIN


def load_dataset_records(name: str, max_questions: int) -> List[QARecord]:
    from datasets import load_dataset

    name = name.lower()
    recs: List[QARecord] = []

    if name == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        ntot = len(ds)
        step = max(1, ntot // max_questions)
        for i in range(0, ntot, step):
            if len(recs) >= max_questions:
                break
            ex = ds[i]
            recs.append(_mc_record(f"mmlu-{i}", ex["question"],
                                   list(ex["choices"]), int(ex["answer"])))
    elif name in ("truthfulqa", "truthfulqa_mc1"):
        ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
        for i, ex in enumerate(ds):
            if len(recs) >= max_questions:
                break
            mc1 = ex["mc1_targets"]
            opts, labels = mc1["choices"], mc1["labels"]
            gold_idx = int(labels.index(1))
            recs.append(_mc_record(f"tqa-{i}", ex["question"], list(opts), gold_idx))
    elif name == "fever":
        ds = load_dataset("fever", "v1.0", split="labelled_dev")
        lab2idx = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
        opts = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        for i, ex in enumerate(ds):
            if len(recs) >= max_questions:
                break
            if ex["label"] not in lab2idx:
                continue
            recs.append(_mc_record(f"fever-{i}",
                                   f"Claim: {ex['claim']}", opts, lab2idx[ex["label"]]))
    elif name == "boolq":
        ds = load_dataset("google/boolq", split="validation")
        opts = ["yes", "no"]
        for i, ex in enumerate(ds):
            if len(recs) >= max_questions:
                break
            gold_idx = 0 if ex["answer"] else 1
            recs.append(_mc_record(f"boolq-{i}", ex["question"], opts, gold_idx,
                                   context=ex["passage"]))
    elif name == "sciq":
        ds = load_dataset("allenai/sciq", split="test")
        for i, ex in enumerate(ds):
            if len(recs) >= max_questions:
                break
            opts = [ex["correct_answer"], ex["distractor1"],
                    ex["distractor2"], ex["distractor3"]]
            order = [(i + k) % 4 for k in range(4)]
            shuffled = [opts[o] for o in order]
            gold_idx = order.index(0)
            recs.append(_mc_record(f"sciq-{i}", ex["question"], shuffled, gold_idx,
                                   context=ex.get("support", "")))
    elif name in ("arc", "arc_challenge", "arc-challenge"):
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        for i, ex in enumerate(ds):
            if len(recs) >= max_questions:
                break
            labels = list(ex["choices"]["label"])
            texts = list(ex["choices"]["text"])
            ak = ex["answerKey"]
            if ak not in labels:
                continue
            gold_idx = labels.index(ak)
            recs.append(_mc_record(f"arc-{i}", ex["question"], texts, gold_idx))
    else:
        raise ValueError(f"Unknown dataset {name!r}")

    return recs
