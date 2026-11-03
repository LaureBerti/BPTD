# Author: Laure Berti-Equille (laure.berti@ird.fr)
# Paper:  Berti-Equille, L. (2026). Where Is the Truth When AI Agents Disagree?
#         Dependence-Aware Belief Propagation for Unsupervised Truth Discovery.
#         Paper accepted at IEEE BigData 2026.

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

try:
    from omegaconf import OmegaConf

    _HAS_OMEGACONF = True
except Exception:
    _HAS_OMEGACONF = False


DEFAULT_AGENTS: List[str] = [
    "qwen2.5:14b",
    "deepseek-r1:latest",
    "qwen2.5:7b",
    "mistral:7b",
    "llama3.2:latest",
    "phi3.5:latest",
    "gemma2:2b",
    "deepseek-r1:1.5b",
]


@dataclass
class AgentSpec:

    name: str
    type: str = "direct"
    model: str = "llama3.2:latest"
    k: int = 5
    temperature: float = 0.7
    persona: str = ""
    rag_top_k: int = 3
    command: List[str] = field(default_factory=list)
    request_timeout_s: float = 120.0


_PERSONAS = {
    "expert": "You are a meticulous domain expert. Answer factual questions precisely.",
    "skeptic": "You are a careful skeptic. Distrust common misconceptions; reason before answering.",
    "cautious": "You are a cautious assistant. Prefer the most defensible, well-evidenced option.",
}


def default_agent_specs() -> List["AgentSpec"]:
    specs: List[AgentSpec] = []
    _direct = [m for m in DEFAULT_AGENTS if not m.startswith("deepseek-r1")]
    for m in _direct:
        specs.append(AgentSpec(name=f"direct:{m}", type="direct", model=m))
    for m in ("llama3.2:latest", "qwen2.5:7b"):
        specs.append(AgentSpec(name=f"sc:{m}", type="self_consistency", model=m, k=5, temperature=0.7))
    for pname, ptext in _PERSONAS.items():
        specs.append(AgentSpec(name=f"persona:{pname}", type="persona",
                               model="llama3.2:latest", persona=ptext))
    for m in ("llama3.2:latest", "qwen2.5:7b"):
        specs.append(AgentSpec(name=f"rag:{m}", type="rag", model=m, rag_top_k=3))
    specs.append(AgentSpec(name="openclaw", type="cli", command=[]))
    return specs


@dataclass
class AgentsConfig:

    specs: List[AgentSpec] = field(default_factory=default_agent_specs)
    models: List[str] = field(default_factory=lambda: list(DEFAULT_AGENTS))
    api_base: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    temperature: float = 0.0
    max_tokens: int = 64
    request_timeout_s: float = 120.0

    backend: str = "ollama"
    vllm_dtype: str = "auto"
    vllm_gpu_mem_util: float = 0.90
    vllm_max_model_len: Optional[int] = None
    vllm_tensor_parallel: int = 1
    vllm_seed: int = 42
    vllm_model_map: Dict[str, str] = field(default_factory=dict)


@dataclass
class DataConfig:

    name: str = "synthetic"
    folder: str = "data"
    max_questions: Optional[int] = None
    syn_n_questions: int = 500
    syn_n_agents: int = 8
    syn_n_values: int = 4
    syn_reliability_low: float = 0.30
    syn_reliability_high: float = 0.92
    syn_missing_rate: float = 0.05
    syn_seed: int = 7


@dataclass
class BPConfig:

    signed: bool = True
    dependence_aware: bool = True
    dep_mode: str = "cooccur"
    dep_threshold: float = 0.5
    mode: str = "loopy"
    max_iter: int = 100
    eps: float = 1e-4
    damping: float = 0.30
    alpha0: float = 2.0
    beta0: float = 1.0
    sign_lambda: float = 0.40
    seed_fraction: float = 0.0
    seed_rng: int = 42


@dataclass
class OutputConfig:
    dir: str = "outputs"
    prefix: str = "bptd"
    dpi: int = 150


@dataclass
class BPTDConfig:
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    bp: BPConfig = field(default_factory=BPConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    mode: str = "compare"
    seeds: List[int] = field(default_factory=lambda: [42, 123, 7, 2024, 31])


def default_config() -> BPTDConfig:
    return BPTDConfig()


def to_yaml(cfg: BPTDConfig) -> str:
    if _HAS_OMEGACONF:
        return OmegaConf.to_yaml(OmegaConf.structured(cfg))
    import json

    return json.dumps(asdict(cfg), indent=2)


def write_default_yaml(path: str = "conf/bptd.yaml") -> str:
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(to_yaml(default_config()))
    return path


def apply_overrides(cfg: BPTDConfig, overrides: Optional[List[str]]) -> BPTDConfig:
    if not overrides:
        return cfg
    if _HAS_OMEGACONF:
        base = OmegaConf.structured(cfg)
        merged = OmegaConf.merge(base, OmegaConf.from_dotlist(list(overrides)))
        return OmegaConf.to_object(merged)

    def _coerce(v: str):
        low = v.lower()
        if low in ("true", "false"):
            return low == "true"
        if v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            return [_coerce(x.strip()) for x in inner.split(",")] if inner else []
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            return v

    for ov in overrides:
        if "=" not in ov:
            continue
        key, val = ov.split("=", 1)
        obj = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], _coerce(val))
    return cfg
