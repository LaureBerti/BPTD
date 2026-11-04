# BP-TD: Dependence-Aware Belief Propagation for Unsupervised Truth Discovery among AI Agents

Code and cached data for the paper **"Where Is the Truth When AI Agents Disagree?
Dependence-Aware Belief Propagation for Unsupervised Truth Discovery"** by
**Laure Berti-Equille** (IRD, ESPACE-DEV, Montpellier, France).
**Paper accepted at IEEE BigData 2026** (short paper).

**Method in one paragraph.** Several LLM agents answer the same multiple-choice questions;
many are variants of one base model, so their errors are correlated. BP-TD infers the true
answers and each agent's reliability without labels, by mean-field variational inference on a
one-coin Dawid–Skene model in which each agent's vote carries a **decorrelation weight** `w_a`:
agents that share a base model (`family`) or that agree far more than chance (`cooccur`,
threshold θ) count together as about one vote. The first iteration is exactly a `w_a`-weighted
plurality vote; later iterations add per-agent reliability estimates.

**Main findings.** On MMLU, BP-TD beats majority vote without labels (0.716 vs 0.624 on 160
questions; 0.707 vs 0.674 on 10,000). Over 2,852 sub-pools of four benchmarks, decorrelation
helps when the largest correlated group of agents is weaker than the rest and hurts when it is
the strongest; no label-free statistic we tested tells the two cases apart. With 20 labelled
questions, choosing the single best agent beats every aggregator tested.

## Setup (CPU only)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"
bash scripts/get_care.sh          # only for the CARE comparison (exp=concurrent)
```

All numbers recompute on a laptop CPU from the cached answers in `data/`; no model is queried.
The data are at most 10,000 questions × 14 agents, so no GPU is needed.
`scripts/get_care.sh` clones the CARE code (Zhao et al., ICML 2026) at the commit used in the
paper into `experiments/third_party/` (git-ignored): the CARE repository has no licence, so it
is not redistributed here.

## Data

`data/<dataset>/answers.csv` holds one row per (question, agent) with the parsed answer and the
agent's base model; `gold.csv` holds the gold answer and the answer options. Loading keeps only
questions that at least one agent answered.

| Folder | Questions | Agents | Source benchmark (licence) |
|---|---|---|---|
| `mmlu` | 160 | 13 answering (+1 never parses) | MMLU, `cais/mmlu` (MIT) |
| `sciq` | 160 | same | SciQ, `allenai/sciq` (CC BY-NC 3.0) |
| `truthfulqa` | 100 | same | TruthfulQA MC1 (Apache 2.0) |
| `arc` | 160 | same | ARC-Challenge, `allenai/ai2_arc` (CC BY-SA 4.0) |
| `mmlu_vllm_5k`, `mmlu_vllm_10k` | 5,000 / 10,000 | same, vLLM fp16 | MMLU (MIT) |
| `boolq` | 128 | same | BoolQ (CC BY-SA 3.0) — pilot only; 11 of 13 agents did not see the passage |

Only the derived agent answers and gold labels are redistributed, not the benchmark texts.
The agent `openclaw` never produced a parseable answer and is kept as an all-abstain row; it
casts no vote.

## Reproduce the paper

| Paper item | Command | Output |
|---|---|---|
| Table I (agent pool) | `python experiments/rev_experiments.py exp=agents` | `outputs/E10_agents.csv` |
| Table II (all methods, 4 benchmarks) | `python experiments/rev_experiments.py exp=main` and `exp=concurrent` | `E1_main_*.csv`, `E11_concurrent_*.csv` |
| Table III (MMLU 5k/10k) | `python experiments/rev_experiments.py exp=scale` | `E1_main_mmlu_vllm_*.csv` |
| θ sensitivity, Otsu θ (Fig. 2a data) | `python experiments/rev_experiments.py exp=theta` | `E2_theta.csv` |
| Sub-pools, label-free diagnostic (Fig. 2b data) | `python experiments/rev_experiments.py exp=subpools` | `E3_subpools.csv` |
| Label budget (Fig. 2c data) | `python experiments/rev_experiments.py exp=budget` | `E6_budget.csv` |
| Strong-bloc copies | `python experiments/rev_experiments.py exp=strongbloc` | `E4_strongbloc.csv` |
| EM baselines on independent agents | `python experiments/rev_experiments.py exp=ds_sanity` | `E5_ds_sanity.csv` |
| Seeding | `python experiments/rev_experiments.py exp=seeding` | `E8_seeding.csv` |
| Reliability over the 40 test halves | `python experiments/rev_experiments.py exp=rho` | `E12_rho_splits.csv` |
| Summaries and LaTeX tables | `python experiments/analyze_revision.py`, `make_tables.py` | `outputs/` |
| Reliability recovery (full set) | `python scripts/run_exp_v3_freebies.py` | `outputs/` |
| Gain vs disagreement quartiles | `python bptd_main.py mode=disagreement data.name=mmlu` (and `sciq`, `truthfulqa`) | `outputs/` |
| Wrong-bloc stress test (ARC) | `python scripts/run_exp_W2B_stress.py` | `outputs/` |
| Runtime scaling (synthetic) | `python scripts/analyze_E3_scalability.py` | `outputs/` |

The result CSVs used in the paper are committed under `outputs/`, so every
number can be checked without re-running anything. Figure-drawing scripts are not part of
this repository; every figure in the paper plots columns of these CSVs. All experiment settings are Hydra fields;
override them on the command line (e.g. `datasets=[mmlu] n_splits=10`) or dump the defaults with
`python experiments/rev_experiments.py --write-config` (→ `conf/rev_experiments.yaml`).
Most experiments take minutes; `exp=subpools` (713 pools per dataset) and `exp=concurrent`
(CARE's convex solver, 11 values of γ per split) take tens of minutes per dataset.

**Protocol.** 40 random validation/test splits per dataset (50/50). Methods with a
hyperparameter pick it on the validation half; label-free methods (`… no labels` in the paper)
fix θ = 0.5, use the Otsu threshold of the pairwise agreements, or use provenance. One test for
every comparison: the Nadeau–Bengio corrected resampled-t over the splits, Holm-corrected.

**Defaults to know.** `bptd_main.py` uses loopy inference by default (`bp.mode=loopy`: damping 0.3,
signed reliability smoothing λ = 0.4, stop at belief change < 1e-4, at most 100 iterations);
`bp.mode=onepass` gives the weighted vote. `data.max_questions` defaults to all cached
questions; set it to cap a run.

## Code layout

```
bptd/                    core package
  bp.py                  BP-TD: mean-field updates + decorrelation weights (family / cooccur / shared_error)
  baselines.py           MV, WeightedMV, Dawid–Skene, LTM, TruthFinder, Investment, Sums, CRH, CATD,
                         SSTF-IV/VI, ClusterVote, ACCU, DEPEN, PochampallyCorr, IsingMF
  data.py, metrics.py    answer loader, metrics
  agents.py, qa.py, harvest.py, harvest_vllm.py   agent pool and (optional) answer harvesting
  config.py              dataclass config (Hydra / OmegaConf)
bptd_main.py             CLI: demo / harvest / tuned / reliability / disagreement / c5
experiments/             camera-ready experiments (Tables I–III, Fig. 2, Secs. V–VI)
  rev_experiments.py     all experiments (Hydra; exp=...)
  concurrent_methods.py  CARE-SVD (authors' code), ISP and OW-I (from their equations)
  analyze_revision.py, make_tables.py
scripts/                 earlier analysis scripts (E1–E4, stress test) and get_care.sh (fetch CARE)
outputs/                 result CSVs behind the paper's numbers (all scripts write here)
data/                    cached agent answers
```

## Re-harvesting answers (optional, needs local models)

Agents run locally through Ollama (`http://localhost:11434`): `python bptd_main.py mode=harvest
data.name=mmlu data.max_questions=160`. For large harvests on a GPU, `bptd/harvest_vllm.py` is a
batched backend (`agents.backend=vllm`, one model per process via `BPTD_VLLM_ONLY_MODEL=<tag>`;
`qwen2.5:14b` in fp16 needs a 40 GB GPU; a Hugging Face token is needed for gated models).

## Citation

```bibtex
@inproceedings{bertiequille2026bptd,
  title     = {Where Is the Truth When {AI} Agents Disagree? Dependence-Aware Belief Propagation
               for Unsupervised Truth Discovery},
  author    = {Berti-Equille, Laure},
  booktitle = {Proceedings of the IEEE International Conference on Big Data (BigData 2026)},
  year      = {2026},
  note      = {Short paper}
}

@misc{bertiequille2026bptdextended,
  title         = {Where Is the Truth When {AI} Agents Disagree? Dependence-Aware Belief Propagation
                   for Unsupervised Truth Discovery (Extended Version)},
  author        = {Berti-Equille, Laure},
  year          = {2026},
  eprint        = {2611.04805},
  archivePrefix = {arXiv},
  doi           = {10.48550/arXiv.2611.04805}
}
```

Extended version with all analyses: arXiv [2611.04805](https://arxiv.org/abs/2611.04805)
(DOI [10.48550/arXiv.2611.04805](https://doi.org/10.48550/arXiv.2611.04805)).

Source code: <https://github.com/LaureBerti/BPTD>.
Code and data archive: Zenodo, DOI [10.5281/zenodo.21294747](https://doi.org/10.5281/zenodo.21294747)
(all versions; it resolves to the latest release, v2.0.0 for this camera-ready code). Machine-readable citation: `CITATION.cff`.

## License

Code: MIT — see [`LICENSE`](LICENSE). © 2026 Laure Berti-Equille. Derived benchmark data keep
the licence of their source benchmark (table above).
