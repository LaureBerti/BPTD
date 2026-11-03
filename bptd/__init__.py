# Author: Laure Berti-Equille (laure.berti@ird.fr)
# Paper:  Berti-Equille, L. (2026). Where Is the Truth When AI Agents Disagree?
#         Dependence-Aware Belief Propagation for Unsupervised Truth Discovery.
#         Paper accepted at IEEE BigData 2026.

from .config import (
    BPTDConfig,
    AgentsConfig,
    DataConfig,
    BPConfig,
    OutputConfig,
    default_config,
)
from .data import TDData, load_td_data, make_synthetic
from .bp import run_bptd, BPTDResult

__all__ = [
    "BPTDConfig",
    "AgentsConfig",
    "DataConfig",
    "BPConfig",
    "OutputConfig",
    "default_config",
    "TDData",
    "load_td_data",
    "make_synthetic",
    "run_bptd",
    "BPTDResult",
]

__version__ = "0.1.0"
