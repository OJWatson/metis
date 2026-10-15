"""Train and interrogate neural surrogates from simulation banks.

The same objects power Python calls and Hydra-configured experiments.
"""

from metis.data import DatasetSplit, SimulationDataset, Standardizer
from metis.datasets import load_example
from metis.design import assess_coverage, suggest_simulations
from metis.evaluation import EvaluationReport, evaluate
from metis.surrogate import Surrogate, TrainingConfig

__version__ = "0.3.0"
__all__ = [
    "DatasetSplit", "SimulationDataset", "Standardizer", "Surrogate",
    "TrainingConfig", "EvaluationReport", "evaluate", "load_example",
    "assess_coverage", "suggest_simulations",
]
