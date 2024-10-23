from datetime import datetime
import numpy as np
from typing import Iterable, List
import pandas as pd

from fromhopetoheuristics.utils.maxcut_utils import provide_random_maxcut_QUBO
from fromhopetoheuristics.utils.spectral_gap_calculator import annealing
import logging

log = logging.getLogger(__name__)


def run_maxcut_annealing(
    seed: int,
    num_anneal_fractions: int,
    maxcut_n_qubits: int,
    maxcut_graph_density: float,
):
    fractions = np.linspace(0, 1, num=num_anneal_fractions, endpoint=True)

    log.info(
        f"Running QAOA maxcut for n={maxcut_n_qubits} "
        f"with density={maxcut_graph_density}"
    )

    qubo = provide_random_maxcut_QUBO(maxcut_n_qubits, maxcut_n_qubits, seed)
    results = annealing(
        qubo,
        fractions,
    )

    results = pd.DataFrame.from_records(results)
    return {"results": results}
