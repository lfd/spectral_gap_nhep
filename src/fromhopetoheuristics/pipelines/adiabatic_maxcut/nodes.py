from datetime import datetime
import numpy as np
from typing import Dict
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
) -> Dict[str, pd.DataFrame]:
    """
    Runs the maxcut problem on the adiabatic quantum computer.

    Parameters
    ----------
    seed: int
        The seed used to generate the random maxcut problem.
    num_anneal_fractions: int
        The number of points in the annealing schedule to use.
    maxcut_n_qubits: int
        The number of qubits to use.
    maxcut_graph_density: float
        The density of the graph to use.

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary with a single key, "results", which contains a DataFrame
        with the results of the computation. The columns of the DataFrame are
        the different points in the annealing schedule, and the rows are the
        different possible states of the quantum computer.
    """
    fractions = np.linspace(0, 1, num=num_anneal_fractions, endpoint=True)

    log.info(
        f"Running maxcut for n={maxcut_n_qubits} "
        f"with density={maxcut_graph_density}"
    )

    qubo = provide_random_maxcut_QUBO(maxcut_n_qubits, maxcut_n_qubits, seed)
    results = annealing(
        qubo,
        fractions,
    )

    results = pd.DataFrame.from_records(results)
    return {"results": results}
