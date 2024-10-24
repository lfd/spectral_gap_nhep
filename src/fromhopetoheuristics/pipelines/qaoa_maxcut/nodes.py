import pandas as pd

from fromhopetoheuristics.utils.maxcut_utils import provide_random_maxcut_QUBO
from fromhopetoheuristics.utils.qaoa_utils import (
    compute_min_energy_solution,
    run_QAOA,
)

from typing import Dict

import logging

log = logging.getLogger(__name__)


def run_maxcut_qaoa(
    seed: int,
    max_p: int,
    q: int,
    maxcut_n_qubits: int = 4,
    maxcut_graph_density: float = 0.7,
    optimiser: str = "COBYLA",
) -> Dict[str, pd.DataFrame]:
    """
    Runs the QAOA algorithm on a randomly generated MaxCut problem.

    Parameters
    ----------
    seed: int
        The seed used to generate the random maxcut problem.
    max_p: int
        The number of layers of the QAOA circuit.
    q: int
        The number of parameters in the FOURIER strategy.
    maxcut_n_qubits: int, optional
        The number of qubits to use. Defaults to 4.
    maxcut_graph_density: float, optional
        The density of the graph to use. Defaults to 0.7.
    optimiser: str, optional
        The optimiser to use. Defaults to "COBYLA".

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary with a single key, "results", which contains a DataFrame
        with the results of the computation. The columns of the DataFrame are
        the different points in the annealing schedule, and the rows are the
        different possible states of the quantum computer.
    """
    results = []
    res_info = {}
    log.info(
        f"Running QAOA maxcut for n={maxcut_n_qubits} "
        f"with density={maxcut_graph_density}"
    )

    qubo = provide_random_maxcut_QUBO(maxcut_n_qubits, maxcut_graph_density, seed)

    res_info["min_energy"], res_info["opt_var_assignment"] = (
        compute_min_energy_solution(qubo)
    )

    res_data = run_QAOA(
        qubo,
        seed,
        max_p=max_p,
        q=q,
        optimiser=optimiser,
    )
    for res in res_data:
        res.update(res_info)
        results.append(res)

    results = pd.DataFrame.from_records(results)
    return {"results": results}
