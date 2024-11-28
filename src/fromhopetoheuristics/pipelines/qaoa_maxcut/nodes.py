import pandas as pd

from fromhopetoheuristics.utils.maxcut_utils import provide_random_maxcut_QUBO
from fromhopetoheuristics.utils.qaoa_utils import (
    compute_min_energy_solution,
    run_QAOA,
)

import numpy as np
from typing import Dict, Any, List
import logging

log = logging.getLogger(__name__)


def run_maxcut_qaoa(
    seed: int,
    max_p: int,
    q: int,
    num_random_perturbations: int,
    maxcut_n_qubits: int,
    maxcut_graph_density: float,
    optimiser: str,
    tolerance: float,
    maxiter: int,
    apply_bounds: bool,
    initialisation: str,
    options: Dict[str, Any],
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
    num_random_perturbations : int
        The number of random perturbations for the FOURIER strategy.
    maxcut_n_qubits: int
        The number of qubits to use.
    maxcut_graph_density: float
        The density of the graph to use.
    optimiser: str
        The optimiser to use.
    tolerance: float
        The tolerance for the optimization algorithm.
    maxiter: int
        The maximum number of iterations.
    apply_bounds : bool
        Whether parameter bounds should be applied during optimisation.
    initialisation: str
        Initialisation strategy for QAOA parameters
    options: Dict[str, Any]
        Additional options for the optimiser.

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary with a single key, "results", which contains a DataFrame
        with the results of the computation. The columns of the DataFrame are
        the different points in the annealing schedule, and the rows are the
        different possible states of the quantum computer.
    """
    results: List[Dict[str, Any]] = []
    res_info: Dict[str, Any] = {}
    log.info(
        f"Running QAOA maxcut for n={maxcut_n_qubits} "
        f"with density={maxcut_graph_density}"
    )

    qubo: np.ndarray = provide_random_maxcut_QUBO(
        maxcut_n_qubits, maxcut_graph_density, seed
    )

    res_info["min_energy"], res_info["opt_var_assignment"] = (
        compute_min_energy_solution(qubo)
    )

    res_data: List[Dict[str, Any]] = run_QAOA(
        qubo,
        seed,
        max_p=max_p,
        q=q,
        r=num_random_perturbations,
        optimiser=optimiser,
        tolerance=tolerance,
        maxiter=maxiter,
        apply_bounds=apply_bounds,
        initialisation=initialisation,
        options=options,
    )
    for res in res_data:
        res.update(res_info)
        results.append(res)

    results: pd.DataFrame = pd.DataFrame.from_records(results)
    return {"results": results}
