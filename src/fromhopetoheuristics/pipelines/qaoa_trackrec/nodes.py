import numpy as np
from typing import Optional, List, Dict

from fromhopetoheuristics.utils.qaoa_utils import (
    compute_min_energy_solution,
    dict_QUBO_to_matrix,
    run_QAOA,
)
import pandas as pd

import logging

log = logging.getLogger(__name__)


def run_track_reconstruction_qaoa(
    qubos: List[Optional[np.ndarray]],  # List of QUBO matrices
    seed: int,  # Random seed
    max_p: int,  # Maximum number of layers in the QAOA circuit
    q: int,  # Number of parameters in the FOURIER strategy
    optimiser: str,  # Optimiser to use
    geometric_index: int,  # Index of the geometric QUBO
) -> Dict[str, pd.DataFrame]:  # Returns a dictionary with a single key "results"
    """
    Runs the QAOA algorithm on a given list of QUBO matrices.

    Parameters
    ----------
    qubos : List[Optional[np.ndarray]]
        List of QUBO matrices to be solved.
    seed : int
        Random seed for reproducibility.
    max_p : int
        Maximum number of layers in the QAOA circuit.
    q : int
        Number of parameters in the FOURIER strategy.
    optimiser : str, optional
        The optimiser to use. Defaults to "COBYLA".

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary with a single key "results", which contains a DataFrame
        with the results of the computation. The columns of the DataFrame are
        the different points in the annealing schedule, and the rows are the
        different possible states of the quantum computer.
    """
    results = []

    if geometric_index != -1:
        qubo = dict_QUBO_to_matrix(qubos[str(geometric_index)])
    else:
        qubo = dict_QUBO_to_matrix(qubos["0"])  # FIXME

    if qubo is None or qubo.size == 0:
        log.warning("Skipping qubo")
        return {"results": pd.DataFrame()}

    log.info(f"Optimising QUBO with n={len(qubo)}")
    res_info = {"num_qubits": len(qubo)}

    res_info["min_energy"], res_info["opt_var_assignment"] = (
        compute_min_energy_solution(qubo)
    )

    res_data = run_QAOA(
        qubo=qubo,
        seed=seed,
        q=q,
        max_p=max_p,
        optimiser=optimiser,
    )
    for res in res_data:
        res.update(res_info)
        results.append(res)

    results = pd.DataFrame.from_records(results)
    return {"results": results}
