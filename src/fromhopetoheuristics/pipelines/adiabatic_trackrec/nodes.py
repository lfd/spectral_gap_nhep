import numpy as np
from fromhopetoheuristics.utils.qaoa_utils import dict_QUBO_to_matrix

from fromhopetoheuristics.utils.spectral_gap_calculator import annealing
import pandas as pd

from typing import List, Dict, Any

import logging

log = logging.getLogger(__name__)


def run_track_reconstruction_annealing(
    qubos: List[Dict[str, Any]],
    num_anneal_fractions: int,
    geometric_index: int,
) -> Dict[str, pd.DataFrame]:
    """
    Runs the track reconstruction algorithm using the adiabatic quantum computer.

    Parameters
    ----------
    qubos : List[Dict[str, Any]]
        The QUBO matrix to be solved
    num_anneal_fractions : int
        The number of points in the annealing schedule to use

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary with a single key, "results", which contains a DataFrame
        with the results of the computation. The columns of the DataFrame are
        the different points in the annealing schedule, and the rows are the
        different possible states of the quantum computer.
    """
    if geometric_index != -1:
        qubo = dict_QUBO_to_matrix(qubos[str(geometric_index)])
    else:
        qubo = dict_QUBO_to_matrix(qubos["0"])

    fractions = np.linspace(0, 1, num=num_anneal_fractions, endpoint=True)

    results = []
    if qubo.size == 0:
        log.warning("Skipping QUBO")
        return {"results": pd.DataFrame()}
    log.info("Computing spectral gaps")
    res_info = {"num_qubits": len(qubo)}
    res_data = annealing(
        qubo=qubo,
        fractions=fractions,
    )
    for res in res_data:
        res.update(res_info)
        results.append(res)

    results = pd.DataFrame.from_records(results)
    return {"results": results}
