from typing import Optional, List, Dict, Any

from fromhopetoheuristics.utils.qaoa_utils import (
    compute_min_energy_solution,
    dict_QUBO_to_matrix,
    run_QAOA,
)
import pandas as pd

import logging

log = logging.getLogger(__name__)


def run_track_reconstruction_qaoa(
    qubos: List[Optional[Dict[str, float]]],
    seed: int,
    max_p: int,
    q: int,
    num_random_perturbations: int,
    optimiser: str,
    tolerance: float,
    maxiter: int,
    geometric_index: int,
    apply_bounds: bool,
    initialisation: str,
    options: Dict[str, Any],
    hyperhyper_trial_id: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Runs the QAOA algorithm on a given list of QUBO matrices.

    Parameters
    ----------
    qubos : List[Optional[Dict[str, float]]]
        List of QUBO matrices to be solved.
    seed : int
        Random seed for reproducibility.
    max_p : int
        Maximum number of layers in the QAOA circuit.
    q : int
        Number of parameters in the FOURIER strategy.
    num_random_perturbations : int
        The number of random perturbations for the FOURIER strategy.
    optimiser : str
        The optimiser to use.
    tolerance : float
        The tolerance for the optimization algorithm.
    maxiter : int
        The maximum number of iterations.
    geometric_index : int
        The index of the geometric QUBO to use.
    apply_bounds : bool
        Whether parameter bounds should be applied during optimisation.
    initialisation: str
        Initialisation strategy for QAOA parameters
    options : Dict[str, Any]
        Additional options for the optimiser.
    hyperhyper_trial_id : Optional[str]
        Temporary output file for hyperparameter optimisation

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
        n_pert=num_random_perturbations,
        max_p=max_p,
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

    results = pd.DataFrame.from_records(results)

    if hyperhyper_trial_id is not None:
        tmp_file_name = f".hyperhyper{hyperhyper_trial_id}.json"
        results.to_json(tmp_file_name)

    return {"results": results}
