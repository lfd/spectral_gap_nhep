import numpy as np
from typing import Optional, List

from fromhopetoheuristics.utils.qaoa_utils import (
    compute_min_energy_solution,
    solve_QUBO_with_QAOA,
)
import pandas as pd

import logging

log = logging.getLogger(__name__)


def track_reconstruction_qaoa(
    qubo: np.ndarray,
    seed: int,
    q=-1,
    max_p=20,
    optimiser="COBYLA",
):
    if q == 0:
        return
    init_params = None

    qaoa_energies = []
    betas = []
    gammas = []
    us = []
    vs = []

    for p in range(1, max_p + 1):
        if q > p:
            this_q = p
        else:
            this_q = q
        log.info(f"Running QAOA for q = {q}, p = {p}/{max_p}")
        qaoa_energy, beta, gamma, u, v = solve_QUBO_with_QAOA(
            qubo,
            p,
            this_q,
            seed=seed,
            initial_params=init_params,
            random_param_init=False,
            optimiser=optimiser,
        )
        if q == -1:
            init_params = np.concatenate([beta, gamma])
        else:
            init_params = np.concatenate([v, u])

        qaoa_energies.append(qaoa_energy)
        betas.append(beta)
        gammas.append(gamma)
        us.append(u)
        vs.append(v)

    return {
        "qaoa_energies": qaoa_energies,
        "betas": betas,
        "gammas": gammas,
        "us": us,
        "vs": vs,
    }


def run_track_reconstruction_qaoa(
    qubos: List[Optional[np.ndarray]],
    seed: int,
    max_p: int,
    q: int,
    optimiser="COBYLA",
):
    results = {}
    for i, qubo in qubos.items():
        if qubo.size == 0:
            log.warning(f"Skipping qubo {i+1}/{len(qubos)}")
            continue

        log.info(f"Optimising QUBO {i+1}/{len(qubos)} (n={len(qubo)})")
        min_energy, opt_var_assignment = compute_min_energy_solution(qubo)

        results[i] = {
            **track_reconstruction_qaoa(
                qubo=qubo,
                seed=seed,
                q=q,
                max_p=max_p,
                optimiser=optimiser,
            ),
            "min_energy": min_energy,
            "opt_var_assignment": opt_var_assignment,
        }

    results = pd.DataFrame.from_dict(results)
    return {"results": results}
