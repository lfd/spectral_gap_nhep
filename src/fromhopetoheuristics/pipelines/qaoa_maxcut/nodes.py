from datetime import datetime
import numpy as np
import os

from fromhopetoheuristics.utils.maxcut_utils import provide_random_maxcut_QUBO
from fromhopetoheuristics.utils.data_utils import save_to_csv
from fromhopetoheuristics.utils.qaoa_utils import (
    compute_min_energy_solution,
    solve_QUBO_with_QAOA,
)

import logging

log = logging.getLogger(__name__)


def maxcut_qaoa(
    qubo,
    seed: int,
    max_p=20,
    q=-1,
    optimiser="COBYLA",
):
    qaoa_energies = []
    betas = []
    gammas = []
    us = []
    vs = []

    if q == 0:
        return
    init_params = None
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
            random_param_init=True,
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


def run_maxcut_qaoa(
    seed: int,
    max_p: int,
    q: int,
    maxcut_max_qubits: int,
    optimiser="COBYLA",
):
    results = {}
    for n_qubits in range(4, maxcut_max_qubits + 1):
        log.info(f"Optimising QUBO with n={n_qubits} of {maxcut_max_qubits}")
        results[n_qubits] = {}

        for density in np.linspace(
            0.5, 1, num=6, endpoint=True
        ):  # FIXME: propagate params
            log.info(f"\twith density={density}")

            qubo = provide_random_maxcut_QUBO(n_qubits, density, seed)

            min_energy, opt_var_assignment = compute_min_energy_solution(qubo)

            results[n_qubits][density] = {
                **maxcut_qaoa(
                    qubo,
                    seed,
                    max_p=max_p,
                    q=q,
                    optimiser=optimiser,
                ),
                "min_energy": min_energy,
                "opt_var_assignment": opt_var_assignment,
            }

    return {"results": results}
