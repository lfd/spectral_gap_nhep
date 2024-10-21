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
    num_qubits: int,
    density: float,
    seed: int,
    result_path_prefix: str,
    include_header: bool = True,
    max_p=20,
    q=-1,
    optimiser="COBYLA",
):
    if include_header:
        header_content = [
            "problem",
            "optimiser",
            "num_qubits",
            "density",
            "seed",
            "energy",
            "optimal_energy",
            "optimal_solution",
            "p",
            "q",
        ]

        for s in ["beta", "gamma", "u", "v"]:
            header_content.extend([f"{s}{i+1:02d}" for i in range(max_p)])

        save_to_csv(header_content, result_path_prefix, "solution.csv")

    qubo = provide_random_maxcut_QUBO(num_qubits, density, seed)

    min_energy, opt_var_assignment = compute_min_energy_solution(qubo)

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

        row_content = [
            "maxcut",
            optimiser,
            len(qubo),
            density,
            seed,
            qaoa_energy,
            min_energy,
            opt_var_assignment,
            p,
            q,
        ]

        for params in [beta, gamma, u, v]:
            row_content.extend(params)
            row_content.extend(
                [None for _ in range(max_p - len(params))]
            )  # padding

        save_to_csv(row_content, result_path_prefix, "solution.csv")


def run_maxcut_qaoa(
    result_path_prefix: str,
    seed: int,
    max_p: int,
    q: int,
    maxcut_max_qubits: int,
    optimiser="COBYLA",
):
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    result_path_prefix = os.path.join(
        result_path_prefix, "MAXCUT/QAOA", time_stamp
    )

    first = True
    for n in range(4, maxcut_max_qubits + 1):
        log.info(f"Optimising QUBO with n={n} of {maxcut_max_qubits}")
        for density in np.linspace(0.5, 1, num=6, endpoint=True):
            log.info(f"\twith density={density}")
            maxcut_qaoa(
                n,
                density,
                seed,
                result_path_prefix,
                max_p=max_p,
                q=q,
                optimiser=optimiser,
                include_header=first,  # FIXME
            )
            first = False

    return {
        "qaoa_solution_path": os.path.join(result_path_prefix, "solution.csv")
    }  # FIXME
