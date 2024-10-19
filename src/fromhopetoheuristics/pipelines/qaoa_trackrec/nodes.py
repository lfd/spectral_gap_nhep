import os
from datetime import datetime
import numpy as np
from typing import Optional, List

from fromhopetoheuristics.utils.data_utils import save_to_csv
from fromhopetoheuristics.utils.qaoa_utils import (
    compute_min_energy_solution,
    solve_QUBO_with_QAOA,
)

import logging

log = logging.getLogger(__name__)


def track_reconstruction_qaoa(
    qubo: np.ndarray,
    seed: int,
    result_path_prefix: str,
    geometric_index: int = 0,
    include_header: bool = True,
    q=-1,
    max_p=20,
    optimiser="COBYLA",
):
    if include_header:
        header_content = [
            "problem",
            "optimiser",
            "num_qubits",
            "geometric_index",
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
            random_param_init=False,
            optimiser=optimiser,
        )
        if q == -1:
            init_params = np.concatenate([beta, gamma])
        else:
            init_params = np.concatenate([v, u])

        row_content = [
            "track reconstruction",
            optimiser,
            len(qubo),
            geometric_index,
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


def run_track_reconstruction_qaoa(
    qubos: List[Optional[np.ndarray]],
    event_path: str,
    seed: int,
    max_p: int,
    q: int,
    geometric_index: int = -1,
    optimiser="COBYLA",
):
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    result_path_prefix = os.path.join(
        os.path.dirname(event_path), "QAOA", time_stamp
    )
    first = True

    for i, qubo in enumerate(qubos):
        if geometric_index != -1:
            i = geometric_index
        if qubo is not None:
            log.info(f"Optimising QUBO {i+1}/{len(qubos)} (n={len(qubo)})")
            track_reconstruction_qaoa(
                qubo,
                seed,
                result_path_prefix,
                geometric_index=i,
                max_p=max_p,
                q=q,
                optimiser=optimiser,
                include_header=first,  # FIXME
            )
            first = False

    return {
        "qaoa_solution_path": os.path.join(result_path_prefix, "solution.csv")
    }  # FIXME
