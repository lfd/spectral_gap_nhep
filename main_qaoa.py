from typing import Iterable
import numpy as np
from datetime import datetime
import sys

from hepqpr.qallse.data_wrapper import DataWrapper

from data_utils import create_dataset_track_reconstruction, save_to_csv
from maxcut_utils import provide_random_maxcut_QUBO
from track_reconstruction_utils import provide_track_reconstruction_QUBO
from qaoa_utils import compute_min_energy_solution, solve_QUBO_with_QAOA
import logging

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


def track_reconstruction_QAOA(
    dw: DataWrapper,
    seed: int,
    result_path_prefix: str,
    data_path: str,
    geometric_index: int = 0,
    include_header: bool = True,
    max_p=20,
):
    if include_header:
        header_content = [
            "problem",
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

    qubo = provide_track_reconstruction_QUBO(dw, data_path, geometric_index)

    if qubo is None:
        return

    min_energy, opt_var_assignment = compute_min_energy_solution(qubo)

    for q in range(-1, 6):
        if q == 0:
            continue
        init_params = None
        for p in range(1, 6):
            if q > p:
                this_q = p
            else:
                this_q = q
            print(f"q = {q}, p = {p}")
            qaoa_energy, beta, gamma, u, v = solve_QUBO_with_QAOA(
                qubo,
                p,
                this_q,
                seed=seed,
                initial_params=init_params,
                random_param_init=True,
            )
            if q == -1:
                init_params = np.concatenate([beta, gamma])
            else:
                init_params = np.concatenate([v, u])

            row_content = [
                "track reconstruction",
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


def maxcut(
    num_qubits: int,
    density: float,
    seed: int,
    result_path_prefix: str,
    include_header: bool = True,
    max_p=20,
):
    if include_header:
        header_content = [
            "problem",
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

    for q in range(-1, 6):
        if q == 0:
            continue
        init_params = None
        for p in range(1, 6):
            if q > p:
                this_q = p
            else:
                this_q = q
            print(f"q = {q}, p = {p}")
            qaoa_energy, beta, gamma, u, v = solve_QUBO_with_QAOA(
                qubo,
                p,
                this_q,
                seed=seed,
                initial_params=init_params,
                random_param_init=True,
            )
            if q == -1:
                init_params = np.concatenate([beta, gamma])
            else:
                init_params = np.concatenate([v, u])

            row_content = [
                "maxcut",
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


if __name__ == "__main__":

    problem = ""
    if len(sys.argv) > 2:
        exit(
            "Usage: python main_qaoa.py [m|t] (m=maxcut, t=track reconstruction)"
        )
    elif len(sys.argv) == 1:
        problem = "m"
    elif sys.argv[1] not in ["m", "t"]:
        exit(
            f"Invalid option {sys.argv[1]} \n"
            "Usage: python main_qaoa.py [m|t] (m=maxcut, t=track reconstruction)"
        )

    first = True
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if problem == "m":
        seed = 777
        result_path_prefix = f"results/MAXCUT_QAOA/{time_stamp}"
        n = 5
        density = 0.7
        maxcut(
            n,
            density,
            seed,
            result_path_prefix,
            include_header=first,
        )
        first = False
    else:
        seed = 12345
        result_path_prefix = f"results/TR_QAOA/{time_stamp}"

        dw, data_path = create_dataset_track_reconstruction(
            result_path_prefix, seed
        )

        i = 1
        track_reconstruction_QAOA(
            dw,
            seed,
            result_path_prefix,
            data_path,
            geometric_index=i,
            include_header=first,
        )
        first = False
