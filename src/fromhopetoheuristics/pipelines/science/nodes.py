import pickle
import os
from datetime import datetime
import numpy as np
from typing import Iterable, Optional, List

from qallse.cli.func import (
    build_model,
    solve_neal,
    print_stats,
)
from qallse.data_wrapper import DataWrapper
from fromhopetoheuristics.utils.model import QallseSplit
from fromhopetoheuristics.utils.data_utils import save_to_csv, store_qubo
from fromhopetoheuristics.utils.maxcut_utils import provide_random_maxcut_QUBO
from fromhopetoheuristics.utils.spectral_gap_calculator import (
    calculate_spectral_gap,
)
from fromhopetoheuristics.utils.qaoa_utils import (
    compute_min_energy_solution,
    solve_QUBO_with_QAOA,
    dict_QUBO_to_matrix,
)

import logging

log = logging.getLogger(__name__)


def build_qubos(
    data_wrapper: DataWrapper,
    event_path: str,
    num_angle_parts: int,
):
    """
    Creates partial QUBO from TrackML data using Qallse. The data is split into
    several parts by the angle in the XY-plane of the detector, from which the
    QUBO is built.

    :param data_wrapper: Qallse data wrapper
    :type data_wrapper: DataWrapper
    :param event_path: path, from which to load the TrackML data
    :type event_path: str
    :param num_angle_parts: Number of angle segments in the detector, equals
        the number of resulting QUBOs
    :type num_angle_parts: int
    :return: the path to the QUBO in dict form
    :rtype: List[str]
    """
    qubo_paths = []
    qubo_base_path = os.path.join(os.path.dirname(event_path), "QUBO")
    for i in range(num_angle_parts):
        qubo_prefix = f"angle_index{i:02d}_"
        qubo_path = os.path.join(qubo_base_path, f"{qubo_prefix}qubo.pickle")
        if not os.path.exists(qubo_path):
            extra_config = {
                "geometric_index": i,
                "xy_angle_parts ": num_angle_parts,
            }
            model = QallseSplit(data_wrapper, **extra_config)
            build_model(event_path, model, False)

            qubo_path = store_qubo(event_path, model, qubo_prefix=qubo_prefix)
        qubo_paths.append(qubo_path)

    return {"qubo_paths": qubo_paths}


def load_qubos(qubo_paths: List[str]):
    qubos = []
    for qubo_path in qubo_paths:
        with open(qubo_path, "rb") as f:
            Q = pickle.load(f)

        qubo = dict_QUBO_to_matrix(Q)

        if len(qubo) > 18:
            log.warning(f"Too many variables for qubo {qubo_path}")
            qubo = None
        elif len(qubo) == 0:
            log.warning(f"Empty QUBO {qubo_path}")
            qubo = None
        qubos.append(qubo)

    return {"qubos": qubos}


def solve_qubos(
    data_wrapper: DataWrapper,
    qubos: List[Optional[np.ndarray]],
    result_path_prefix: str,
    seed: int,
):
    responses = []
    for i, qubo in enumerate(qubos):
        if qubo is None:
            responses.append(None)
            continue
        prefix = f"cl_solver{i:02d}"

        response = solve_neal(qubo, seed=seed)
        print_stats(data_wrapper, response, qubo)
        oname = os.path.join(
            result_path_prefix, prefix + "neal_response.pickle"
        )
        with open(oname, "wb") as f:
            pickle.dump(response, f)
        print(f"Wrote response to {oname}")
        responses.append(response)

    return {"responses": responses}


def track_reconstruction_qaoa(
    qubo: Optional[np.ndarray],
    seed: int,
    result_path_prefix: str,
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

    min_energy, opt_var_assignment = compute_min_energy_solution(qubo)

    for q in range(-1, max_p):
        if q == 0:
            continue
        init_params = None
        for p in range(1, max_p):
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


def maxcut_qaoa(
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


def maxcut_annealing(
    num_qubits: int,
    density: float,
    seed: int,
    fractions: Iterable[float],
    result_path_prefix: str,
    include_header: bool = True,
):
    csv_data_list = []
    if include_header:
        csv_data_list.append(
            [
                "problem",
                "num_qubits",
                "density",
                "seed",
                "fraction",
                "gs",
                "fes",
                "gap",
            ]
        )

    qubo = provide_random_maxcut_QUBO(num_qubits, density, seed)

    for fraction in fractions:
        gs_energy, fes_energy, gap = calculate_spectral_gap(fraction, qubo)
        csv_data_list.append(
            [
                "maxcut",
                num_qubits,
                density,
                seed,
                np.round(fraction, 2),
                gs_energy,
                fes_energy,
                gap,
            ]
        )

    for csv_data in csv_data_list:
        save_to_csv(
            csv_data,
            result_path_prefix,
            "spectral_gap_evolution.csv",
        )


def track_reconstruction_annealing(
    qubo: Optional[np.ndarray],
    seed: int,
    fractions: Iterable[float],
    result_path_prefix: str,
    geometric_index: int = 0,
    include_header: bool = True,
):
    csv_data_list = []
    if include_header:
        csv_data_list.append(
            [
                "problem",
                "num_qubits",
                "geometric_index",
                "seed",
                "fraction",
                "gs",
                "fes",
                "gap",
            ]
        )

    for fraction in fractions:
        gs_energy, fes_energy, gap = calculate_spectral_gap(
            fraction,
            qubo,
        )
        csv_data_list.append(
            [
                "track reconstruction",
                len(qubo),
                geometric_index,
                seed,
                np.round(fraction, 2),
                gs_energy,
                fes_energy,
                gap,
            ]
        )

    for csv_data in csv_data_list:
        save_to_csv(csv_data, result_path_prefix, "spectral_gap_evolution.csv")


def run_maxcut_qaoa(seed):
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    result_path_prefix = f"results/MAXCUT_QAOA/{time_stamp}"
    n = 5
    density = 0.7
    maxcut_qaoa(
        n,
        density,
        seed,
        result_path_prefix,
        include_header=True,  # FIXME
    )

    return {}  # FIXME


def run_track_reconstruction_qaoa(
    qubos: List[Optional[np.ndarray]], event_path: str, seed: int, max_p: int
):
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    result_path_prefix = os.path.join(os.path.dirname(event_path), "QAOA", time_stamp)
    first = True

    for i, qubo in enumerate(qubos):
        if qubo is not None:
            track_reconstruction_qaoa(
                qubo,
                seed,
                result_path_prefix,
                geometric_index=i,
                max_p=max_p,
                include_header=first,  # FIXME
            )
            first = False
            break  # for now, just do this for the first QUBO

    return {}  # FIXME


def run_maxcut_annealing(seed):
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    fractions = np.linspace(0, 1, num=11, endpoint=True)
    result_path_prefix = f"results/MAXCUT/{time_stamp}"
    first = True
    for n in range(4, 11):
        for density in np.linspace(0.5, 1, num=6, endpoint=True):
            maxcut_annealing(
                n,
                density,
                seed,
                fractions,
                result_path_prefix,
                include_header=first,  # FIXME
            )
            first = False

    return {}  # FIXME


def run_track_reconstruction_annealing(
    qubos: List[Optional[np.ndarray]], event_path: str, seed: int
):
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    fractions = np.linspace(0, 1, num=11, endpoint=True)
    result_path_prefix = os.path.join(
        os.path.dirname(event_path), "spectral_gap", time_stamp,
    )
    first = True

    for i, qubo in enumerate(qubos):
        if qubo is not None:
            track_reconstruction_annealing(
                qubo,
                seed,
                fractions,
                result_path_prefix,
                geometric_index=i,
                include_header=first,  # FIXME
            )
            first = False

    return {}  # FIXME
