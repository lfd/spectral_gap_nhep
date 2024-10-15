from typing import Iterable
import numpy as np
import sys
from datetime import datetime

from qallse.data_wrapper import DataWrapper

from fromhopetoheuristics.utils.spectral_gap_calculator import calculate_spectral_gap
from fromhopetoheuristics.utils.maxcut_utils import provide_random_maxcut_QUBO
from fromhopetoheuristics.utils.track_reconstruction_utils import provide_track_reconstruction_QUBO
from fromhopetoheuristics.utils.data_utils import create_dataset_track_reconstruction, save_to_csv
import logging

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


def track_reconstruction(
    dw: DataWrapper,
    seed: int,
    fractions: Iterable[float],
    result_path_prefix: str,
    data_path: str,
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
    qubo = provide_track_reconstruction_QUBO(dw, data_path, geometric_index)

    if qubo is None:
        return

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


def maxcut(
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


if __name__ == "__main__":
    problem = ""
    if len(sys.argv) > 2:
        exit("Usage: python main.py [m|t] (m=maxcut, t=track reconstruction)")
    elif len(sys.argv) == 1:
        problem = "m"
    elif sys.argv[1] not in ["m", "t"]:
        exit(
            f"Invalid option {sys.argv[1]} \n"
            "Usage: python main.py [m|t] (m=maxcut, t=track reconstruction)"
        )

    first = True
    fractions = np.linspace(0, 1, num=11, endpoint=True)
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if problem == "m":
        seed = 777
        result_path_prefix = f"results/MAXCUT/{time_stamp}"
        for n in range(4, 11):
            for density in np.linspace(0.5, 1, num=6, endpoint=True):
                maxcut(
                    n,
                    density,
                    seed,
                    fractions,
                    result_path_prefix,
                    include_header=first,
                )
                first = False
    else:
        seed = 12345
        result_path_prefix = f"results/TR/{time_stamp}"

        dw, data_path = create_dataset_track_reconstruction(
            result_path_prefix, seed
        )

        for i in range(64):
            track_reconstruction(
                dw,
                seed,
                fractions,
                result_path_prefix,
                data_path,
                geometric_index = i,
                include_header=first,
            )
            first = False
