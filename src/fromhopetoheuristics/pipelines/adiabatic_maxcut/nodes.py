from datetime import datetime
import numpy as np
from typing import Iterable
import os

from fromhopetoheuristics.utils.maxcut_utils import provide_random_maxcut_QUBO
from fromhopetoheuristics.utils.data_utils import save_to_csv
from fromhopetoheuristics.utils.spectral_gap_calculator import (
    calculate_spectral_gap,
)
import logging

log = logging.getLogger(__name__)


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


def run_maxcut_annealing(
    result_path_prefix: str,
    seed: int,
    num_anneal_fractions: int,
    maxcut_max_qubits: int,
):
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    fractions = np.linspace(0, 1, num=num_anneal_fractions, endpoint=True)
    result_path_prefix = os.path.join(
        result_path_prefix, "MAXCUT/spectral_gap", time_stamp
    )
    first = True
    for n in range(4, maxcut_max_qubits + 1):
        log.info(f"Computing spectral gaps for QUBO with n={n} of {maxcut_max_qubits}")
        for density in np.linspace(0.5, 1, num=6, endpoint=True):
            log.info(f"\twith density={density}")
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
