import os
from datetime import datetime
import numpy as np
from typing import Iterable, Optional, Dict

from fromhopetoheuristics.utils.spectral_gap_calculator import (
    calculate_spectral_gap,
)
from fromhopetoheuristics.utils.data_utils import save_to_csv
import logging

log = logging.getLogger(__name__)


def track_reconstruction_annealing(
    qubo: np.ndarray,
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

    # for csv_data in csv_data_list:
    # save_to_csv(csv_data, result_path_prefix, "spectral_gap_evolution.csv")
    return {"results": csv_data_list}


def run_track_reconstruction_annealing(
    qubos: Dict,
    event_path: str,
    seed: int,
    num_anneal_fractions: int,
):
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    fractions = np.linspace(0, 1, num=num_anneal_fractions, endpoint=True)
    result_path_prefix = os.path.join(
        os.path.dirname(event_path),
        "spectral_gap",
        time_stamp,
    )
    first = True

    results = {}
    for i, qubo in qubos.items():
        log.info(f"Computing spectral gaps for QUBO {i+1}/{len(qubos)}")
        results[i] = track_reconstruction_annealing(
            qubo,
            seed,
            fractions,
            result_path_prefix,
            geometric_index=i,
            include_header=first,  # FIXME
        )
        first = False

    return {"results": results}
