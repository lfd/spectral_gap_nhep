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
):
    qubo = provide_random_maxcut_QUBO(num_qubits, density, seed)

    gs_energies = []
    fes_energies = []
    gaps = []

    for fraction in fractions:
        gs_energy, fes_energy, gap = calculate_spectral_gap(fraction, qubo)
        gs_energies.append(gs_energy)
        fes_energies.append(fes_energy)
        gaps.append(gap)

    return {"gs_energies": gs_energies, "fes_energies": fes_energies, "gaps": gaps}


def run_maxcut_annealing(
    seed: int,
    num_anneal_fractions: int,
    maxcut_max_qubits: int,
):
    fractions = np.linspace(0, 1, num=num_anneal_fractions, endpoint=True)

    results = {}
    for n_qubits in range(4, maxcut_max_qubits + 1):
        log.info(
            f"Computing spectral gaps for QUBO with n={n_qubits} of "
            "{maxcut_max_qubits}"
        )
        results[n_qubits] = {}

        for density in np.linspace(
            0.5, 1, num=6, endpoint=True
        ):  # FIXME: propagate params
            log.info(f"\twith density={density}")

            results[n_qubits][density] = (
                maxcut_annealing(
                    n_qubits,
                    density,
                    seed,
                    fractions,
                ),
            )

    return {"results": results}
