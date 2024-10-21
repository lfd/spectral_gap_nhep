import numpy as np
from typing import Iterable, Dict

from fromhopetoheuristics.utils.spectral_gap_calculator import (
    calculate_spectral_gap,
)
import pandas as pd
import logging

log = logging.getLogger(__name__)


def track_reconstruction_annealing(
    qubo: np.ndarray,
    fractions: Iterable[float],
):
    gs_energies = []
    fes_energies = []
    gaps = []

    for fraction in fractions:
        gs_energy, fes_energy, gap = calculate_spectral_gap(
            fraction,
            qubo,
        )
        gs_energies.append(gs_energy)
        fes_energies.append(fes_energy)
        gaps.append(gap)

    return {"gs_energies": gs_energies, "fes_energies": fes_energies, "gaps": gaps}


def run_track_reconstruction_annealing(
    qubos: Dict,
    num_anneal_fractions: int,
):
    fractions = np.linspace(0, 1, num=num_anneal_fractions, endpoint=True)

    results = {}
    for i, qubo in qubos.items():
        if qubo.size == 0:
            log.warning(f"Skipping qubo {i+1}/{len(qubos)}")
            continue
        log.info(f"Computing spectral gaps for QUBO {i+1}/{len(qubos)}")
        results[i] = track_reconstruction_annealing(
            qubo,
            fractions,
        )

    results = pd.DataFrame.from_dict(results)
    return {"results": results}
