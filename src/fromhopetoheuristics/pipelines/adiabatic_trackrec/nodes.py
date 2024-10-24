import numpy as np
from fromhopetoheuristics.utils.qaoa_utils import dict_QUBO_to_matrix

from fromhopetoheuristics.utils.spectral_gap_calculator import annealing
import pandas as pd
import logging

log = logging.getLogger(__name__)


def run_track_reconstruction_annealing(
    qubos,
    num_anneal_fractions: int,
):
    qubo = dict_QUBO_to_matrix(qubos[0])  # FIXME
    fractions = np.linspace(0, 1, num=num_anneal_fractions, endpoint=True)

    results = []
    if qubo.size == 0:
        log.warning("Skipping QUBO")
        return {"results": pd.DataFrame()}
    log.info("Computing spectral gaps")
    res_info = dict()
    res_data = annealing(
        qubo=qubo,
        fractions=fractions,
    )
    for res in res_data:
        res.update(res_info)
        results.append(res)

    results = pd.DataFrame.from_records(results)
    return {"results": results}
