import numpy as np
from typing import Optional, List

from fromhopetoheuristics.utils.qaoa_utils import (
    compute_min_energy_solution,
    run_QAOA,
)
import pandas as pd

import logging

log = logging.getLogger(__name__)


def run_track_reconstruction_qaoa(
    qubos: List[Optional[np.ndarray]],
    seed: int,
    max_p: int,
    q: int,
    optimiser="COBYLA",
):
    results = []
    qubo = qubos[0] # FIXME
    if qubo is None or qubo.size == 0:
        log.warning(f"Skipping qubo")
        return {}

    log.info(f"Optimising QUBO with n={len(qubo)}")
    res_info = {"num_qubits": len(qubo)}

    res_info["min_energy"], res_info["opt_var_assignment"] = (
        compute_min_energy_solution(qubo)
    )

    res_data = run_QAOA(
            qubo=qubo,
            seed=seed,
            q=q,
            max_p=max_p,
            optimiser=optimiser,
    )
    for res in res_data:
        res.update(res_info)
        results.append(res)

    results = pd.DataFrame.from_records(results)
    return {"results": results}
