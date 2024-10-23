import pandas as pd

from fromhopetoheuristics.utils.maxcut_utils import provide_random_maxcut_QUBO
from fromhopetoheuristics.utils.qaoa_utils import (
    compute_min_energy_solution,
    run_QAOA,
)

import logging

log = logging.getLogger(__name__)


def run_maxcut_qaoa(
    seed: int,
    max_p: int,
    q: int,
    maxcut_n_qubits: int = 4,
    maxcut_graph_density: float = 0.7,
    optimiser="COBYLA",
):
    results = []
    res_info = {}
    log.info(
        f"Running QAOA maxcut for n={maxcut_n_qubits} "
        f"with density={maxcut_graph_density}"
    )

    qubo = provide_random_maxcut_QUBO(maxcut_n_qubits, maxcut_graph_density, seed)

    res_info["min_energy"], res_info["opt_var_assignment"] = (
        compute_min_energy_solution(qubo)
    )

    res_data = run_QAOA(
        qubo,
        seed,
        max_p=max_p,
        q=q,
        optimiser=optimiser,
    )
    for res in res_data:
        res.update(res_info)
        results.append(res)

    results = pd.DataFrame.from_records(results)
    return {"results": results}
