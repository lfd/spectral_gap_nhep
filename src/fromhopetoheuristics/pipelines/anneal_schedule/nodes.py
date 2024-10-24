import numpy as np
import pandas as pd

import logging

from fromhopetoheuristics.utils.qaoa_utils import (
    annealing_schedule_from_QAOA_params,
)

log = logging.getLogger(__name__)


def create_anneal_schedule(
    results,
):
    beta_colnames = [b for b in results.columns if "beta" in b]
    beta_colnames.sort()
    gamma_colnames = [g for g in results.columns if "gamma" in g]
    gamma_colnames.sort()

    last_results = results.query(f"p == {len(beta_colnames)}")

    betas = np.array([last_results[b] for b in beta_colnames]).flatten()
    gammas = np.array([last_results[g] for g in gamma_colnames]).flatten()

    anneal_schedule = annealing_schedule_from_QAOA_params(betas, gammas)

    results = pd.DataFrame.from_records(
        anneal_schedule, columns=["anneal_time", "anneal_fraction"]
    )
    return {"results": results}
