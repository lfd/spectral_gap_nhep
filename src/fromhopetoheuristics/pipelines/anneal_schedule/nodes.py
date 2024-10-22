import numpy as np
import pandas as pd

import logging

from fromhopetoheuristics.utils.qaoa_utils import (
    annealing_schedule_from_QAOA_params,
)

log = logging.getLogger(__name__)


def create_maxcut_anneal_schedule(
    results,
    maxcut_max_qubits: int = 10,
):
    # header_content = ["problem", "p", "q", "anneal_time", "anneal_fraction"]

    # header_content.extend(["n", "density"])
    # save_to_csv(header_content, result_path_prefix, "anneal_schedule.csv")

    result = {}
    for n in range(4, maxcut_max_qubits + 1):
        result[n] = {}
        for density in np.linspace(0.5, 1, num=6, endpoint=True):
            # betas, gammas = load_params_from_csv(
            #     qaoa_result_file,
            #     q=q,
            #     p=max_p,
            #     num_qubits=n,
            #     density=density,
            # )
            betas = results[f"{n}"][f"{density}"]["betas"]
            gammas = results[f"{n}"][f"{density}"]["gammas"]

            if betas is None or gammas is None:
                continue

            anneal_schedule = annealing_schedule_from_QAOA_params(betas, gammas)

            result[n][density] = anneal_schedule
            # for anneal_time, anneal_fraction in anneal_schedule:
            #     row_content = [
            #         "maxcut",
            #         max_p,
            #         q,
            #         anneal_time,
            #         anneal_fraction,
            #         n,
            #         density,
            #     ]
            #     save_to_csv(row_content, result_path_prefix, "anneal_schedule.csv")

    results = pd.DataFrame.from_dict(result)
    return {"results": results}


def create_trackrecon_anneal_schedule(
    results,
    num_angle_parts: int = 64,
):
    # header_content.append("geometric_index")
    # save_to_csv(header_content, result_path_prefix, "anneal_schedule.csv")

    result = {}
    for i in range(num_angle_parts):
        # betas, gammas = load_params_from_csv(
        #     qaoa_result_file, q=q, p=max_p, geometric_index=i
        # )
        betas = results[f"{i}"]["betas"]
        gammas = results[f"{i}"]["gammas"]

        if betas is None or gammas is None:
            continue

        anneal_schedule = annealing_schedule_from_QAOA_params(betas, gammas)

        results[i] = anneal_schedule

        # for anneal_time, anneal_fraction in anneal_schedule:

        #     row_content = [
        #         "track reconstruction",
        #         max_p,
        #         q,
        #         anneal_time,
        #         anneal_fraction,
        #         i,
        #     ]
        #     save_to_csv(row_content, result_path_prefix, "anneal_schedule.csv")

    results = pd.DataFrame.from_dict(results)
    return {"results": results}
