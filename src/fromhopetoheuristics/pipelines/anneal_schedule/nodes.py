import os
import numpy as np

from fromhopetoheuristics.utils.data_utils import (
    load_params_from_csv,
    save_to_csv,
)

import logging

from fromhopetoheuristics.utils.qaoa_utils import (
    annealing_schedule_from_QAOA_params,
)

log = logging.getLogger(__name__)


def create_anneal_schedule(
    qaoa_result_file: str,
    q: int,
    max_p: int,
    num_angle_parts: int = 64,
    maxcut_max_qubits: int = 10,
):
    result_path_prefix = os.path.dirname(qaoa_result_file)
    header_content = ["problem", "p", "q", "anneal_time", "anneal_fraction"]

    if qaoa_result_file == "":
        log.warning(
            "Empty QAOA result path provided. "
            "Forgot to execute a QAOA pipeline first?"
        )
        return {}
    if "MAXCUT" in qaoa_result_file:
        header_content.extend(["n", "density"])
        save_to_csv(header_content, result_path_prefix, "anneal_schedule.csv")
        for n in range(4, maxcut_max_qubits + 1):
            for density in np.linspace(0.5, 1, num=6, endpoint=True):
                betas, gammas = load_params_from_csv(
                    qaoa_result_file,
                    q=q,
                    p=max_p,
                    num_qubits=n,
                    density=density,
                )
                if betas is None or gammas is None:
                    continue

                anneal_schedule = annealing_schedule_from_QAOA_params(
                    betas, gammas
                )

                for anneal_time, anneal_fraction in anneal_schedule:
                    row_content = [
                        "maxcut",
                        max_p,
                        q,
                        anneal_time,
                        anneal_fraction,
                        n,
                        density,
                    ]
                    save_to_csv(
                        row_content, result_path_prefix, "anneal_schedule.csv"
                    )
    else:  # track reconstruction
        header_content.append("geometric_index")
        save_to_csv(header_content, result_path_prefix, "anneal_schedule.csv")
        for i in range(num_angle_parts):
            betas, gammas = load_params_from_csv(
                qaoa_result_file, q=q, p=max_p, geometric_index=i
            )
            if betas is None or gammas is None:
                continue

            anneal_schedule = annealing_schedule_from_QAOA_params(betas, gammas)

            for anneal_time, anneal_fraction in anneal_schedule:
                row_content = [
                    "track reconstruction",
                    max_p,
                    q,
                    anneal_time,
                    anneal_fraction,
                    i,
                ]
                save_to_csv(
                    row_content, result_path_prefix, "anneal_schedule.csv"
                )

    return {}
