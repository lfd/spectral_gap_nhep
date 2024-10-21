from trackml.dataset import load_event, load_dataset
import pandas as pd
from typing import Tuple
import os
import re

from qallse.data_wrapper import DataWrapper
from qallse.dsmaker import create_dataset

import logging

log = logging.getLogger(__name__)

# TODO: currently we're loading data using the the external trackml-library
# However, we should switch to the kedro internal dataloader and catalog


def load_event_data(event) -> dict:
    hits, cells, particles, truth = load_event(f"dataset/{event}")

    return {
        "hits": hits,
        "cells": cells,
        "particles": particles,
        "truth": truth,
    }


def load_dataset_data() -> dict:
    dataset = pd.DataFrame()

    for event_id, hits, cells, particles, truth in load_dataset("dataset"):

        dataset = dataset.append(
            {
                "event_id": event_id,
                "hits": hits,
                "cells": cells,
                "particles": particles,
                "truth": truth,
            },
            ignore_index=True,
        )

    return {"dataset": dataset}


def create_metadata(
    seed: int,
    num_angle_parts: str,
    f: float,
    event_cells,
    event_hits,
    event_particles,
    event_truth,
) -> Tuple[dict, str]:
    """
    Creates/filters and stores TrackML event data for a given fraction of data.
    Uses Qallse to filter the data.

    :param result_path_prefix: Where to store the data
    :type result_path_prefix: str
    :param seed: seed for randomised data filtering, defaults to 12345
    :type seed: int, optional
    :param trackml_input_path: Path of the TrackML event, which is to be used
    :type trackml_input_path: str
    :param f: fraction of the data, which is to be used, defaults to 0.1
    :type f: float, optional

    :return: Qallse event metadata
    :rtype: dict
    :return: Path, where the filtered data is stored
    :rtype: str
    """
    prefix = f"data_frac{int(f*100)}_seed{seed}_num_parts{num_angle_parts}"

    hits, truth, particles, doublets, metadata = create_dataset(
        hits=event_hits,
        truth=event_truth,
        particles=event_particles,
        density=f,
        random_seed=seed,
    )

    return {
        "hits": hits,
        "truth": truth,
        "particles": particles,
        "doublets": doublets,
        "metadata": metadata,
    }


def create_qallse_datawrapper(hits, truth) -> DataWrapper:
    """
    Creates a qallse data wrapper form a given event path

    :return: Qallse data wrapper
    :rtype: DataWrapper
    """
    dw = DataWrapper(hits=hits, truth=truth)
    return {"data_wrapper": dw}
