from trackml.dataset import load_event, load_dataset
import pandas as pd
from typing import Tuple
import os
from datetime import datetime

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
    result_path_prefix: str, seed: int, f: float = 0.1
) -> Tuple[dict, str]:
    """
    Creates/filters and stores TrackML event data for a given fraction of data.
    Uses Qallse to filter the data.

    :param result_path_prefix: Where to store the data
    :type result_path_prefix: str
    :param seed: seed for randomised data filtering, defaults to 12345
    :type seed: int, optional
    :param f: fraction of the data, which is to be used, defaults to 0.1
    :type f: float, optional

    :return: Qallse event metadata
    :rtype: dict
    :return: Path, where the filtered data is stored
    :rtype: str
    """
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_path = os.path.join(result_path_prefix, time_stamp, "qallse_data")
    prefix = f"data_frac{int(f*100)}_seed{seed}"

    metadata, event_path = create_dataset(
        density=f,
        output_path=output_path,
        prefix=prefix,
        gen_doublets=True,
        random_seed=seed,
    )

    return {
        "metadata": metadata,
        "event_path": event_path,
    }


def create_qallse_datawrapper(event_path: str) -> DataWrapper:
    """
    Creates a qallse data wrapper form a given event path

    :param event_path: The event path
    :type event_path: str
    :return: Qallse data wrapper
    :rtype: DataWrapper
    """
    dw = DataWrapper.from_path(event_path)
    return {"data_wrapper": dw}
