from trackml.dataset import load_event, load_dataset
import pandas as pd
from typing import List
import logging

log = logging.getLogger(__name__)


def load_event_data(sample) -> dict:
    hits, cells, particles, truth = load_event(f"dataset/{sample}")

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
