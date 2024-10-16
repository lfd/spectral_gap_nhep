import os
import csv
from typing import Tuple, Optional
import pandas as pd
import numpy as np

from qallse import dumper

from fromhopetoheuristics.utils.model import QallseSplit

import logging

logger = logging.getLogger(__file__)


def save_to_csv(data: list, path: str, filename: str) -> None:
    """
    Stores one result tuple to a CSV file

    :param data: The data row to be saved
    :type data: list
    :param path: The parent path of the csv file
    :type path: str
    :param filename: The csv file name
    :type filename: str
    """

    sd = os.path.abspath(path)
    os.makedirs(sd, exist_ok=True)

    with open(path + "/" + filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(data)


def store_qubo(data_path: str, model: QallseSplit, qubo_prefix: str) -> str:
    """
    Stores a qubo in dict form using Qallse

    :param data_path: The path of the TrackML data and parent path of the QUBO
    :type data_path: str
    :param model: The Qallse model, from which the QUBO is generated
    :type model: QallseSplit
    :param qubo_prefix: Path, where to store the QUBO
    :type qubo_prefix: str
    :return: Path to QUBO in dict form
    :rtype: str
    """
    qubo_path = os.path.join(os.path.dirname(data_path), "QUBO")
    os.makedirs(qubo_path, exist_ok=True)
    dumper.dump_model(
        model,
        qubo_path,
        qubo_prefix,
        qubo_kwargs=dict(w_marker=None, c_marker=None),
    )
    qubo_path = os.path.join(qubo_path, qubo_prefix + "qubo.pickle")
    logger.info("Wrote qubo to", qubo_path)

    return qubo_path


def load_params_from_csv(
    file_path: str, **filter_args
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Loads result tuples from CSV file

    :param file_path: The csv file path
    :type file_path: str
    :return: The beta and gamma parameters for the QAOA execution
    :rtype: Tuple[Optional[np.ndarray], Optional[np.ndarray]]
    """
    df = pd.read_csv(file_path)
    query_str = " and ".join([f"{k} == {v}" for k, v in filter_args.items()])
    df = df.query(query_str)
    if len(df.index) == 0:
        logger.warning(f"No data for filter_args: {filter_args}")
        return None, None
    assert len(df.index) == 1, (
        "In the filtered QAOA dataframe only one row should be present, got "
        f"{len(df.index)}. Maybe you need to adapt your filter args?"
    )

    beta_colnames = [cn for cn in df.columns if cn[:4] == "beta"]
    gamma_colnames = [cn for cn in df.columns if cn[:5] == "gamma"]
    beta_colnames.sort()
    gamma_colnames.sort()
    p = len(beta_colnames)
    betas = np.ndarray(p)
    gammas = np.ndarray(p)

    for i, row in df.iterrows():
        for c in range(len(beta_colnames)):
            betas[c] = row[beta_colnames[c]]
            gammas[c] = row[gamma_colnames[c]]

    return betas, gammas
