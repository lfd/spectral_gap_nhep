import os
import csv
from typing import Tuple, Dict
import pickle

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
