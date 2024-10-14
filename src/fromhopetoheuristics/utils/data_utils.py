import os
import csv
from typing import Tuple, Dict
import pickle

from qallse.dsmaker import create_dataset
from qallse.data_wrapper import DataWrapper
from qallse import dumper

from fromhopetoheuristics.utils.model import QallseSplit

import logging

logger = logging.getLogger(__file__)


def create_dataset_track_reconstruction(
    result_path_prefix: str, seed: int = 12345, f: float = 0.1
) -> Tuple[DataWrapper, str]:
    ## Build QUBO
    """
    Creates/filters and stores TrackML event data for a given fraction of data.
    Uses Qallse to filter the data.

    :param result_path_prefix: Where to store the data
    :type result_path_prefix: str
    :param seed: seed for randomised data filtering, defaults to 12345
    :type seed: int, optional
    :param f: fraction of the data, which is to be used, defaults to 0.1
    :type f: float, optional

    :return: Qallse data wrapper
    :rtype: DataWrapper
    :return: Path, where the filtered data is stored
    :rtype: str
    """
    output_path = os.path.join(result_path_prefix, "qallse_data")
    prefix = f"data_frac{int(f*100)}_seed{seed}"

    metadata, path = create_dataset(
        density=f,
        output_path=output_path,
        prefix=prefix,
        gen_doublets=True,
        random_seed=seed,
    )

    dw = DataWrapper.from_path(path)

    return dw, path


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


def store_qubo(
    data_path: str, model: QallseSplit, geometric_index: int
) -> Dict[Tuple[str, str], float]:
    """
    Stores a qubo in dict form using Qallse

    :param data_path: The path of the TrackML data and parent path of the QUBO
    :type data_path: str
    :param model: The Qallse model, from which the QUBO is generated
    :type model: QallseSplit
    :param geometric_index: The index of the angle segment in the detector, for
        which the QUBO is built
    :type geometric_index: int
    :return: QUBO in dict form
    :rtype: Dict[Tuple[str, str], float]
    """
    qubo_path = os.path.join(os.path.dirname(data_path), "QUBO")
    os.makedirs(qubo_path, exist_ok=True)
    qubo_prefix = f"angle_index{geometric_index:02d}_"
    dumper.dump_model(
        model,
        qubo_path,
        qubo_prefix,
        qubo_kwargs=dict(w_marker=None, c_marker=None),
    )
    qubo_path = os.path.join(qubo_path, qubo_prefix + "qubo.pickle")
    logger.info("Wrote qubo to", qubo_path)

    with open(qubo_path, "rb") as f:
        Q = pickle.load(f)

    return Q
