import pickle
import os
from typing import List

from qallse.cli.func import build_model, solve_neal
from qallse.data_wrapper import DataWrapper
from fromhopetoheuristics.utils.model import QallseSplit
from fromhopetoheuristics.utils.data_utils import store_qubo
from fromhopetoheuristics.utils.qaoa_utils import (
    dict_QUBO_to_matrix,
)

import logging

log = logging.getLogger(__name__)


def build_qubos(
    data_wrapper: DataWrapper,
    event_path: str,
    num_angle_parts: int,
):
    """
    Creates partial QUBO from TrackML data using Qallse. The data is split into
    several parts by the angle in the XY-plane of the detector, from which the
    QUBO is built.

    :param data_wrapper: Qallse data wrapper
    :type data_wrapper: DataWrapper
    :param event_path: path, from which to load the TrackML data
    :type event_path: str
    :param num_angle_parts: Number of angle segments in the detector, equals
        the number of resulting QUBOs
    :type num_angle_parts: int
    :return: the path to the QUBO in dict form
    :rtype: List[str]
    """
    qubo_paths = []
    qubo_base_path = os.path.join(os.path.dirname(event_path), "QUBO")
    for i in range(num_angle_parts):
        qubo_prefix = f"angle_index{i:02d}_"
        qubo_path = os.path.join(qubo_base_path, f"{qubo_prefix}qubo.pickle")
        if not os.path.exists(qubo_path):
            extra_config = {
                "geometric_index": i,
                "xy_angle_parts ": num_angle_parts,
            }
            model = QallseSplit(data_wrapper, **extra_config)
            build_model(event_path, model, False)

            qubo_path = store_qubo(event_path, model, qubo_prefix=qubo_prefix)
            log.info(f"Generated QUBO at {qubo_path}")
        qubo_paths.append(qubo_path)

    return {"qubo_paths": qubo_paths}


def load_qubos(qubo_paths: List[str]):
    qubos = []
    for qubo_path in qubo_paths:
        log.info(f"Loading QUBO from {qubo_path}")
        with open(qubo_path, "rb") as f:
            Q = pickle.load(f)

        qubo = dict_QUBO_to_matrix(Q)

        if len(qubo) > 18:
            log.warning(f"Too many variables for qubo {qubo_path}")
            qubo = None
        elif len(qubo) == 0:
            log.warning(f"Empty QUBO {qubo_path}")
            qubo = None
        qubos.append(qubo)

    return {"qubos": qubos}


def solve_qubos(
    data_wrapper: DataWrapper,
    qubo_paths: List[str],
    event_path: str,
    seed: int,
):
    responses = []
    cl_solver_path = os.path.join(os.path.dirname(event_path), "cl_solver")
    os.makedirs(cl_solver_path, exist_ok=True)
    for i, qubo_path in enumerate(qubo_paths):
        with open(qubo_path, "rb") as f:
            qubo = pickle.load(f)
        response = solve_neal(qubo, seed=seed)
        # print_stats(data_wrapper, response, qubo) # FIXME: solve no track found case

        filename = f"neal_response{i:02d}.pickle"
        oname = os.path.join(cl_solver_path, filename)
        with open(oname, "wb") as f:
            pickle.dump(response, f)
        log.info(f"Wrote response to {oname}")
        responses.append(response)

    return {"responses": responses}
