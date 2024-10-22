import os
from typing import Dict

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
    doublets,
    num_angle_parts: int,
    geometric_index: int = -1,
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
    :param geometric_index: The angle part, for which to build the QUBO, if -1
        build all
    :type geometric_index: int
    :return: the path to the QUBO in dict form
    :rtype: List[str]
    """
    qubos = {}
    log.info(f"Generating {num_angle_parts} QUBOs")

    if geometric_index == -1:
        angle_parts = range(num_angle_parts)
    else:
        angle_parts = [geometric_index]

    for i in angle_parts:
        extra_config = {
            "geometric_index": i,
            "xy_angle_parts ": num_angle_parts,
        }
        model = QallseSplit(data_wrapper, **extra_config)
        build_model(doublets=doublets, model=model, add_missing=False)

        qubos[i] = model
        log.info(f"Generated QUBO {i+1}/{num_angle_parts}")
    return {"qubos": qubos}


def solve_qubos(
    qubos: Dict,
    seed: int,
):
    responses = {}
    log.info(f"Solving {len(qubos)} QUBOs")

    for i, qubo in qubos.items():
        response = solve_neal(qubo, seed=seed)
        # print_stats(data_wrapper, response, qubo) # FIXME: solve no track found case

        log.info(f"Solved QUBO {i+1}/{len(qubos)}")
        responses[i] = response

    return {"responses": responses}  # FIXME find suitable catalog entry
