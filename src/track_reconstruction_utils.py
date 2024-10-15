from typing import Optional
import numpy as np

from qallse.cli.func import build_model
from qallse.data_wrapper import DataWrapper

from qallse_wrapper.model import QallseSplit
from qaoa_utils import dict_QUBO_to_matrix
from data_utils import store_qubo

import logging

logger = logging.getLogger(__file__)


def provide_track_reconstruction_QUBO(
    dw: DataWrapper,
    data_path: str,
    geometric_index: int,
) -> Optional[np.ndarray]:
    """
    Creates partial QUBO from TrackML data using Qallse. The data is split into
    several parts by the angle in the XY-plane of the detector, from which the
    QUBO is built.

    :param dw: Qallse data wrapper
    :type dw: DataWrapper
    :param data_path: path, from which to load the TrackML data
    :type data_path: str
    :param geometric_index: Angle segment in the detector, from which to build
        the QUBO
    :type geometric_index: int
    :return: the QUBO matrix
    :rtype: Optional[np.ndarray]
    """
    extra_config = {"geometric_index": geometric_index}
    model = QallseSplit(dw, **extra_config)
    build_model(data_path, model, False)

    Q = store_qubo(data_path, model, geometric_index)
    qubo = dict_QUBO_to_matrix(Q)

    if len(qubo) > 18:
        logger.warning(f"Too many variables at index {geometric_index}")
        return None
    if len(qubo) == 0:
        logger.warning(f"Empty QUBO at index {geometric_index}")
        return None

    return qubo
