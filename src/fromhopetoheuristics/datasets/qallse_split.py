from pathlib import PurePosixPath
from typing import Any, Dict
import pickle
import os
import glob

import numpy as np
from fromhopetoheuristics.utils.qaoa_utils import (
    dict_QUBO_to_matrix,
)
from kedro.io import AbstractDataset

from qallse import dumper

import logging

log = logging.getLogger(__file__)


class QuboDataset(AbstractDataset):

    def __init__(self, filepath: str, **kwargs):
        self._filepath = PurePosixPath(filepath)

    def load(self) -> np.ndarray:
        qubos = {}
        qubos_paths = glob.glob(os.path.join(self._filepath, "*qubo.pickle"))
        self._n_qubos = len(qubos_paths)
        for i, qubo_path in enumerate(qubos_paths):
            with open(qubo_path, "rb") as f:
                Q = pickle.load(f)

            qubo = dict_QUBO_to_matrix(Q)

            # FIXME: Warning triggered
            # if len(qubo) > 18:
            #     log.warning(f"Too many variables for qubo {qubo_path}")
            #     qubo = None
            # elif len(qubo) == 0:
            #     log.warning(f"Empty QUBO {qubo_path}")
            #     qubo = None
            qubos[i] = qubo

        return qubos

    def save(self, qubos: Dict) -> None:
        for i, qubo in qubos.items():
            dumper.dump_model(
                qubo,
                self._filepath,
                f"{i}_",
                qubo_kwargs=dict(w_marker=None, c_marker=None),
            )

    def _describe(self) -> Dict[str, Any]:
        qubos_paths = glob.glob(os.path.join(self._filepath, "*qubo.pickle"))

        return dict(
            filepath=self._filepath,
            number_of_qubos=len(qubos_paths),
        )
