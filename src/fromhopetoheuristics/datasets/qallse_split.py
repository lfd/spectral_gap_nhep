from pathlib import PurePosixPath
from typing import Any, Dict
import pickle
import os
import glob

import numpy as np
from fromhopetoheuristics.utils.qaoa_utils import (
    dict_QUBO_to_matrix,
)
from kedro.io import AbstractVersionedDataset

from qallse import dumper

import logging

log = logging.getLogger(__file__)


class QuboDataset(AbstractVersionedDataset):

    def __init__(self, filepath: str, version):
        self._filepath = PurePosixPath(filepath)
        super().__init__(self._filepath, version)

    def _get_versioned_path(self, version: str) -> PurePosixPath:
        return self._filepath / version / f"*{self._filepath.name}.pickle"

    def load(self) -> np.ndarray:
        actual_path = self._get_load_path().parent

        qubos = {}
        qubos_paths = glob.glob(os.path.join(actual_path, "*qubo.pickle"))
        self._n_qubos = len(qubos_paths)
        for i, qubo_path in enumerate(qubos_paths):
            with open(qubo_path, "rb") as f:
                Q = pickle.load(f)

            # FIXME: Warning triggered
            # if len(qubo) > 18:
            #     log.warning(f"Too many variables for qubo {qubo_path}")
            #     qubo = None
            # elif len(qubo) == 0:
            #     log.warning(f"Empty QUBO {qubo_path}")
            #     qubo = None
            qubos[i] = Q

        return qubos

    def save(self, qubos: Dict) -> None:
        actual_path = self._get_save_path().parent
        os.makedirs(actual_path, exist_ok=True)
        for i, qubo in qubos.items():
            dumper.dump_model(
                qubo,
                actual_path,
                f"{i}_",
                qubo_kwargs=dict(w_marker=None, c_marker=None),
            )

    def _exists(self) -> bool:
        path = self._get_load_path()
        return os.Path(path.as_posix()).exists()

    def _describe(self) -> Dict[str, Any]:
        qubos_paths = glob.glob(os.path.join(self._filepath, "*qubo.pickle"))

        return dict(
            version=self._version,
            filepath=self._filepath,
            number_of_qubos=len(qubos_paths),
        )
