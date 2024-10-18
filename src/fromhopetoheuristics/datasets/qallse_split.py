from pathlib import PurePosixPath
from typing import Any, Dict
import pickle
import os
import glob

import fsspec
import numpy as np
from fromhopetoheuristics.utils.model import QallseSplit
from fromhopetoheuristics.utils.qaoa_utils import (
    dict_QUBO_to_matrix,
)
from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path, Version

from qallse import dumper

import logging

log = logging.getLogger(__file__)


class QallseSplitDataset(AbstractDataset):
    def __init__(self, filepath: str):
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._fs = fsspec.filesystem(self._protocol)

        super().__init__(
            filepath=PurePosixPath(path),
            exists_function=self._fs.exists,
            glob_function=self._fs.glob,
        )

    # def load(self) -> np.ndarray:
    #     load_path = get_filepath_str(self._get_load_path(), self._protocol)
    #     with self._fs.open(load_path, mode="r") as f:
    #         image = Image.open(f).convert("RGBA")
    #         return np.asarray(image)

    def save(self, model: QallseSplit, qubo_prefix: str) -> None:
        qubo_path = os.path.join(self._get_save_path(), "QUBO")
        os.makedirs(qubo_path, exist_ok=True)
        dumper.dump_model(
            model,
            qubo_path,
            qubo_prefix,
            qubo_kwargs=dict(w_marker=None, c_marker=None),
        )
        qubo_path = os.path.join(qubo_path, qubo_prefix + "qubo.pickle")
        log.info("Wrote qubo to", qubo_path)

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath, version=self._version, protocol=self._protocol
        )


class QuboDataset(AbstractDataset):
    def __init__(self, filepath: str):
        self._filepath = PurePosixPath(filepath)

    def load(self) -> np.ndarray:
        qubos = {}
        qubos_paths = glob.glob(os.path.join(self._filepath, "*qubo.pickle"))
        for i, qubo_path in enumerate(qubos_paths):
            with open(qubo_path, "rb") as f:
                Q = pickle.load(f)

            qubo = dict_QUBO_to_matrix(Q)

            if len(qubo) > 18:
                log.warning(f"Too many variables for qubo {qubo_path}")
                qubo = None
            elif len(qubo) == 0:
                log.warning(f"Empty QUBO {qubo_path}")
                qubo = None
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
        return dict(
            filepath=self._filepath,
        )
