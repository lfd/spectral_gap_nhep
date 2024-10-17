from pathlib import PurePosixPath
from typing import Any, Dict
import os

import fsspec
import numpy as np
from fromhopetoheuristics.utils.model import QallseSplit

from kedro.io import AbstractVersionedDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path, Version

from qallse import dumper

import logging

logger = logging.getLogger(__file__)


class QallseSplitDataset(AbstractVersionedDataset[np.ndarray, np.ndarray]):
    def __init__(self, filepath: str, version: Version = None):
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._fs = fsspec.filesystem(self._protocol)

        super().__init__(
            filepath=PurePosixPath(path),
            version=version,
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
        logger.info("Wrote qubo to", qubo_path)

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath, version=self._version, protocol=self._protocol
        )
