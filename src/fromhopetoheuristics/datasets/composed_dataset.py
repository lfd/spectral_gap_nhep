from kedro.io import AbstractVersionedDataset
from pathlib import PurePosixPath
import pickle
import os
import glob
from typing import Any, Dict
import importlib
import logging

log = logging.getLogger(__file__)


class ComposedDataset(AbstractVersionedDataset):
    def __init__(self, filepath: str, dataset, version, **kwargs):
        self.fp = PurePosixPath(filepath)
        self.dataset = dataset
        self.init_kwargs = kwargs
        package, module = self.dataset["type"].split(".")
        resolver = self.dataset["resolver"]

        try:
            self.dataset_module = getattr(
                importlib.import_module(f"kedro_datasets.{package}.{resolver}"),
                module,
            )
        except Exception as err:
            raise RuntimeError(
                f'Error importing "{module}" from \
                    "{package}.{resolver}". Are you sure it exists ?'
            ) from err

        super().__init__(self.fp.parent / self.fp.stem, version)

    def _get_versioned_path(self, version: str) -> PurePosixPath:
        return self.fp.parent / self.fp.stem / version / f"*{self.fp.suffix}"

    def load(self) -> Dict[str, Any]:
        outputs = {}
        for ident in glob.glob(self._get_load_path().as_posix()):
            outputs[PurePosixPath(ident).stem] = self.dataset_module(
                filepath=ident
            ).load()
        return outputs

    def save(self, outputs: Dict[str, Any], **kwargs) -> None:
        actual_path = self._get_save_path().parent
        os.makedirs(actual_path, exist_ok=True)

        for ident, output in outputs.items():
            self.dataset_module(
                filepath=os.path.join(actual_path, f"{ident}{self.fp.suffix}"),
            ).save(output, **kwargs)

    def _exists(self) -> bool:
        return os.path.exists(self._get_load_path().as_posix())

    def _describe(self) -> Dict[str, Any]:
        return dict(
            version=self._version,
            filepath=self.fp,
        )
