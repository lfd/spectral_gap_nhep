from qallse.qallse_d0 import QallseD0, D0Config
from qallse.dumper import use_markers, xplets_to_serializable_dict
import numpy as np

from fromhopetoheuristics.utils.data_structures import (
    ExtendedDoublet,
    ExtendedTriplet,
)

from typing import Tuple, List
import logging

log = logging.getLogger(__name__)


class SplitConfig(D0Config):
    xy_angle_parts = 64
    geometric_index = 0


class QallseSplit(QallseD0):
    """
    A QallseSplit is a QallseD0 model that is split into multiple parts based on
    the angle in the XY plane of the first hit of each doublet. This is done by
    overriding the `_create_doublets` and `_create_triplets` methods to apply
    early cuts to the doublets and triplets based on the angle part of the first
    hit. The angle part is determined by the `geometric_index` parameter of
    the `config` attribute.

    Attributes
    ----------
    config : SplitConfig
        The configuration for the QallseSplit model.

    Methods
    -------
    _create_doublets(initial_doublets)
        Generate Doublet structures from the initial doublets, calling
        `_is_invalid_doublet` to apply early cuts.
    _create_triplets()
        Generate Triplet structures from Doublets, calling
        `_is_invalid_triplet` to apply early cuts.
    _is_invalid_triplet(triplet)
        Check if a Triplet is invalid based on the angle part of the first hit.
    _get_base_config()
        Return the base configuration for the QallseSplit model.
    serialize()
        Serialize the model and their associated xplets.
    """

    config = SplitConfig()

    def _create_doublets(self, initial_doublets: List[Tuple[int, int]]) -> None:
        """
        Generate Doublet structures from the initial doublets, calling
        _is_invalid_doublet to apply early cuts.

        Parameters
        ----------
        initial_doublets : List[Tuple[int, int]]
            A list of tuples containing start and end hit IDs.

        Returns
        -------
        None
        """
        doublets: List[ExtendedDoublet] = []
        for start_id, end_id in initial_doublets:
            start, end = self.hits[start_id], self.hits[end_id]
            d = ExtendedDoublet(start, end)
            if not self._is_invalid_doublet(d):
                start.outer.append(d)
                end.inner.append(d)
                doublets.append(d)

        self.logger.info(f"created {len(doublets)} doublets.")
        self.doublets = doublets

    def _create_triplets(self) -> None:
        """
        Generate Triplet structures from Doublets, calling
        _is_invalid_triplet to apply early cuts.

        Returns
        -------
        None
        """
        triplets: List[ExtendedTriplet] = []
        for d1 in self.doublets:
            for d2 in d1.h2.outer:
                t = ExtendedTriplet(d1, d2)
                if not self._is_invalid_triplet(t):
                    d1.outer.append(t)
                    d2.inner.append(t)
                    triplets.append(t)
        self.logger.info(f"created {len(triplets)} triplets.")
        self.triplets = triplets

    def _is_invalid_triplet(self, triplet: ExtendedTriplet) -> bool:
        """
        Check if a triplet is invalid with respect to the geometric index.

        Parameters
        ----------
        triplet : ExtendedTriplet
            The triplet to check

        Returns
        -------
        bool
            True if invalid, False otherwise.
        """
        if super()._is_invalid_triplet(triplet):
            return True

        angle_part_size = 2 * np.pi / self.config.xy_angle_parts
        angle_min = -np.pi + self.config.geometric_index * angle_part_size
        angle_max = -np.pi + (self.config.geometric_index + 1) * angle_part_size

        if triplet.xy_angle < angle_min or triplet.xy_angle >= angle_max:
            return True

        return False

    def _get_base_config(self):
        return SplitConfig()

    def serialize(self) -> Tuple:
        """
        Serialize model and their associated xplets.

        Parameters
        ----------
        qubos : Dict[str, QallseSplit]
            A dictionary of QUBOs, where the keys are the angle part indices and
            the values are the QUBOs themselves.

        Returns
        -------
        Dict[str, pd.DataFrame]
            A dictionary with two keys: "qubos" and "xplets". The value for "qubos"
            is a Pandas DataFrame, where the index is the angle part index and the
            columns are the QUBO matrix elements. The value for "xplets" is also a
            Pandas DataFrame, where the index is the angle part index and the columns
            are the xplet elements.
        """
        qubo_kwargs = dict(w_marker=None, c_marker=None)

        xplet = xplets_to_serializable_dict(self)
        with use_markers(self, **qubo_kwargs) as altered_model:
            qubo = altered_model.to_qubo()

        return qubo, xplet


def build_model(
    doublets: List[ExtendedDoublet], model: QallseSplit, add_missing: bool
) -> None:
    """
    Prepares and builds the QUBO model from the given doublets.

    Parameters
    ----------
    doublets : List[ExtendedDoublet]
        A list of extended doublets to be used in the model.
    model : QallseSplit
        The QallseSplit model to build the QUBO.
    add_missing : bool
        Flag indicating whether to add missing doublets.

    Returns
    -------
    None
    """
    if add_missing:
        log.info("Cheat on, adding missing doublets.")
        doublets = model.dataw.add_missing_doublets(doublets)
    else:
        p, r, ms = model.dataw.compute_score(doublets)
        log.info(
            f"Precision: {p * 100:.4f}%, Recall:{r * 100:.4f}%, Missing: {len(ms)}"
        )

    model.build_model(doublets=doublets)
