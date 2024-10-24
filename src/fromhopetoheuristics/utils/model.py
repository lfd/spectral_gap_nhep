from qallse.qallse_d0 import QallseD0, D0Config
import numpy as np

from fromhopetoheuristics.utils.data_structures import (
    ExtendedDoublet,
    ExtendedTriplet,
)


class SplitConfig(D0Config):
    xy_angle_parts = 64
    geometric_index = 0


class QallseSplit(QallseD0):
    config = SplitConfig()

    def _create_doublets(self, initial_doublets):
        # Generate Doublet structures from the initial doublets,
        # calling _is_invalid_doublet to apply early cuts
        doublets = []
        for start_id, end_id in initial_doublets:
            start, end = self.hits[start_id], self.hits[end_id]
            d = ExtendedDoublet(start, end)
            if not self._is_invalid_doublet(d):
                start.outer.append(d)
                end.inner.append(d)
                doublets.append(d)

        self.logger.info(f"created {len(doublets)} doublets.")
        self.doublets = doublets

    def _create_triplets(self):
        # Generate Triplet structures from Doublets,
        # calling _is_invalid_triplet to apply early cuts
        triplets = []
        for d1 in self.doublets:
            for d2 in d1.h2.outer:
                t = ExtendedTriplet(d1, d2)
                if not self._is_invalid_triplet(t):
                    d1.outer.append(t)
                    d2.inner.append(t)
                    triplets.append(t)
        self.logger.info(f"created {len(triplets)} triplets.")
        self.triplets = triplets

    def _is_invalid_triplet(self, triplet: ExtendedTriplet):
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


def build_model(doublets, model, add_missing):

    # prepare doublets
    if add_missing:
        print("Cheat on, adding missing doublets.")
        doublets = model.dataw.add_missing_doublets(doublets)
    else:
        p, r, ms = model.dataw.compute_score(doublets)
        print(
            f"INPUT -- precision (%): {p * 100:.4f}, recall (%):\
                {r * 100:.4f}, missing: {len(ms)}"
        )

    # build the qubo
    model.build_model(doublets=doublets)
