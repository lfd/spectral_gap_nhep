from hepqpr.qallse.qallse_d0 import QallseD0, D0Config

from qallse_wrapper.data_structures import ExtendedDoublet, ExtendedTriplet

class SplitConfig(D0Config):
    num_triplets = 500


class QallseSplit(QallseD0):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = SplitConfig()

    def _create_doublets(self, initial_doublets):
        # Generate Doublet structures from the initial doublets, calling _is_invalid_doublet to apply early cuts
        doublets = []
        for (start_id, end_id) in initial_doublets:
            start, end = self.hits[start_id], self.hits[end_id]
            d = ExtendedDoublet(start, end)
            if not self._is_invalid_doublet(d):
                start.outer.append(d)
                end.inner.append(d)
                doublets.append(d)

        self.logger.info(f'created {len(doublets)} doublets.')
        self.doublets = doublets

    def _create_triplets(self):
        # Generate Triplet structures from Doublets, calling _is_invalid_triplet to apply early cuts
        triplets = []
        for d1 in self.doublets:
            for d2 in d1.h2.outer:
                t = ExtendedTriplet(d1, d2)
                if not self._is_invalid_triplet(t):
                    d1.outer.append(t)
                    d2.inner.append(t)
                    triplets.append(t)
        self.logger.info(f'created {len(triplets)} triplets.')

        triplets.sort(key=lambda t: t.xy_angle)

        self.triplets = triplets[:self.config.num_triplets]

