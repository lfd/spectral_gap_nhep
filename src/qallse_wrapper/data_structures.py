import numpy as np

from qallse.data_structures import Triplet, Doublet, Hit

class ExtendedDoublet(Doublet):
    """Same as the Qallse doublet, but with additional information on the angle
        in the XY Plane"""

    def __init__(self, hit_start: Hit, hit_end: Hit):
        super().__init__(hit_start, hit_end)

        #: The angle in the XY plane between this doublet and the Y axis.
        self.xy_angle = np.arctan2(*self.coord_2d)

class ExtendedTriplet(Triplet):
    """Same as the Qallse triplet, but with additional information on the angle
        in the XY Plane for the first doublet"""

    def __init__(self, d1: ExtendedDoublet, d2: ExtendedDoublet):
        super().__init__(d1, d2)

        # Angle in the XY-Plane of the first hit
        self.xy_angle = self.d1.xy_angle



