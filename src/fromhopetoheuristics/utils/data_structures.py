import numpy as np

from qallse.data_structures import Triplet, Doublet, Hit


class ExtendedDoublet(Doublet):
    """Same as the Qallse doublet, but with additional information on the angle
    in the XY Plane"""

    def __init__(self, hit_start: Hit, hit_end: Hit):
        """
        Create an extended doublet.

        Parameters
        ----------
        hit_start : Hit
            The hit at the start of the doublet.
        hit_end : Hit
            The hit at the end of the doublet.

        Notes
        -----
        The angle in the XY plane between this doublet and the Y axis
        is stored in `xy_angle`.
        """
        super().__init__(hit_start, hit_end)

        #: The angle in the XY plane between this doublet and the Y axis.
        self.xy_angle = np.arctan2(*self.coord_2d)


class ExtendedTriplet(Triplet):
    """Same as the Qallse triplet, but with additional information on the angle
    in the XY Plane for the first doublet"""

    def __init__(self, d1: ExtendedDoublet, d2: ExtendedDoublet):
        """
        Create an extended triplet.

        Parameters
        ----------
        d1, d2 : ExtendedDoublet
            The two doublets composing this triplet.

        Notes
        -----
        The angle in the XY plane of the first hit is stored in `xy_angle`.
        """
        super().__init__(d1, d2)

        # Angle in the XY-Plane of the first hit
        self.xy_angle = self.d1.xy_angle
