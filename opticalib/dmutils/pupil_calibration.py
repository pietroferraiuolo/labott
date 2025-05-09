"""
Author(s)
---------
- Pietro Ferraiuolo : written in 2025

Description
-----------

"""

import numpy as _np
from opticalib import typings as _ot


class PupilCalibrator():
    """
    Class to calibrate a DM given a pupil diofferent from that of the calibration
    data loaded.
    """

    def __init__(self, tn: str, dm: _ot.DeformableMirrorDevice) -> None:
        self._dm = dm
        self._tn = tn


    def act_coordinates_tranformation(self, dm: _ot.DeformableMirrorDevice, img: _ot.Optional[_ot.ImageData] = None) -> _ot.MatrixLike:
        # Get the dm's actuator coordinates, confronts them with the image coordinates
        # and returns the transformation matrix
        ## Sudo code here
        # 
        ...