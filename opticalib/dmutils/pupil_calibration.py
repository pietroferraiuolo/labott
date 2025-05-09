"""
Author(s)
---------
- Pietro Ferraiuolo : written in 2025

Description
-----------

"""

import numpy as _np
from opticalib import typings as _ot

def act_coordinates_tranformation(dm: _ot.DeformableMirrorDevice, img: _ot.Optional[_ot.ImageData] = None) -> _ot.MatrixLike:
    # Get the dm's actuator coordinates, confronts them with the image coordinates
    # and returns the transformation matrix

    
    ...