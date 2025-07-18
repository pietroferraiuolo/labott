"""
IFF Module
==========
Author(s):
----------
- Pietro Ferraiuolo
- Runa Briguglio

Description:
------------
This module contains the necessary high/user-leve functions to acquire the IFF data, 
given a deformable mirror and an interferometer.
"""

import os as _os
import numpy as _np
from opticalib.core.root import folders as _fn
from opticalib.core import read_config as _rif
from . import iff_acquisition_preparation as _ifa
from opticalib.ground.osutils import newtn as _ts, save_fits as _sf
from opticalib import typings as _ot


def iffDataAcquisition(
    dm: _ot.DeformableMirrorDevice,
    interf: _ot.InterferometerDevice,
    modesList: _ot.Optional[_ot.ArrayLike] = None,
    amplitude: _ot.Optional[float | _ot.ArrayLike] = None,
    template: _ot.Optional[_ot.ArrayLike] = None,
    shuffle: bool = False,
) -> str:
    """
    This is the user-lever function for the acquisition of the IFF data, given a
    deformable mirror and an interferometer.

    Except for the devices, all the arguments are optional, as, by default, the
    values are taken from the `iffConfig.ini` configuration file.

    Parameters
    ----------
    dm: DeformableMirrorDevice
        The inizialized deformable mirror object
    interf: InterferometerDevice
        The initialized interferometer object to take measurements
    modesList: ArrayLike , optional
        list of modes index to be measured, relative to the command matrix to be used
    amplitude: float | ArrayLike, optional
        command amplitude
    template: ArrayLike , oprional
        template file for the command matrix
    shuffle: bool , optional
        if True, shuffle the modes before acquisition

    Returns
    -------
    tn: str
        The tracking number of the dataset acquired, saved in the OPDImages folder
    """
    ifc = _ifa.IFFCapturePreparation(dm)
    tch = ifc.createTimedCmdHistory(modesList, amplitude, template, shuffle)
    info = ifc.getInfoToSave()
    tn = _ts()
    iffpath = _os.path.join(_fn.IFFUNCTIONS_ROOT_FOLDER, tn)
    if not _os.path.exists(iffpath):
        _os.mkdir(iffpath)
    try:
        for key, value in info.items():
            if not isinstance(value, _np.ndarray):
                tvalue = _np.array(value)
            else:
                tvalue = value
            if key == "shuffle":
                with open(_os.path.join(iffpath, f"{key}.dat"), "w") as f:
                    f.write(str(value))
            else:
                _sf(_os.path.join(iffpath, f"{key}.fits"), tvalue, overwrite=True)
    except KeyError as e:
        print(f"KeyError: {key}, {e}")
    _rif.copyIffConfigFile(tn)
    for param, value in zip(
        ["modeid", "modeamp", "template"], [modesList, amplitude, template]
    ):
        if value is not None:
            _rif.updateIffConfig(tn, param, value)
    dm.uploadCmdHistory(tch)
    dm.runCmdHistory(interf, save=tn)
    return tn


# def iffCapture(tn):
#     """
#     This function manages the interfacing equence for collecting the IFF data
#     Parameters
#     ----------------
#     tn: string
#         the tracking number in the xxx folder where the cmd history is saved
#     Returns
#     -------
#     """

#     cmdHist = getCmdHist(tn)
#     dm.uploadCmdHist(cmdHist)
#     dm.runCmdHist()
#     print('Now launching the acquisition sequence')
#     start4DAcq(tn)
#     print('Acquisition completed. Dataset tracknum:')
#     print(tn)
