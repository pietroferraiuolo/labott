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
from aoptics.core.root import _folds as _fn
from aoptics.core import read_iffconfig as _rif
from . import iff_acquisition_preparation as _ifa
from aoptics.ground.osutils import newtn as _ts, save_fits as _sf


def iffDataAcquisition(
    dm, interf, modesList=None, amplitude=None, template=None, shuffle=False
):
    """
    This is the user-lever function for the acquisition of the IFF data, given a
    deformable mirror and an interferometer.

    Except for the devices, all the arguments are optional, as, by default, the
    values are taken from the `iffConfig.ini` configuration file.

    Parameters
    ----------
    dm: object
        The inizialized deformable mirror object
    interf: object
        The initialized interferometer object to take measurements
    modesList: int | list, array like , optional
        list of modes index to be measured, relative to the command matrix to be used
    amplitude: float , optional
        command amplitude
    template: string , oprional
        template file for the command matrix
    modalBase: string , optional
        identifier of the modal base to be used
    shuffle: bool , optional
        if True, shuffle the modes before acquisition
    *dmargs: list
        additional arguments to be passed to the deformable mirror's `runCmdHistory`
        method.

    Returns
    -------
    tn: string
        The tracking number of the dataset acquired, saved in the OPDImages folder
    """
    ifc = _ifa.IFFCapturePreparation(dm)
    tch = ifc.createTimedCmdHistory(modesList, amplitude, template, shuffle)
    info = ifc.getInfoToSave()
    tn = _ts.now()
    iffpath = _os.path.join(_fn.IFFUNCTIONS_ROOT_FOLDER, tn)
    if not _os.path.exists(iffpath):
        _os.mkdir(iffpath)
    try:
        for key, value in info.items():
            if not isinstance(value, _np.ndarray):
                value = _np.array(value)
            if key == "shuffle":
                with open(_os.path.join(iffpath, f"{key}.dat"), "w") as f:
                    f.write(str(value))
            else:
                _sf(
                    _os.path.join(iffpath, f"{key}.fits"), value, overwrite=True
                )
    except KeyError as e:
        print(f"KeyError: {key}, {e}")
    _rif.copyConfingFile(tn)
    for param, value in zip(['modeid', 'modeamp', 'template'], [modesList, amplitude, template]):
        if value is not None:
            _rif.updateConfigFile('IFFUNC', param, value, bpath=iffpath)
    delay = _rif.getCmdDelay()
    dm.uploadCmdHistory(tch)
    dm.runCmdHistory(interf, save=tn, delay=delay)
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
