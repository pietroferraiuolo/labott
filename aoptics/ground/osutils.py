"""
Author(s)
---------
- Chiara Selmi:  written in 2019
- Pietro Ferraiuolo: updated in 2025

"""

import os as _os
import numpy as _np
import time as _time
import h5py as _h5py
from numpy import uint8 as _uint8
from aoptics.core import root as _fn
from astropy.io import fits as _fits
from numpy.ma import masked_array as _masked_array

_OPTDATA = _fn.OPT_DATA_ROOT_FOLDER
_OPDIMG = _fn.OPD_IMAGES_ROOT_FOLDER


def findTracknum(tn: str, complete_path: bool = False) -> str | list[str]:
    """
    Search for the tracking number given in input within all the data path subfolders.

    Parameters
    ----------
    tn : str
        Tracking number to be searched.
    complete_path : bool, optional
        Option for wheter to return the list of full paths to the folders which
        contain the tracking number or only their names.

    Returns
    -------
    tn_path : list of str
        List containing all the folders (within the OPTData path) in which the
        tracking number is present, sorted in alphabetical order.
    """
    tn_path = []
    for fold in _os.listdir(_OPTDATA):
        search_fold = _os.path.join(_OPTDATA, fold)
        if not _os.path.isdir(search_fold):
            continue
        if tn in _os.listdir(search_fold):
            if complete_path:
                tn_path.append(_os.path.join(search_fold, tn))
            else:
                tn_path.append(fold)
    path_list = sorted(tn_path)
    if len(path_list) == 1:
        path_list = path_list[0]
    return path_list


def getFileList(tn: str = None, fold: str = None, key: str = None) -> list[str]:
    """
    Search for files in a given tracking number or complete path, sorts them and
    puts them into a list.

    Parameters
    ----------
    tn : str
        Tracking number of the data in the OPDImages folder.
    fold : str, optional
        Folder in which searching for the tracking number. If None, the default
        folder is the OPD_IMAGES_ROOT_FOLDER.
    key : str, optional
        A key which identify specific files to return

    Returns
    -------
    fl : list of str
        List of sorted files inside the folder.

    How to Use it
    -------------
    If the complete path for the files to retrieve is available, then this function
    should be called with the 'fold' argument set with the path, while 'tn' is
    defaulted to None.

    In any other case, the tn must be given: it will search for the tracking
    number into the OPDImages folder, but if the search has to point another
    folder, then the fold argument comes into play again. By passing both the
    tn (with a tracking number) and the fold argument (with only the name of the
    folder) then the search for files will be done for the tn found in the
    specified folder. Hereafter there is an example, with the correct use of the
    key argument too.

    Examples
    --------

    Here are some examples regarding the use of the 'key' argument. Let's say w
    e need a list of files inside ''tn = '20160516_114916' '' in the IFFunctions
    folder.

        >>> iffold = 'IFFunctions'
        >>> tn = '20160516_114916'
        >>> getFileList(tn, fold=iffold)
        ['.../M4/m4/data/M4Data/OPTData/IFFunctions/20160516_114916/cmdMatrix.fits',
         '.../M4/m4/data/M4Data/OPTData/IFFunctions/20160516_114916/mode_0000.fits',
         '.../M4/m4/data/M4Data/OPTData/IFFunctions/20160516_114916/mode_0001.fits',
         '.../M4/m4/data/M4Data/OPTData/IFFunctions/20160516_114916/mode_0002.fits',
         '.../M4/m4/data/M4Data/OPTData/IFFunctions/20160516_114916/mode_0003.fits',
         '.../M4/m4/data/M4Data/OPTData/IFFunctions/20160516_114916/modesVector.fits']

    Let's suppose we want only the list of 'mode_000x.fits' files:

        >>> getFileList(tn, fold=iffold, key='mode_')
        ['.../M4/m4/data/M4Data/OPTData/IFFunctions/20160516_114916/mode_0000.fits',
         '.../M4/m4/data/M4Data/OPTData/IFFunctions/20160516_114916/mode_0001.fits',
         '.../M4/m4/data/M4Data/OPTData/IFFunctions/20160516_114916/mode_0002.fits',
         '.../M4/m4/data/M4Data/OPTData/IFFunctions/20160516_114916/mode_0003.fits']

    Notice that, in this specific case, it was necessary to include the undersc
    ore after 'mode' to exclude the 'modesVector.fits' file from the list.
    """
    if tn is None and fold is not None:
        fl = sorted([_os.path.join(fold, file) for file in _os.listdir(fold)])
    else:
        try:
            paths = findTracknum(tn, complete_path=True)
            if isinstance(paths, str):
                paths = [paths]
            for path in paths:
                if fold is None:
                    fl = []
                    fl.append(
                        sorted(
                            [_os.path.join(path, file) for file in _os.listdir(path)]
                        )
                    )
                elif fold in path.split("/")[-2]:
                    fl = sorted(
                        [_os.path.join(path, file) for file in _os.listdir(path)]
                    )
                else:
                    raise Exception
        except Exception as exc:
            raise FileNotFoundError(
                f"Invalid Path: no data found for tn '{tn}'"
            ) from exc
    if len(fl) == 1:
        fl = fl[0]
    if key is not None:
        try:
            selected_list = []
            for file in fl:
                if key in file.split("/")[-1]:
                    selected_list.append(file)
        except TypeError as err:
            raise TypeError("'key' argument must be a string") from err
        fl = selected_list
    if len(fl) == 1:
        fl = fl[0]
    return fl


def tnRange(tn0: str, tn1: str) -> list[str]:
    """
    Returns the list of tracking numbers between tn0 and tn1, within the same
    folder, if they both exist in it.

    Parameters
    ----------
    tn0 : str
        Starting tracking number.
    tn1 : str
        Finish tracking number.

    Returns
    -------
    tnMat : list of str
        A list or a matrix of tracking number in between the start and finish ones.

    Raises
    ------
    FileNotFoundError
        An exception is raised if the two tracking numbers are not found in the same folder
    """
    tn0_fold = findTracknum(tn0)
    tn1_fold = findTracknum(tn1)
    if len(tn0_fold) == 1 and len(tn1_fold) == 1:
        if tn0_fold[0] == tn1_fold[0]:
            fold_path = _os.path.join(_OPTDATA, tn0_fold[0])
            tn_folds = sorted(_os.listdir(fold_path))
            id0 = tn_folds.index(tn0)
            id1 = tn_folds.index(tn1)
            tnMat = [_os.path.join(fold_path, tn) for tn in tn_folds[id0 : id1 + 1]]
        else:
            raise FileNotFoundError("The tracking numbers are in different foldes")
    else:
        tnMat = []
        for ff in tn0_fold:
            if ff in tn1_fold:
                fold_path = _os.path.join(_OPTDATA, ff)
                tn_folds = sorted(_os.listdir(fold_path))
                id0 = tn_folds.index(tn0)
                id1 = tn_folds.index(tn1)
                tnMat.append(
                    [_os.path.join(fold_path, tn) for tn in tn_folds[id0 : id1 + 1]]
                )
    return tnMat


def read_phasemap(file_path: str) -> _np.ma.MaskedArray:
    """
    Function to read interferometric data, in the three possible formats
    (FITS, 4D, H5)

    Parameters
    ----------
    file_path: str
        Complete filepath of the file to load.

    Returns
    -------
    image: numpy masked array
        Image as a masked array.
    """
    ext = file_path.split(".")[-1]
    if ext == "fits":
        image = load_fits(file_path)
    elif ext == "4D":
        image = _InterferometerConverter.fromPhaseCam6110(file_path)
    elif ext == "4Ds":
        image = load_fits(file_path)
    elif ext == "h5":
        image = _InterferometerConverter.fromPhaseCam4020(file_path)
    return image


def load_fits(filepath: str) -> _np.ndarray | _np.ma.MaskedArray:
    """
    Loads a FITS file.

    Parameters
    ----------
    filepath : str
        Path to the FITS file.

    Returns
    -------
    fit : np.ndarray or np.ma.MaskedArray
        FITS file data.
    """
    with _fits.open(filepath) as hdul:
        fit = hdul[0].data
        if len(hdul) > 1 and hasattr(hdul[1], "data"):
            mask = hdul[1].data.astype(bool)
            fit = _masked_array(fit, mask=mask)
    return fit


def save_fits(
    filepath: str,
    data: _np.ndarray | _np.ma.MaskedArray,
    overwrite: bool = True,
    header: dict[str, any] | _fits.Header = None,
) -> None:
    """
    Saves a FITS file.

    Parameters
    ----------
    filepath : str
        Path to the FITS file.
    data : np.array
        Data to be saved.
    overwrite : bool, optional
        Whether to overwrite an existing file. Default is True.
    header : dict[str, any] | fits.Header, optional
        Header information to include in the FITS file. Can be a dictionary or
        a fits.Header object.
    """
    # Prepare the header
    if header is not None:
        if isinstance(header, dict):
            header = _fits.Header(header)
        elif not isinstance(header, _fits.Header):
            raise TypeError(
                "'header' must be a dictionary or an astropy.io.fits.Header object"
            )
    # Save the FITS file
    if isinstance(data, _masked_array):
        _fits.writeto(filepath, data.data, header=header, overwrite=overwrite)
        if hasattr(data, "mask"):
            _fits.append(filepath, data.mask.astype(_uint8))
    else:
        _fits.writeto(filepath, data, header=header, overwrite=overwrite)


def newtn() -> str:
    """
    Returns a timestamp in a string of the format `YYYYMMDD_HHMMSS`.

    Returns
    -------
    str
        Current time in a string format.
    """
    return _time.strftime("%Y%m%d_%H%M%S")


def rename4D(folder: str) -> None:
    """
    Renames the produced 'x.4D' files into '0000x.4D'

    Parameters
    ----------
    folder : str
        The folder where the 4D data is stored.
    """
    fold = _os.path.join(_OPDIMG, folder)
    files = _os.listdir(fold)
    for file in files:
        if file.endswith(".4D"):
            num_str = file.split(".")[0]
            if num_str.isdigit():
                num = int(num_str)
                new_name = f"{num:05d}.4D"
                old_file = _os.path.join(fold, file)
                new_file = _os.path.join(fold, new_name)
                _os.rename(old_file, new_file)


def getCameraSettings(tn: str) -> list[int]:
    """
    Reads the interferometer settings from a given configuration file.

    Return
    ------
    output: list of int
        List of camera settings:
        [width_pixel, height_pixel, offset_x, offset_y]
    """
    path = findTracknum(tn, complete_path=True)
    try:
        file_path = _os.path.join(path, _fn.COPIED_SETTINGS_CONF_FILE)
        setting_reader = _fn.ConfSettingReader4D(file_path)
    except Exception as e:
        print(f"Error: {e}")
        file_path = _os.path.join(path, "4DSettings.ini")
        setting_reader = _fn.ConfSettingReader4D(file_path)
    width_pixel = setting_reader.getImageWidhtInPixels()
    height_pixel = setting_reader.getImageHeightInPixels()
    offset_x = setting_reader.getOffsetX()
    offset_y = setting_reader.getOffsetY()
    return [width_pixel, height_pixel, offset_x, offset_y]


def getFrameRate(tn: str) -> float:
    """
    Reads the frame rate of the camera from a given configuration file.

    Return
    ------
    frame_rate: float
        Frame rate of the interferometer
    """
    path = findTracknum(tn, complete_path=True)
    try:
        file_path = _os.path.join(path, _fn.COPIED_SETTINGS_CONF_FILE)
        setting_reader = _fn.ConfSettingReader4D(file_path)
    except Exception as e:
        print(f"Error: {e}")
        file_path = _os.path.join(path, "4DSettings.ini")
        setting_reader = _fn.ConfSettingReader4D(file_path)
    frame_rate = setting_reader.getFrameRate()
    return frame_rate


class _InterferometerConverter:
    """
    This class is crucial to convert H5 files into masked array
    """

    @staticmethod
    def fromPhaseCam4020(h5filename):
        """
        Function for PhaseCam4020

        Parameters
        ----------
        h5filename: string
            Path of the h5 file to convert

        Returns
        -------
        ima: numpy masked array
            Masked array image
        """
        file = _h5py.File(h5filename, "r")
        genraw = file["measurement0"]["genraw"]["data"]
        data = _np.array(genraw)
        mask = _np.zeros(data.shape, dtype=bool)
        mask[_np.where(data == data.max())] = True
        ima = _np.ma.masked_array(data * 632.8e-9, mask=mask)
        return ima

    @staticmethod
    def fromPhaseCam6110(i4dfilename):
        """
        Function for PhaseCam6110

        Parameters
        ----------
        i4dfilename: string
            Path of the 4D file to convert

        Returns
        -------
        ima: numpy masked array
            Masked array image
        """
        with _h5py.File(i4dfilename, "r") as ff:
            data = ff.get("/Measurement/SurfaceInWaves/Data")
            meas = data[()]
            mask = _np.invert(_np.isfinite(meas))
        image = _np.ma.masked_array(meas * 632.8e-9, mask=mask)
        return image

    @staticmethod
    def fromFakeInterf(filename):
        """
        Function for fake interferometer

        Parameters
        ----------
        filename: string
            Path name for data

        Returns
        -------
        ima: numpy masked array
            Masked array image
        """
        masked_ima = load_fits(filename)
        return masked_ima

    @staticmethod
    def fromI4DToSimplerData(i4dname, folder, h5name):
        """
        Function for converting files from 4D 6110 files to H5 files

        Parameters
        ----------
        i4dname: string
            File name path of 4D data
        folder: string
            Folder path for new data
        h5name: string
            Name for H5 data

        Returns
        -------
        file_name: string
            Final path name
        """
        file = _h5py.File(i4dname, "r")
        data = file.get("/Measurement/SurfaceInWaves/Data")
        file_name = _os.path.join(folder, h5name)
        hf = _h5py.File(file_name, "w")
        hf.create_dataset("Data", data=data)
        return file_name
