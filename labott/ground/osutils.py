"""
Author(s)
---------
- Chiara Selmi:  written in 2019
- Pietro Ferraiuolo: updated in 2025

"""

import os
import h5py
import time
import numpy as np
from numpy import uint8
from astropy.io import fits
from numpy.ma import masked_array
from gentle.core import root as _fn

OPTDATA = _fn.BASE_DATA_PATH
OPDIMG = _fn.OPD_IMAGES_ROOT_FOLDER


def findTracknum(tn, complete_path: bool = False):
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
    for fold in os.listdir(OPTDATA):
        search_fold = os.path.join(OPTDATA, fold)
        if not os.path.isdir(search_fold):
            continue
        if tn in os.listdir(search_fold):
            if complete_path:
                tn_path.append(os.path.join(search_fold, tn))
            else:
                tn_path.append(fold)
    path_list = sorted(tn_path)
    if len(path_list) == 1:
        path_list = path_list[0]
    return path_list


def getFileList(tn=None, fold=None, key: str = None):
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
    fl : list os str
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
        fl = sorted([os.path.join(fold, file) for file in os.listdir(fold)])
    else:
        try:
            paths = findTracknum(tn, complete_path=True)
            if isinstance(paths, str):
                paths = [paths]
            for path in paths:
                if fold is None:
                    fl = []
                    fl.append(
                        sorted([os.path.join(path, file) for file in os.listdir(path)])
                    )
                elif fold in path.split("/")[-2]:
                    fl = sorted([os.path.join(path, file) for file in os.listdir(path)])
                else:
                    raise Exception
        except Exception as exc:
            raise FileNotFoundError(
                f"Invalid Path: no data found for tn '{tn}'"
            ) from exc
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


def tnRange(tn0, tn1):
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
        A list or a matrix of tracking number in between the start and finish
        ones.

    Raises
    ------
    Exception
        An exception is raised if the two tracking numbers are not found in the
        same folder
    """
    tn0_fold = findTracknum(tn0)
    tn1_fold = findTracknum(tn1)
    if len(tn0_fold) == 1 and len(tn1_fold) == 1:
        if tn0_fold[0] == tn1_fold[0]:
            fold = os.path.join(OPTDATA, tn0_fold[0])
            tn_folds = sorted(os.listdir(fold))
            id0 = tn_folds.index(tn0)
            id1 = tn_folds.index(tn1)
            tnMat = [os.path.join(fold, tn) for tn in tn_folds[id0 : id1 + 1]]
        else:
            raise FileNotFoundError("The tracking numbers are in different foldes")
    else:
        tnMat = []
        for ff in tn0_fold:
            if ff in tn1_fold:
                fold = os.path.join(OPTDATA, ff)
                tn_folds = sorted(os.listdir(fold))
                id0 = tn_folds.index(tn0)
                id1 = tn_folds.index(tn1)
                tnMat.append([os.path.join(fold, tn) for tn in tn_folds[id0 : id1 + 1]])
    return tnMat


def read_phasemap(file_path):
    """
    Function to read interferometric data, in the trhee possible formats
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
        image = InterferometerConverter.fromPhaseCam6110(file_path)
    elif ext == "4Ds":
        image = load_fits(file_path)
    elif ext == "h5":
        image = InterferometerConverter.fromPhaseCam4020(file_path)
    return image


def load_fits(filepath):
    """
    Loads a FITS file.

    Parameters
    ----------
    filepath : str
        Path to the FITS file.

    Returns
    -------
    np.array
        FITS file data.
    """
    with fits.open(filepath) as hdul:
        fit = hdul[0].data
        if len(hdul) > 1 and hasattr(hdul[1], "data"):
            mask = hdul[1].data.astype(bool)
            fit = masked_array(fit, mask=mask)
    return fit


def save_fits(filepath, data):
    """
    Saves a FITS file.

    Parameters
    ----------
    filepath : str
        Path to the FITS file.

    data : np.array
        Data to be saved.
    """
    if isinstance(data, masked_array):
        fits.writeto(filepath, data.data, overwrite=True)
        if hasattr(data, "mask"):
            fits.append(filepath, data.mask.astype(uint8))
    else:
        fits.writeto(filepath, data, overwrite=True)


def newtn():
    """
    Returns a timestamp in a string of the format `YYYYMMDD_HHMMSS`.

    Returns
    -------
    str
        Current time in a string format.
    """
    return time.strftime("%Y%m%d_%H%M%S")


def rename4D(folder):
    """
    Renames the produced 'x.4D' files into '0000x.4D'

    Parameters
    ----------
    folder : str
        The folder where the 4D data is stored.
    """
    fold = os.path.join(OPDIMG, folder)
    files = os.listdir(fold)
    for file in files:
        if file.endswith(".4D"):
            num_str = file.split(".")[0]
            if num_str.isdigit():
                num = int(num_str)
                new_name = f"{num:05d}.4D"
                old_file = os.path.join(fold, file)
                new_file = os.path.join(fold, new_name)
                os.rename(old_file, new_file)


class InterferometerConverter:
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
                 path of h5 file to convert

        Returns
        -------
                ima: numpy masked array
                     masked array image
        """
        file = h5py.File(h5filename, "r")
        genraw = file["measurement0"]["genraw"]["data"]
        data = np.array(genraw)
        mask = np.zeros(data.shape, dtype=bool)
        mask[np.where(data == data.max())] = True
        ima = np.ma.masked_array(data * 632.8e-9, mask=mask)
        return ima

    @staticmethod
    def fromPhaseCam6110(i4dfilename):
        """
        Function for PhaseCam6110
        Parameters
        ----------
            h5filename: string
                 path of h5 file to convert

        Returns
        -------
                ima: numpy masked array
                     masked array image
        """
        with h5py.File(i4dfilename, "r") as ff:
            data = ff.get("/Measurement/SurfaceInWaves/Data")
            meas = data[()]
            mask = np.invert(np.isfinite(meas))

        image = np.ma.masked_array(meas * 632.8e-9, mask=mask)

        return image

    @staticmethod
    def fromFakeInterf(filename):
        """
        Function for fake interferometer
        Parameters
        ----------
            file: string
                 path name for data

        Returns
        -------
                ima: numpy masked array
                     masked array image
        """
        masked_ima = load_fits(filename)
        return masked_ima

    @staticmethod
    def fromI4DToSimplerData(i4dname, folder, h5name):
        """Function for converting files from 4d 6110 files to H5 files
        Parameters
        ----------
        i4dname: string
            file name path of 4d data
        folder: string
            folder path for new data
        h5name: string
            name for h5 data

        Returns
        -------
        file_name: string
            finale path name
        """
        file = h5py.File(i4dname, "r")
        data = file.get("/Measurement/SurfaceInWaves/Data")

        file_name = os.path.join(folder, h5name)
        hf = h5py.File(file_name, "w")
        hf.create_dataset("Data", data=data)
        return file_name
