"""
Author(s):
----------
    - Pietro Ferraiuolo
    - Runa Briguglio
    
Written in June 2024

Description
-----------
Module containing all the functions necessary to process the data acquired for 
the Influence Function measurements done on M4.

High-level Functions
--------------------
process(tn, registration=False) 
    Function that processes the data contained in the OPDImages/tn folder. by p
    erforming the differential algorithm, it procudes fits images for each comm
    anded mode into the IFFunctions/tn folder, and creates a cube from these in
    to INTMatrices/tn. If 'registration is not False', upon createing the cube,
    the registration algorithm is performed.

stackCubes(tnlist)
    Function that, given as imput a tracking number list containing cubes data,
    will stack the found cubes into a new one with a new tracking number, into 
    INTMatrices/new_tn. A 'flag.txt' file will be created to give more informat
    ion on the process.

Notes
-----
In order for the module to work properly, the tower initialization must be run
so that the folder names configuration file is populated. 
From the IPython console

>>> run '/path/to/m4/initOTT.py'
>>> from m4.dmutils import iff_processing as ifp

Example
-------
>>> tn1 = '20160516_114916'
>>> tn2 = '20160516_114917' # A copy of tn1 (simulated) data
>>> ifp.process(tn1)
Cube saved in '.../m4/data/M4Data/OPTData/INTMatrices/20160516_114916/IMcube.fits'
>>> ifp.process(tn2)
Cube saved in '.../m4/data/M4Data/OPTData/INTMatrices/20160516_114917/IMcube.fits'
>>> tnlist = [tn1, tn2]
>>> ifp.stackCubes(tnlist)
Stacekd cube and matrices saved in '.../m4/data/M4Data/OPTData/INTMatrices/'new_tn'/IMcube.fits'
"""

import os as _os
import numpy as _np
import shutil as _sh
import configparser as _cp
from opticalib.core.root import _folds
from opticalib.ground import osutils as _osu
from opticalib.ground import zernike as _zern
from opticalib.core import read_config as _rif
from astropy.io.fits import Header as _header
from opticalib import typings as _ot

# from scripts.misc.IFFPackage import actuator_identification_lib as _fa

_fn = _folds()
_config = _cp.ConfigParser()
_imgFold = _fn.OPD_IMAGES_ROOT_FOLDER
_ifFold = _fn.IFFUNCTIONS_ROOT_FOLDER
_intMatFold = _fn.INTMAT_ROOT_FOLDER
_confFold = _fn.CONFIGURATION_FOLDER
_frameCenter = [200, 200]
_ts = _osu.newtn

ampVecFile = "ampVector.fits"
modesVecFile = "modesVector.fits"
templateFile = "template.fits"
regisActFile = "regActs.fits"
shuffleFile = "shuffle.dat"
indexListFile = "indexList.fits"
coordfile = ""  # TODO
cmdMatFile = "cmdMatrix.fits"
cubeFile = "IMCube.fits"
flagFile = "flag.txt"


def process(
    tn: str, register: bool = False, save: bool = False, rebin: int = 1
) -> None:
    """
    High level function with processes the data contained in the given tracking
    number OPDimages folder, performing the differential algorithm and saving
    the final cube.

    Parameters
    ----------
    tn : str
        Tracking number of the data in the OPDImages folder.
    register : bool, optional
        Parameter which enables the registration option. The default is False.
    save_and_rebin_cube : bool | int | tuple, optional
        If a bool is passed, the value is used to save the cube. If an int is
        passed, the value is used to rebin and save the cube. If a tuple is passed, the
        first value is used to save the cube, and the second to rebin it. The
        default is (False, 1).
    """
    ampVector, modesVector, template, _, registrationActs, shuffle = _getAcqPar(tn)
    _, regMat = getRegFileMatrix(tn)
    modesMat = getIffFileMatrix(tn)
    new_fold = _os.path.join(_intMatFold, tn)
    if not _os.path.exists(new_fold):
        _os.mkdir(new_fold)
    actImgList = registrationRedux(tn, regMat)
    modesMatReorg = _modesReorganization(modesMat)
    iffRedux(tn, modesMatReorg, ampVector, modesVector, template, shuffle)
    if register and not len(regMat) == 0:
        dx = findFrameOffset(tn, actImgList, registrationActs)
    else:
        dx = register
    if save:
        saveCube(tn, rebin=rebin, register=dx)


def saveCube(tn: str, rebin: int, register: bool = False) -> _np.ma.MaskedArray:
    """
    Creates and save a cube from the fits files contained in the tn folder,
    along with the command matrix and the modes vector fits.

    Parameters
    ----------
    tn : str
        Tracking number of the IFFunctions data folder from which create the cu
        be.
    rebin : int
        Rebinning factor to apply to the images before stacking them into the
        cube.
    register : int or tuple, optional
        If not False, and int or a tuple of int must be passed as value, and
        the registration algorithm is performed on the images before stacking them
        into the cube. Default is False.

    Returns
    -------
    cube : masked_array
        Data cube of the images, with shape (npx, npx, nmodes).
    """
    from opticalib.analyzer import cubeRebinner, createCube

    old_fold = _os.path.join(_ifFold, tn)
    filelist = _osu.getFileList(fold=old_fold, key="mode_")
    cube = createCube(filelist, register=register)
    # Rebinning the cube
    if rebin > 1:
        cube = cubeRebinner(cube, rebin)
    # Saving the cube
    new_fold = _os.path.join(_intMatFold, tn)
    if not _os.path.exists(new_fold):
        _os.mkdir(new_fold)
    cube_path = _os.path.join(new_fold, cubeFile)
    _osu.save_fits(cube_path, cube, overwrite=True)
    # Copying the cmdMatrix and the ModesVector into the INTMAT Folder
    cmat = _osu.load_fits(_os.path.join(_ifFold, tn, "cmdMatrix.fits"))
    mvec = _osu.load_fits(_os.path.join(_ifFold, tn, "modesVector.fits"))
    _osu.save_fits(
        _os.path.join(_intMatFold, tn, "cmdMatrix.fits"), cmat, overwrite=True
    )
    _osu.save_fits(
        _os.path.join(_intMatFold, tn, "modesVector.fits"), mvec, overwrite=True
    )
    # Creating the flag file
    with open(_os.path.join(_intMatFold, tn, flagFile), "w", encoding="utf-8") as f:
        f.write(
            f"Cube created from '{old_fold.split('/')[-1]}' data.\nRebin={rebin}\n \n"
        )
    print(f"Cube saved in '{cube_path}'")
    print(f"Shape: {cube.shape}")
    return cube


def stackCubes(tnlist: str) -> None:
    """
    Stack the cubes sontained in the corresponding tracking number folder, creating
    a new cube, along with stacked command matrix and modes vector.

    Parameters
    ----------
    tnlist : list of str
        List containing the tracking numbers of the cubes to stack.

    Returns
    -------
    stacked_cube : masked_array
        Final cube, stacked along the 3th axis.
    """
    new_tn = _ts()
    stacked_cube_fold = _os.path.join(_fn.INTMAT_ROOT_FOLDER, new_tn)
    if not _os.path.exists(stacked_cube_fold):
        _os.mkdir(stacked_cube_fold)
    cube_parameters = _getCubeList(tnlist)
    flag = _checkStackedCubes(tnlist)
    # Stacking the cube and the matrices
    stacked_cube = _np.ma.dstack(cube_parameters[0])
    stacked_cmat = _np.hstack(cube_parameters[1])
    stacked_mvec = _np.dstack(cube_parameters[2])
    # Saving everithing to a new file into a new tn
    save_cube = _os.path.join(stacked_cube_fold, cubeFile)
    save_cmat = _os.path.join(stacked_cube_fold, "cmdMatrix.fits")
    save_mvec = _os.path.join(stacked_cube_fold, "modesVector.fits")
    _osu.save_fits(save_cube, stacked_cube)
    _osu.save_fits(save_cmat, stacked_cmat)
    _osu.save_fits(save_mvec, stacked_mvec)
    with open(
        _os.path.join(stacked_cube_fold, flagFile), "w", encoding="UTF-8"
    ) as file:
        flag.write(file)
    print(f"Stacked cube and matrices saved in {new_tn}")


def filterZernikeCube(
    tn: str, zern_modes: list = None, save: bool = True
) -> _np.ma.MaskedArray:
    """
    Function which filters out the desired zernike modes from a cube.

    Parameters
    ----------
    tn : str
        Tracking number of the cube to filter.
    zern_modes : list, optional
        List of zernike modes to filter out. The default is [1,2,3]
        (piston, tip and tilt).

    Returns
    -------
    ffcube : masked array
        Filtered cube.
    """
    new_tn = _os.path.join(_intMatFold, _ts())
    _os.mkdir(new_tn)
    oldCube = _os.path.join(_intMatFold, tn, cubeFile)
    ocFlag = _os.path.join(_intMatFold, tn, flagFile)
    newCube = _os.path.join(new_tn, cubeFile)
    ocFlag = _os.path.join(_intMatFold, tn, flagFile)
    newFlag = _os.path.join(new_tn, flagFile)
    CmdMat = _os.path.join(_intMatFold, tn, cmdMatFile)
    ModesVec = _os.path.join(_intMatFold, tn, modesVecFile)
    cube = _osu.load_fits(oldCube)
    zern2filter = zern_modes if zern_modes is not None else [1, 2, 3]
    fcube = []
    for i in range(cube.shape[-1]):
        filtered = _zern.removeZernike(cube[:, :, i], zern2filter)
        fcube.append(filtered)
    ffcube = _np.ma.dstack(fcube)
    if save:
        _osu.save_fits(newCube, ffcube)
        _sh.copyfile(CmdMat, _os.path.join(new_tn, cmdMatFile))
        _sh.copyfile(ModesVec, _os.path.join(new_tn, modesVecFile))
        with open(ocFlag, "r", encoding="utf-8") as oflag:
            flag = oflag.readlines()
        flag.pop(-1)
        flag += f"Zernike modes filtered = {zern2filter}"
        with open(newFlag, "w", encoding="utf-8") as nflag:
            nflag.writelines(flag)
        print(f"Filtered cube saved at {new_tn}")
    return ffcube, new_tn.split("/")[-1]


def iffRedux(
    tn: str,
    fileMat: list[list[str]],
    ampVect: list | _np.ndarray,
    modeList: list | _np.ndarray,
    template: list | _np.ndarray,
    shuffle: int = 0,
) -> None:
    """
    Reduction function that performs the push-pull analysis on each mode, saving
    out the final processed image for each mode.<br>
    The differential algorithm for each mode is the sum over the push-pull
    realizations of the images, and it is performed as follows:

    > \sum_i \dfrac{I_i \cdot t_i - I_{i-1}\cdot t_{i-1}}{A\cdot(n-1)}

    Parameters
    ----------
    tn : str
        Tracking number of the data in the OPDImages folder.
    fileMat : ndarray
        A matrix of images in string format, in which each row is a mode and the
        columns are its template realization.
    ampVect : float | ArrayLike
        Vector containing the amplitude for each commanded mode.
    modeList : int | ArrayLike
        Vector conaining the list of commanded modes.
    template : int | ArrayLike
        Template for the push-pull command actuation.
    shuffle : int, optional
        A value different from 0 activates the shuffle option, and the imput
        value is the number of repetition for each mode's push-pull packet. The
        default is 0, which means the shuffle is OFF.
    """
    fold = _os.path.join(_ifFold, tn)
    nmodes = len(modeList)
    for i in range(0, nmodes):
        img = pushPullRedux(fileMat[i, :], template, shuffle)
        norm_img = img / (2 * ampVect[i])
        img_name = _os.path.join(fold, f"mode_{modeList[i]:04d}.fits")
        # header = _header()
        # header["MODEID"] = modeList[i]
        # header["AMP"] = (ampVect[i], 'mode amplitude')
        # header["TEMPLATE"] = str(template)
        _osu.save_fits(img_name, norm_img, overwrite=True)


def pushPullRedux(
    fileVec: list[str], template: list[int] | _np.ndarray, shuffle: int = 0
) -> _np.ma.MaskedArray:
    """
    Performs the basic operation of processing PushPull data.

    Parameters
    ----------
    fileVec : string | array
        It is a row in the fileMat (the organized matrix of the images filename),
        corresponding to all the realizations of the same mode (or act), with a
        given template. If shuffle option has been used, the fileMat (and fileVec)
        shall be reorganized before running the script.
    template: int | ArrayLike
        Template for the PushPull acquisition.
    shuffle: int, optional
        A value different from 0 activates the shuffle option, and the imput
        value is the number of repetition for each mode's templated sampling.
        The default value is 0, which means the shuffle option is OFF.

    Returns
    -------
    image: masked_array
        Final processed mode's image.
    """
    image_list = []
    template = _np.array(template)
    for i in range(0, template.shape[0]):
        ima = _osu.read_phasemap(fileVec[i])
        image_list.append(ima)
    image = _np.zeros((ima.shape[0], ima.shape[1]))
    if shuffle == 0:
        if len(template) == 1:
            image = image_list[0] * template[0]
            master_mask = image_list[0].mask
        else:
            for x in range(1, len(image_list)):
                opd2add = (
                    image_list[x] * template[x] + image_list[x - 1] * template[x - 1]
                )
                master_mask2add = _np.ma.mask_or(
                    image_list[x].mask, image_list[x - 1].mask
                )
                if x == 1:
                    master_mask = master_mask2add
                else:
                    master_mask = _np.ma.mask_or(master_mask, master_mask2add)
                image += opd2add
    else:
        print("Shuffle option")
        for i in range(0, shuffle - 1):
            for x in range(1, 2):
                opd2add = (
                    image_list[i * 3 + x] * template[x]
                    + image_list[i * 3 + x - 1] * template[x - 1]
                )
                master_mask2add = _np.ma.mask_or(
                    image_list[i * 3 + x].mask, image_list[i * 3 + x - 1].mask
                )
                if i == 0 and x == 1:
                    master_mask = master_mask2add
                else:
                    master_mask = _np.na.mask_or(master_mask, master_mask2add)
                image += opd2add
    image = _np.ma.masked_array(image, mask=master_mask) / _np.max(
        ((template.shape[0] - 1), 1)
    )  #!!!
    return image


def registrationRedux(tn: str, fileMat: list[str]) -> list[_np.ma.MaskedArray]:
    """
    Reduction function that performs the push-pull analysis on the registration
    data.

    Parameters
    ----------
    fileMat : ndarray
        A matrix of images in string format, in which each row is a mode and the
        columns are its template realization.

    Returns
    -------
    imgList : ArrayLike
        List of the processed registration images.
    """
    _, infoR, _ = _getAcqInfo(tn)
    template = infoR["template"]
    if _np.array_equal(fileMat, _np.array([])) and len(infoR["modesid"]) == 0:
        print("No registration data found")
        return []
    nActs = fileMat.shape[0]
    imglist = []
    for i in range(0, nActs - 1):
        img = pushPullRedux(fileMat[i, :], template)
        imglist.append(img)
    cube = _np.ma.masked_array(imglist)
    # _osu.save_fits(_os.path.join(_intMatFold, tn, "regActCube.fits"), cube)
    return imglist


def findFrameOffset(
    tn: str, imglist: list[_np.ma.MaskedArray], actlist: int | _np.ndarray
) -> float:
    """
    This function computes the position difference between the current frame and
    a reference one.

    Parameters
    ----------
    tn : str
        Tracking number
    imglist : list | masked arrays
        List of the actuator images to be used
    actlist: int | array
        List of actuators (index)

    Returns
    -------
    dp: float
        Position difference
    """
    actCoordFile = _os.path.join(_ifFold, tn, coordfile)
    actCoord = _osu.load_fits(actCoordFile)
    xy = _fa.findFrameCoord(imglist, actlist, actCoord)
    dp = xy - _frameCenter
    return dp


def getTriggerFrame(tn: str, amplitude: int | float = None) -> int:
    """
    Analyze the tracking number's images list and search for the trigger frame.

    Parameters
    ----------
    tn : str
        Tracking number of the data in the OPDImages folder.
    amplitude : int or float, optional
        Amplitude of the commanded trigger mode, which serves as the check value
        for finding the frame. If no value is passed it is loaded from the iffConfig.ini
        file.

    Returns
    -------
    trigFrame : int
        Index which identifies the trigger frame in the images folder file list.

    Raises
    ------
    RuntimeError
        Error raised if the file iteration goes beyon the expected trigger frame
        wich can be inferred through the number of trigger zeros in the iffConfig.ini
        file.
    """
    infoT, _, _ = _getAcqInfo(tn)
    if amplitude is not None:
        infoT["amplitude"] = amplitude
    fileList = _osu.getFileList(tn)
    img0 = _osu.read_phasemap(fileList[0])
    go = i = 1
    # add the condition where if there are not trigger frames the code is skipped and the
    # the rest is handled with care
    if infoT["zeros"] == 0 and len(infoT["modes"]) == 0:
        trigFrame = 0
        return trigFrame
    while go != 0:
        thresh = infoT["amplitude"] / 3
        img1 = _osu.read_phasemap(fileList[i])
        rr2check = _zern.removeZernike(img1 - img0, [1, 2, 3]).std()
        if go > infoT["zeros"]:
            raise RuntimeError(
                f"Frame {go}. Heading Zeros exceeded: std= {rr2check:.2e} < {thresh:.2e} =Amp/3"
            )
        if rr2check > thresh:
            go = 0
        else:
            i += 1
            go += 1
            img0 = img1
    trigFrame = i
    return trigFrame


def getRegFileMatrix(tn: str) -> tuple[int, _np.ndarray]:
    """
    Search for the registration frames in the images file list, and creates the
    registration file matrix.

    Parameters
    ----------
    tn : str
        Tracking number of the data in the OPDImages folder.

    Returns
    -------
    regEnd : int
        Index which identifies the last registration frame in the images file
        list.
    regMat : ndarray
        A matrix of images in string format, containing the registration frames.
        It has shape (registration_modes, n_push_pull).
    """
    fileList = _osu.getFileList(tn)
    _, infoR, _ = _getAcqInfo(tn)
    timing = _rif.getTiming()
    trigFrame = getTriggerFrame(tn)
    if infoR["zeros"] == 0 and len(infoR["modes"]) == 0:
        regStart = regEnd = (trigFrame + 1) if trigFrame != 0 else 0
    else:
        regStart = trigFrame + infoR["zeros"] * timing + (1 if trigFrame != 0 else 0)
        regEnd = regStart + len(infoR["modes"]) * len(infoR["template"]) * timing
    regList = fileList[regStart:regEnd]
    regMat = _np.reshape(regList, (len(infoR["modes"]), len(infoR["template"])))
    return regEnd, regMat


def getIffFileMatrix(tn: str) -> _np.ndarray:
    """
    Creates the iffMat

    Parameters
    ----------
    tn : str
        Tracking number of the data in the OPDImages folder.

    Returns
    -------
    iffMat : ndarray
        A matrix of images in string format, conatining all the images for the
        IFF acquisition, that is all the modes with each push-pull realization.
        It has shape (modes, n_push_pull)
    """
    fileList = _osu.getFileList(tn)
    _, _, infoIF = _getAcqInfo(tn)
    regEnd, _ = getRegFileMatrix(tn)
    iffList = fileList[regEnd + infoIF["zeros"] :]
    iffMat = _np.reshape(iffList, (len(infoIF["modes"]), len(infoIF["template"])))
    return iffMat


def _getCubeList(
    tnlist: str,
) -> tuple[list[_np.ma.MaskedArray], list[_np.ndarray], list[_np.ndarray], int]:
    """
    Retireves the cubes from each tn in the tnlist.

    Parameters
    ----------
    tnlist : list of str
        List containing the tracking number of the cubes to stack.

    Returns
    -------
    cubeList : list of masked_array
        List containing the cubes to stack.
    matrixList : list of ndarray
        List containing the command matrices for each cube.
    modesVectList : list of ndarray
        List containing the modes vectors for each cube.
    """
    cubeList = []
    matrixList = []
    modesVectList = []
    rebins = []
    for tn in tnlist:
        fold = _os.path.join(_intMatFold, tn)
        cube_name = _os.path.join(fold, "IMCube.fits")
        matrix_name = _os.path.join(fold, "cmdMatrix.fits")
        modesVec_name = _os.path.join(fold, "modesVector.fits")
        flag_file = _os.path.join(fold, "flag.txt")
        cubeList.append(_osu.load_fits(cube_name))
        matrixList.append(_osu.load_fits(matrix_name))
        modesVectList.append(_osu.load_fits(modesVec_name))
        with open(flag_file, "r", encoding="UTF-8") as f:
            flag = f.readlines()
        rebins.append(int(flag[1].split("=")[1].strip()))
    if not all([rebin == rebins[0] for rebin in rebins]):
        raise ValueError("Cubes have different rebinning factors")
    rebin = rebins[0]
    return cubeList, matrixList, modesVectList, rebin


def _getAcqPar(
    tn: str,
) -> tuple[
    _np.ndarray[float],
    _np.ndarray[int],
    _np.ndarray[int],
    _np.ndarray[int],
    _np.ndarray[int],
    int,
]:
    """
    Reads ad returns the acquisition parameters from fits files.

    Parameters
    ----------
    tn : str
        Tracking number of the data in the OPDImages folder.

    Returns
    -------
    ampVector : float | ArrayLike
        Vector containg the amplitude of each commanded mode.
    modesVector : int | ArrayLike
        Vector containing the list of commanded modes.
    template : int | ArrayLike
        Sampling template ampplied on each mode.
    indexList : int | ArrayLike
        Indexing of the modes inside the commanded matrix.
    registrationActs : int | ArrayLike
        Vector containing the commanded actuators for the registration.
    shuffle : int
        Shuffle information. If it's nor 0, the values indicates the number of
        template sampling repetition for each mode.
    """
    base = _os.path.join(_ifFold, tn)
    ampVector = _osu.load_fits(_os.path.join(base, ampVecFile))
    template = _osu.load_fits(_os.path.join(base, templateFile))
    modesVector = _osu.load_fits(_os.path.join(base, modesVecFile))
    indexList = _osu.load_fits(_os.path.join(base, indexListFile))
    registrationActs = _osu.load_fits(_os.path.join(base, regisActFile))
    with open(_os.path.join(base, shuffleFile), "r", encoding="UTF-8") as shf:
        shuffle = int(shf.read())
    return ampVector, modesVector, template, indexList, registrationActs, shuffle


def _getAcqInfo(tn: str = None) -> tuple[dict, dict, dict]:
    """
    Returns the information read from the iffConfig.ini file.

    Parameters
    ----------
    tn : str, optional
        Tracking number of the data in the IFFunctions folder. The default is None,
        which points to configuration root folder.

    Returns
    -------
    infoT : dict
        Information read about the TRIGGER options.
    infoR : dict
        Information read about the REGISTRATION options.
    infoIF : dict
        Information read about the IFFUNC option.
    """
    path = _os.path.join(_ifFold, tn) if tn is not None else _fn.CONFIGURATION_FOLDER
    infoT = _rif.getIffConfig("TRIGGER", bpath=path)
    infoR = _rif.getIffConfig("REGISTRATION", bpath=path)
    infoIF = _rif.getIffConfig("IFFUNC", bpath=path)
    return infoT, infoR, infoIF


def _checkStackedCubes(tnlist: str) -> dict:
    """
    Inspect the cubes to stack, to check whether there are shared modes, or not.

    Parameters
    ----------
    tnlist : list of str
        List containing the tracking number of the cubes to stack.

    Returns
    -------
    flag : dict
        Dictionary containing the flagging information about the stacked cube,
        to be later dump into the 'flag.txt' file.
    """
    _, _, modesVectList, rebin = _getCubeList(tnlist)
    nmodes = len(modesVectList[0])
    nvects = len(modesVectList)
    for i in range(nvects):
        for j in range(i + 1, nvects):
            common_modes = set(modesVectList[i]).intersection(modesVectList[j])
    c_nmodes = len(common_modes)
    if c_nmodes in range(1, nmodes):
        flag = __flag(tnlist, modesVectList, rebin, 2)
    elif c_nmodes == nmodes:
        flag = __flag(tnlist, modesVectList, rebin, 1)
    else:
        flag = __flag(tnlist, modesVectList, rebin, 0)
    return flag


def __flag(
    tnlist: list[str],
    modesVectList: list[int] | _np.ndarray[int],
    rebin: int,
    type: int,
) -> dict:
    """
    Creates the dictionary to dump into the 'flag.txt' file accordingly to
    sequentially stacked cubes with no repeated modes.

    Parameters
    ----------
    tnlist : list of str
        List containing the tracking number of the cubes to stack.
    modesVectList : list of ndarray
        A list containing the modes vectors for each cube.
    type : int
        Type of stacked cube created.
        0 for sequential, 1 for mean, 2 for shared modes.

    Returns
    -------
    config : dict
        Dictionary containing the flagging information about the stacked cube.
    """
    c_type = [
        "Sequentially stacked cubes",
        "Mean of cubes",
        "!!!Warning: repeated modes in stacked cube",
    ]
    text = ""
    for i, tn in enumerate(tnlist):
        if _np.array_equal(
            modesVectList[i],
            _np.arange(modesVectList[i][0], modesVectList[i][-1] + 1, 1),
        ):
            text += f"""
{tn}, modes {modesVectList[i][0]} to {modesVectList[i][-1]}"""
        else:
            text += f"""
{tn}, modes {list(modesVectList[i])}"""
    flag = {
        "Flag": {
            "Rebin": str(rebin),
            "Cube type": c_type[type],
            "Source cubes": text,
        }
    }
    _config["Flag"] = {}
    for key, value in flag["Flag"].items():
        _config["Flag"][key] = value
    return _config


# TODO
def _ampReorganization(ampVector):
    reorganizaed_amps = ampVector
    return reorganizaed_amps


def _modesReorganization(modesVector):
    reorganizaed_modes = modesVector
    return reorganizaed_modes
