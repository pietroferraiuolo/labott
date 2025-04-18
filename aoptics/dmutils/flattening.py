"""
Author(s) 
---------
    - Pietro Ferraiuolo : written in 2024

Description
-----------
Module containing the class which computes the flattening command for a deformable
mirror, given an imput shape and a (filtered) interaction cube.

From the loaded tracking number (tn) the interaction cube will be loaded (and
filtered, if it's not already) from which the interaction matrix will be computed.
If an image to shape is provided on class instance, then the reconstructor will
be automatically computed, while if not, the load_img2shape methos is available
to upload a shape from which compute the reconstructor.

How to Use it
=============
Instancing the class only with the tn of the interaction cube

    >>> from m4.dmutils import flattening as flt
    >>> tn = '20240906_110000' # example tn
    >>> f = flt.Flattening(tn)
    >>> # say we have acquired an image
    >>> img = interf.acquire_phasemap()
    >>> f.load_image2shape(img)
    >>> f.computeRecMat()
    'Computing reconstruction matrix...'

all is ready to compute the flat command, by simply running the method

    >>> flatCmd = flat.computeFlatCmd()

Update : all the steps above have been wrapped into the `applyFlatCommand` method,
which will also save the flat command and the images used for the computation in a
dedicated folder in the flat root folder.

"""

import os as _os
import numpy as _np
from . import iff_processing as _ifp
from aoptics.ground import osutils as _osu
from aoptics.core.root import folders as _fn
from aoptics.ground import computerec as _crec

_ts = _osu.newtn

class Flattening:
    """
    Class which handles the flattening command computation

    Public Methods
    -------
    computeFlatCmd :
        Method which computes the flattening command to apply to a given shape,
        which must be already in memory, through the class instancing or the
        load_img2shape method

    load_image2shape :
        method to (re)upload and image to shape in the class, after which the
        reconstructor will be automatically computed for it.
    """

    def __init__(self, tn):
        """The Constructor"""
        self.tn                 = tn
        self.shape2flat         = None
        self.flatCmd            = None
        self.rebin              = None
        self.filtered           = False
        self._path              = _os.path.join(_ifp._intMatFold, self.tn)
        self._oldtn             = tn
        self._intCube           = self._loadIntCube()
        self._cmdMat            = self._loadCmdMat()
        self._rec               = self._loadReconstructor()
        self._recMat            = None
        self._frameCenter       = None
        self._flatOffset        = None
        self._cavityOffset      = None
        self._synthFlat         = None
        self._flatResidue       = None
        self._flatteningModes   = None

    def applyFlatCommand(
        self, dm, interf, modes2flat, nframes: int = 5, modes2discard=None
    ):
        f"""
        Computes, applies and saves the computed flat command to the DM, given
        the {self.tn} calibration.

        Parameters
        ----------
        dm : object
            Deformable mirror object.
        interf : object
            Interferometer object to acquire phasemaps.
        modes2flat : int or list
            Modes to flatten.
        nframes : int, optional
            Number of frames to average for phasemap acquisition. Default is 5.
        modes2discard : int, optional
            Number of modes to discard when computing the reconstruction matrix. Default is 3.
        """
        new_tn = _ts()
        imgstart = interf.acquire_map(nframes, rebin=self.rebin)
        self.loadImage2Shape(imgstart)
        self.computeRecMat(modes2discard)
        deltacmd = self.computeFlatCmd(modes2flat)
        cmd = dm.get_shape()
        dm.set_shape(deltacmd, differential = True)
        imgflat = interf.acquire_map(nframes, rebin=self.rebin)
        files = [
            "flatCommand.fits",
            "flatDeltaCommand.fits",
            "imgstart.fits",
            "imgflat.fits",
        ]
        data = [cmd, deltacmd, imgstart, imgflat]
        fold = _os.path.join(_fn.FLAT_ROOT_FOLDER, new_tn)
        if not _os.path.exists(fold):
            _os.mkdir(fold)
        for f, d in zip(files, data):
            path = _os.path.join(fold, f)
            if isinstance(d, _np.ma.masked_array):
                _osu.save_fits(path, d)
            else:
                _osu.save_fits(path, d)
        with open(_os.path.join(fold, "info.txt"), "w") as info:
            info.write(f"Flattened with `{self.tn}` data")
        print(f"Flat command saved in {'/'.join(fold.split('/')[-2:])}")

    def computeFlatCmd(self, n_modes):
        """
        Compute the command to apply to flatten the input shape.

        Returns
        -------
        flat_cmd : ndarray
            Flat command.
        """
        img = _np.ma.masked_array(self.shape2flat, mask=self._getMasterMask())
        _cmd = -_np.dot(img.compressed(), self._recMat)
        if isinstance(n_modes, int):
            flat_cmd = self._cmdMat[:, :n_modes] @ _cmd[:n_modes]
        elif isinstance(n_modes, list):
            _cmdMat = _np.zeros((self._cmdMat.shape[1], len(n_modes)))
            _scmd = _np.zeros(_cmd.shape[0])
            for i, mode in enumerate(n_modes):
                _cmdMat.T[i] = self._cmdMat.T[mode]
                _scmd[i] = _cmd[mode]
            flat_cmd = _cmdMat @ _cmd
        else:
            raise TypeError("n_modes must be either an int or a list of int")
        self.flatCmd = flat_cmd
        return flat_cmd

    def loadImage2Shape(self, img, compute: int = None):
        """
        (Re)Loader for the image to flatten.

        Parameters
        ----------
        img : MaskedArray
            Image to flatten.
        compute_rec : bool, optional
            Wether to direclty compute the reconstructor with the imput image or
            not. The default is True.
        """
        self.shape2flat = img
        self._rec = self._rec.loadShape2Flat(img)
        if compute is not None:
            self.computeRecMat(compute)

    def computeRecMat(self, threshold=None):
        """
        Compute the reconstruction matrix for the loaded image.
        """
        print("Computing recontruction matrix...")
        self._recMat = self._rec.run(sv_threshold=threshold)

    def filterIntCube(self, zernModes: list = None):
        """
        Filter the interaction cube with the given zernike modes

        Parameters
        ----------
        zernModes : list
            Zernike modes to filter out this cube (if it's not already filtered).
            Default modes are [1,2,3] -> piston/tip/tilt.
        """
        with open(_os.path.join(self._path, _ifp.flagFile), "r", encoding="utf-8") as f:
            flag = f.read()
        if " filtered " in flag:
            print("Cube already filtered, skipping...")
            return
        else:
            print("Filtering cube...")
            self._oldCube = self._intCube
            zern2fit = zernModes if zernModes is not None else [1, 2, 3]
            self._intCube, new_tn = _ifp.filterZernikeCube(self.tn, zern2fit)
            self.loadNewTn(new_tn)
            self.filtered = True
        return self

    def loadNewTn(self, tn):
        """
        Load a new tracking number for the flattening.

        Parameters
        ----------
        tn : str
            Tracking number of the new data.
        """
        self.__update_tn(tn)
        self._reloadClass(tn)


    def _reloadClass(self, tn):
        """
        Reload function for the interaction cube

        Parameters
        ----------
        tn : str
            Tracking number of the new data.
        zernModes : list, optional
            Zernike modes to filter out this cube (if it's not already filtered).
            Default modes are [1,2,3] -> piston/tip/tilt.
        """
        self._cmdMat = self._loadCmdMat()
        self._rec = self._rec.loadInteractionCube(tn=tn)


    def _getMasterMask(self):
        """
        Creates the intersection mask of the interaction cube.
        """
        cubeMask = _np.sum(self._intCube.mask.astype(int), axis=2)
        master_mask = _np.zeros(cubeMask.shape, dtype=_np.bool_)
        master_mask[_np.where(cubeMask > 0)] = True
        return master_mask
    

    def _loadIntCube(self):
        """
        Interaction cube loader

        Return
        ------
        intCube : ndarray
            The interaction cube data array.
        """
        intCube = _osu.read_phasemap(_os.path.join(self._path, _ifp.cubeFile))
        with open(_os.path.join(self._path, _ifp.flagFile), "r") as file:
            lines = file.readlines()
        rebin = eval(lines[1].split("=")[-1])
        with open(_os.path.join(self._path, _ifp.flagFile), "r", encoding="utf-8") as f:
            flag = f.read()
        if " filtered " in flag:
            self.filtered = True
        else:
            self.filtered = False
        self.rebin = rebin
        return intCube

    def _loadCmdMat(self):
        """
        Command matrix loader. It loads the saved command matrix of the loaded
        cube.

        Returns
        -------
        cmdMat : ndarray
            Command matrix of the cube, saved in the tn path.
        """
        cmdMat = _osu.load_fits(_os.path.join(self._path, _ifp.cmdMatFile))
        return cmdMat

    def _loadReconstructor(self):
        """
        Builds the reconstructor object off the input cube

        Returns
        -------
        rec : object
            Reconstructor class.
        """
        rec = _crec.ComputeReconstructor(self._intCube)
        return rec

    def _loadFrameCenter(self):
        """
        Center frame loader, useful for image registration.

        Returns
        -------
        frame_center : TYPE
            DESCRIPTION.

        """
        frame_center = _osu.load_fits("data")
        return frame_center

    def _registerShape(self, shape):
        xxx = None
        dp = _ifp.findFrameOffset(self.tn, xxx)
        # cannot work. we should create a dedicated function, not necessarily linked to IFF or flattening
        return dp

    def __update_tn(self, tn):
        """
        Updates the tn and cube path if the tn is to change

        Parameters
        ----------
        tn : str
            New tracking number.
        """
        self.tn = tn
        self._path = _os.path.join(_ifp._intMatFold, self.tn)
