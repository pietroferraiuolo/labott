import os as _os
import io as _io
import contextlib as _clib
import numpy as _np
from opticalib import folders as _fn
from opticalib.ground import osutils as _osu
from . import iff_module as _iff, iff_processing as _ifp
from opticalib import typings as _ot
from opticalib.core.read_config import getStitchingConfig as _gsc
from ._stitching_algorithm import map_stitching as _map_stitching


_ts = _osu.newtn


class StitchAnalysis:
    """
    Class to process and analyze acquisitions in sub-aperture mode of a mirror,
    to perform the stitching algorithm and produce stitched images.
    """
    
    def __init__(self):
        self.constants = _gsc()


    def processTns(self, tnvec: list[str, tuple[float] | list[float]]) -> str:
        """
        Process the IFF obtained during the acquisition, and produces the modes and cubes
        for each position, and produces a cube for each IFF in different positions.

        Parameters
        ----------
        tnvec: tuple
            A tuple of tuples, where each inner tuple contains the scan name and the coordinates
            e.g. (("scan1", (x1, z1)), ("scan2", (x2, z2)), ...)

        Returns
        -------
        str
            The new TN where everything is saved
        """
        newtn = _ts()
        dir = _os.path.join(_fn.INTMAT_ROOT_FOLDER, newtn)
        if not _os.path.exists(dir):
            _os.mkdir(dir)
        full_coord_header = {}
        for j, (tn, xz) in enumerate(tnvec):
            print(f"Processing {(j+1)/len(tnvec)*100:d}%: {tn}", end="\r", flush=True)
            _ifp.process(tn=tn)
            header = {f"X{j}": xz[0], f"Z{j}": xz[1]}
            full_coord_header.update(header)
            captured_output = _io.StringIO()
            with _clib.redirect_stdout(captured_output):
                _ifp.saveCube(tn=tn, cube_header=header)
        for i in range(self.dm.nActs):
            modevec = []
            for k, (tn, _) in enumerate(tnvec):
                print(
                    f"Processing Mode {i+1}/{self.dm.nActs} : {k/len(tnvec)*100}",
                    end="\r",
                    flush=True,
                )
                fl = _osu.getFileList(tn, fold="IFFunctions", key="mode_")
                img = _osu.load_fits(fl[i])
                modevec.append(img)
            modevec = _np.ma.dstack(modevec)
            _osu.save_fits(
                _os.path.join(dir, f"mode_{i:03d}_cube.fits"),
                modevec,
                header=full_coord_header,
            )
        return newtn


    def stitchAllIffCubes(self, tn: str) -> _ot.CubeData:
        """
        Stitch the IFF cubes obtained during the acquisition, and produces a single cube
        for each IFF in different positions.

        Parameters
        ----------
        tn: str
            The name of the scan to analyze

        Returns
        -------
        stitched_cube = np.MaskedArray
            The cube of stitched iffs.
        """
        newtn = _ts()
        dir = _os.path.join(_fn.INTMAT_ROOT_FOLDER, newtn)
        if not _os.path.exists(dir):
            _os.mkdir(dir)
        cubelist = _osu.getFileList(tn, fold="IntMatrices", key="mode_")
        stitch_list = []
        for cube in cubelist:
            cube, header = _osu.load_fits(cube, True)
            stitch_list.append(self.stitchSingleIffCube(cube, header))
        stitched = _np.ma.dstack(stitch_list)
        header = {}
        header["STITCH"] = (True, "if the cube is the result of stitching")
        header["REBIN"] = (1, "cube rebinning factor")
        _osu.save_fits(_os.path.join(dir, "IMCube.fits"), stitched)
        return stitched


    def stitchSingleIffCube(
        self,
        cube: _ot.CubeData,
        header: dict[str, _ot.Any] | _ot.Header,
        deg: float = None,
        average: _ot.Optional[_ot.ImageData] = None,
    ):
        """
        Analyze a single scan and performs the stitching

        Parameters
        ----------
        cube : _ot.CubeData
            The cube data to be stitched.
        header : dict or astropy.Header
            The header of the cube containing the coordinates.
        average: np.ndarray, optional
            The average image to subtract from the cube images.
        deg: float, optional
            The rotation angle in degrees to apply to the coordinates.

        Returns
        -------
        np.MaskedArray
            The stitched image
        """
        coords = self.retrieveCubeCoords(n_positions=cube.shape[-1], header=header)
        coords = self._transform_coord(coords, deg=deg)
        if average is not None:
            pass
        fm, iv = self._prepare_masks_and_images(cube, coords)
        stitched = _map_stitching(iv, fm, [1, 2, 3])
        return stitched


    def stitchSingleScansionCube(
        self,
        tn: str,
        deg: float = None,
        average: _ot.Optional[_ot.ImageData] = None,
    ) -> _ot.ImageData:
        """
        Analyze a single scansion cube and performs the stitching.

        Parameters
        ----------
        tn; str
            The tracking number of the scansion to analyze.
        average: np.ndarray, optional
            The average image to subtract from the cube images.
        deg: float, optional
            The rotation angle in degrees to apply to the coordinates.

        Returns
        -------
        np.MaskedArray
            The stitched image.
        """
        if deg is None:
            deg = self.constants["alpha"]
        cube, header = _osu.load_fits(
            _os.path.join(_fn.OPD_IMAGES_ROOT_FOLDER, tn), True
        )
        coords = self.retrieveCubeCoords(n_positions=cube.shape[-1], header=header)
        coords = self._transform_coord(coords, deg=deg)
        if average is not None:
            pass
        fm, iv = self._prepare_masks_and_images(cube, coords)
        stitched = _map_stitching(iv, fm, [1, 2, 3])
        return stitched


    def retrieveCubeCoords(
        self, n_positions: int, header: dict[str, _ot.Any] | _ot.Header
    ) -> _ot.ArrayLike:
        """
        Returns the coordinates written in the cube's header

        Parameters
        ----------
        n_positions : int
            The number of pair coordinate positions in the cube.
        header : dict or astropy.Header
            The header of the cube containing the coordinates.

        Returns
        -------
        coords : _ot.ArrayLike
            An array of coordinates in the form of (x, z) for each position.
        """
        coords = []
        for ii in range(n_positions):
            coords.append((header[f"X{ii}"], header[f"Z{ii}"]))
        return _np.array(coords)


    def reloadConstants(self):
        """Reload the constants from the configuration file"""
        self.constants = _gsc()
        print("Constants reloaded")


    def _transform_coord(
        self, coords: _ot.ArrayLike, deg: float = None
    ) -> _ot.ArrayLike:
        """
        Transform the coordinates by applying a rotation and flipping them.

        Parameters
        ----------
        coords : _ot.ArrayLike
            The coordinates to be transformed, expected to be in the shape of `(n_img, 2)`.
        deg : float, optional
            The rotation angle in degrees to apply to the coordinates. If None, uses the default from constants.

        Returns
        -------
        _ot.ArrayLike
            The transformed coordinates after rotation and flipping.
        """
        if deg is None:
            deg = self.constants["alpha"]
        rot_mat = _np.array(
            [
                [_np.cos(_np.deg2rad(deg)), -_np.sin(_np.deg2rad(deg))],
                [_np.sin(_np.deg2rad(deg)), _np.cos(_np.deg2rad(deg))],
            ]
        )
        coords_m = coords / 1000
        coords_m = _np.flip(coords_m, axis=0)
        new_coords = []
        for ii in range(len(coords_m)):
            cc = _np.matmul(rot_mat, coords_m[ii])
            new_coords.append(cc)
        new_coords = _np.array(new_coords)
        rot_coords = new_coords + [-new_coords[:, 0].min(), -new_coords[:, 1].min()]
        return rot_coords


    def _prepare_masks_and_images(
        self, imgcube: _ot.CubeData, pos: _ot.ArrayLike
    ) -> tuple[_ot.ImageData, _ot.CubeData]:
        """
        Extract the mask vector from the cube, created the full mask and prepares
        the images cube masked in the right place.

        Parameters
        ----------
        imgcube : _ot.CubeData
            The input image cube, expected to be in the shape of `(n_img, n_px, n_px)`.
        pos : _ot.ArrayLike
            The positions where the images should be placed, expected to be in the shape of `(n_img, 2)`.

        Returns
        -------
        fullmask : np.ndarray
            A full mask array that combines all individual masks, sized according to the maximum position.
        imgvecout : _ot.CubeData
            A cube of images masked and placed according to the specified positions.
        """
        n1, n2, n3 = _np.shape(imgcube)
        if n1 == n2:
            imgcube = _np.transpose(imgcube, (2, 0, 1))
        elif n2 == n3:
            pass
        else:
            print(
                "Warning: could not determine the right cube orientation. Be sure it is `(n_img, n_px, n_px)`"
            )
        maskvec = []
        for ii in imgcube:
            maskvec.append(ii.mask)
        ps = 1 / self.constants["pixel_scale"] * 1000
        pixpos = (pos * ps).astype(int)
        imgsize = _np.array(_np.shape(maskvec[0])).astype(int)
        imgfsize = imgsize + pixpos.max(0)
        fullmask = _np.ones([int(imgfsize[0]), int(imgfsize[1])])
        imgvecout = []
        for i in range(len(pixpos)):
            yi = slice(pixpos[i, 0], pixpos[i, 0] + imgsize[0])
            xi = slice(pixpos[i, 1], pixpos[i, 1] + imgsize[1])
            fullmask[xi, yi] = fullmask[xi, yi] * (maskvec[i])
            imgf = _np.ma.zeros([int(imgfsize[0]), int(imgfsize[1])])
            maskf = _np.ones([int(imgfsize[0]), int(imgfsize[1])])
            imgf[xi, yi] = imgcube[i]
            # imgf = imgf.data
            maskf[xi, yi] = maskf[xi, yi] * (maskvec[i])
            imgf.mask = maskf
            # imgf = _np.ma.masked_array(imgf,maskf)
            imgvecout.append(imgf)
        imgvecout = _np.ma.dstack(imgvecout)
        return fullmask, _np.transpose(imgvecout, (2,0,1))


class StitchAcquire:
    """
    Class to acquire images in sub-aperture mode of a mirror, to be later processed
    and analyzed with the `StitchAnalysis` class.

    Parameters
    ----------
    dm : _ot.DeformableMirrorDevice
        The deformable mirror device used for the acquisition.
    interf : _ot.InterferometerDevice
        The interferometer device used for the acquisition.
    motors : _ot.GenericDevice
        The motor device controlling the axis of the acquisition.
    """

    def __init__(
        self,
        dm: _ot.DeformableMirrorDevice,
        interf: _ot.InterferometerDevice,
        motors: _ot.GenericDevice,
    ):
        self.dm = dm
        self.interf = interf
        self.axis = motors
        self.dataFold = _fn.BASE_DATA_PATH
        self.cvec = None

    def getAxisPosition(self) -> dict[str, float]:
        """
        Get the current position of the motor's axis.

        Returns
        -------
        dict[str, float]
            A dictionary with the current position of the axis in the format:
            {"x": x_position, "y": y_position, "z": z_position}
        """
        x = self.axis.get_position("x")
        y = self.axis.get_position("y")
        z = self.axis.get_position("z")
        positions = {"x": x, "y": y, "z": z}
        self.axis_pos = positions
        return positions

    def setAxisPosition(self, coord: list[float]):
        """
        Set the position of the motor's axis to the specified coordinates.

        Parameters
        ----------
        coord : list[float]
            A list containing the x and z coordinates to set the axis position.
            e.g. [x, z]
        """
        x, z = coord
        self.axis._go2coord([x, z])
        print(f"Reached position [{x},{z}]")

    def getCoordinatesVector(
        self, nstep: int, step_in_mm: tuple[int, int] = (3, 3), live_pos: bool = False
    ) -> list[tuple[float, float]]:
        """
        Get the coordinates vector of the grid for scanning.

        Parameters
        ----------
        nstep : int
            The number of steps in each direction (x and z).
        step_in_mm : tuple[int, int], optional
            The step size in millimeters for the x and z directions. Default is (3, 3).
        live_pos : bool, optional
            If True, the starting position will be the current position of the axis.
            If False, it will use the starting coordinates defined in the constants.
            Default is False.

        Returns
        -------
        coord_vector : list[tuple[float, float]]
            A list of tuples containing the coordinates (x, z) for each step in the grid.
            The coordinates are calculated based on the starting position and the step size.
        """
        xstep, ystep = step_in_mm
        coord = []
        if live_pos:
            curr_pos = self.getAxisPosition()
            start_pos = [int(curr_pos["x"]), int(curr_pos["z"])]
        else:
            start_pos = self.constants["starting_coords"]
        for i in range(nstep):
            row = []
            for j in range(nstep):
                xx = start_pos[0] + j * xstep
                zz = start_pos[1] + i * ystep
                ci = xx, zz
                row.append(ci)
            if i % 2 == 1:
                row.reverse()
            coord.extend(row)
        return coord

    def acquireSingleScan(self, coord_vec: list[float], nframes: int = 1) -> str:
        """
        Acquire a single scan at each position in the coordinate vector.

        Parameters
        ----------
        coord_vec : list[float]
            A list of tuples with the coordinates (x, z) where the scan will be acquired.
            e.g. [(x1, z1), (x2, z2), ...]
        nframes : int, optional
            The number of frames to acquire at each position. Default is 1.

        Returns
        -------
        tn : str
            The tracking number (TN) of the scan, where is located the cube of acquired
            images at each position.
        """
        tn = _ts()
        print(tn)
        ddir = _os.path.join(_fn.OPD_IMAGES_ROOT_FOLDER, tn)
        if not _os.path.exists(ddir):
            _os.mkdir(ddir)
        header = {}
        imglist = []
        for i, xz in enumerate(coord_vec):
            header[f"X{i}"] = xz[0]
            header[f"Z{i}"] = xz[1]
            self.setAxisPosition(xz)
            imglist.append(self.interf.acquire_map(nframes=nframes))
        cube = _np.ma.dstack(imglist)
        _osu.save_fits(_os.path.join(ddir, "cube.fits"), cube, header=header)
        self.axis.homing()
        return tn

    def acquireSubApertureIFF(
        self, coord_vec: list[float]
    ) -> list[list[str, tuple[float, float]]]:
        """
        Acquire the IFF at each position in the coordinate vector.

        Parameters
        ----------
        coord_vec: list
            A list of tuples with the coordinates (x, z) where the IFF will be acquired.

        Returns
        -------
        tnvec : list
            A list of lists, where each list contains the TN and the coordinates.
        """
        tnvec = []
        for _, xz in enumerate(coord_vec):
            self.setAxisPosition(xz)
            tn = _iff.iffDataAcquisition(self.dm, self.interf)
            tnvec.append([tn, xz])
        self.axis.homing()
        return tnvec
