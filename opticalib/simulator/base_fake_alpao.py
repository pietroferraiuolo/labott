import os
import numpy as np
from tps import ThinPlateSpline
from abc import ABC, abstractmethod
from opticalib import folders as fp
from opticalib.ground import osutils as osu, zernike as zern
from opticalib.core.read_config import load_yaml_config as cl

######################################
## Utility classes for creating the ##
##       simulated Alpao            ##
######################################

_alpao_list = os.path.join(os.path.abspath(__file__), "alpao_list.yaml")


def getDmCoordinates(nacts: int):
    """
    Generates the coordinates of the DM actuators for a given DM size and actuator sequence.

    Parameters
    ----------
    Nacts : int
        Total number of actuators in the DM.

    Returns
    -------
    np.array
        Array of coordinates of the actuators.
    """
    dms = cl(_alpao_list)[f"DM{nacts}"]
    nacts_row_sequence = eval(dms["coords"])
    n_dim = nacts_row_sequence[-1]
    upper_rows = nacts_row_sequence[:-1]
    lower_rows = [l for l in reversed(upper_rows)]
    center_rows = [n_dim] * upper_rows[0]
    rows_number_of_acts = upper_rows + center_rows + lower_rows
    n_rows = len(rows_number_of_acts)
    cx = np.array([], dtype=int)
    cy = np.array([], dtype=int)
    for i in range(n_rows):
        cx = np.concatenate(
            (
                cx,
                np.arange(rows_number_of_acts[i])
                + (n_dim - rows_number_of_acts[i]) // 2,
            )
        )
        cy = np.concatenate((cy, np.full(rows_number_of_acts[i], i)))
    coords = np.array([cx, cy])
    return coords


def createMask(nacts: int, shape: tuple[int] = (512, 512)):
    """
    Generates a circular mask for a mirror based on its optical diameter and pixel scale.

    Parameters
    ----------
    opt_diameter : float
        The mirror's diameter in millimeters.
    pixel_scale : float
        Scale in pixels per millimeter.
    shape : tuple, optional
        The shape of the output mask (height, width), by default (512, 512).

    Returns
    -------
    np.ndarray
        A boolean array of the given shape. True values represent the mirror area.
    """
    dm = cl(_alpao_list)[f"DM{nacts}"]
    opt_diameter = float(dm["opt_diameter"])
    pixel_scale = float(dm["pixel_scale"])
    height, width = shape
    cx, cy = width / 2, height / 2
    radius = (opt_diameter * pixel_scale) / 2  # radius in pixels
    y, x = np.ogrid[:height, :width]
    mask = (x - cx) ** 2 + (y - cy) ** 2 >= radius**2
    return mask


def pixel_scale(nacts: int):
    """
    Returns the pixel scale of the DM.

    Parameters
    ----------
    nacts : int
        Number of actuators in the DM.

    Returns
    -------
    float
        Pixel scale of the DM.
    """
    dm = cl(_alpao_list)[f"DM{nacts}"]
    return float(dm["pixel_scale"])


def generate_zernike_matrix(noll_ids, img_mask, scale_length: float = None):
    """
    Generates the interaction matrix of the Zernike modes with Noll index
    in noll_ids on the mask in input

    Parameters
    ----------
    noll_ids : ndarray(int) [Nzern,]
        Array of Noll indices to fit.

    img_mask : matrix bool
        Mask of the desired image.

    scale_length : float, optional
        The scale length to use for the Zernike fit.
        The default is the maximum of the image mask shape.

    Returns
    -------
    ZernMat : ndarray(float) [Npix,Nzern]
        The Zernike interaction matrix of the given indices on the given mask.

    """
    n_pix = np.sum(1 - img_mask)
    if isinstance(noll_ids, int):
        noll_ids = np.arange(1, noll_ids + 1, 1)
    n_zern = len(noll_ids)
    ZernMat = np.zeros([n_pix, n_zern])
    for i in range(n_zern):
        ZernMat[:, i] = _project_zernike_on_mask(noll_ids[i], img_mask, scale_length)
    return ZernMat


def _project_zernike_on_mask(noll_number: int, mask, scale_length: float = None):
    """
    Project the Zernike polynomials identified by the Noll number in input
    on a given mask.
    The polynomials are computed on the circle inscribed in the mask by default,
    or on a circle of radius scale_length if the corresponding input is given
    Masked data is then normalized as follows:
    data = ma.data[~ma.mask], data = (data - mean(data))/std(data)

    Parameters
    ----------
    noll_number : int
        Noll index of the desired Zernike polynomial.
    mask : matrix bool
        Mask of the desired image.

    Returns
    -------
    masked_data : ndarray
        Flattenned array of the masked values of the Zernike
        shape projected on the mask.

    """
    if noll_number < 1:
        raise ValueError("Noll index must be equal to or greater than 1")
    # Image dimensions
    X, Y = np.shape(mask)
    # Determine circle radius on to which define the Zernike
    if scale_length is not None:
        r = scale_length
    else:
        r = np.max([X, Y]) / 2
    # Conversion to polar coordinates on circle of radius r
    phi = lambda i, j: np.arctan2((j - Y / 2.0) / r, (i - X / 2.0) / r)
    rho = lambda i, j: np.sqrt(((j - Y / 2.0) / r) ** 2 + ((i - X / 2.0) / r) ** 2)
    mode = np.fromfunction(
        lambda i, j: zern._zernikel(noll_number, rho(i, j), phi(i, j)), [X, Y]
    )
    # masked_mode = np.ma.masked_array(mode,mask)
    # plt.figure();plt.imshow(masked_mode,origin='lower');plt.title('Zernike ' + str(noll_number))
    # masked_data = masked_mode.data[~masked_mode.mask]
    masked_data = mode[~mask]
    # Normalization of the masked data: null mean and unit STD
    if noll_number > 1:
        masked_data = (masked_data - np.mean(masked_data)) / np.std(masked_data)
    return masked_data


######################################
## Base class that creates the simu ##
##          lated Alpao             ##
######################################


class BaseFakeAlpao(ABC):
    """
    Base class for deformable mirrors.
    """

    def __init__(self, nActs: int):
        """
        Initializes the base deformable mirror with the number of actuators.
        """
        self.mirrorModes = None
        self.nActs = nActs
        self._pxScale = pixel_scale(self.nActs)
        self.actCoords = getDmCoordinates(self.nActs)
        self.mask = createMask(self.nActs)
        self._scaledActCoords = self._scaleActCoords()
        self._iffCube = None
        self.IM = None
        self.ZM = None
        self.RM = None

        print(" " * 11 + f"DM {self.nActs}\n")
        self._load_matrices()

    @abstractmethod
    def set_shape(self, command: np.array, differential: bool = False):
        """
        Applies the DM to a wavefront.

        Parameters
        ----------
        command : np.array
            Wavefront to which the DM will be applied.

        differential : bool
            If True, the command is the differential wavefront.

        Returns
        -------
        np.array
            Modified wavefront.
        """
        raise NotImplementedError

    @abstractmethod
    def get_shape(self):
        """
        Returns the current shape of the DM.

        Returns
        -------
        np.array
            Current shape of the DM.
        """
        raise NotImplementedError

    def _load_matrices(self):
        """
        Loads the required matrices for the deformable mirror's operations.
        """
        if not os.path.exists(fp.INFLUENCE_FUNCTIONS_FILE(self.nActs)):
            print(
                f"First time simulating DM {self.nActs}. Generating influence functions..."
            )
            self._simulate_Zonal_Iff_Acquisition()
        else:
            print(f"Loaded influence functions.")
            self._iffCube = np.ma.masked_array(
                osu.load_fits(fp.INFLUENCE_FUNCTIONS_FILE(self.nActs))
            )
        self._create_int_and_rec_matrices()
        self._create_zernike_matrix()

    def _create_zernike_matrix(self):
        """
        Create the Zernike matrix for the DM.
        """
        if not os.path.exists(fp.ZERNMAT_FILE(self.nActs)):
            n_zern = self.nActs
            print("Computing Zernike matrix...")
            self.ZM = zern.generate_zernike_matrix(n_zern, self.mask)
            osu.save_fits(fp.ZERNMAT_FILE(self.nActs), self.ZM)
        else:
            print(f"Loaded Zernike matrix.")
            self.ZM = osu.load_fits(fp.ZERNMAT_FILE(self.nActs))

    def _create_int_and_rec_matrices(self):
        """
        Create the interaction matrices for the DM.
        """
        if not os.path.exists(fp.INTMAT_FILE(self.nActs)):
            print("Computing interaction matrix...")
            self.IM = np.array(
                [
                    (self._iffCube[:, :, i].data)[self.mask == 0]
                    for i in range(self._iffCube.shape[2])
                ]
            )
            osu.save_fits(fp.INTMAT_FILE(self.nActs), self.IM)
        else:
            print(f"Loaded interaction matrix.")
            self.IM = osu.load_fits(fp.INTMAT_FILE(self.nActs))
        if not os.path.exists(fp.RECMAT_FILE(self.nActs)):
            print("Computing reconstruction matrix...")
            self.RM = np.linalg.pinv(self.IM)
            osu.save_fits(fp.RECMAT_FILE(self.nActs), self.RM)
        else:
            print(f"Loaded reconstruction matrix.")
            self.RM = osu.load_fits(fp.RECMAT_FILE(self.nActs))

    def _simulate_Zonal_Iff_Acquisition(self):
        """
        Simulate the influence functions by imposing 'perfect' zonal commands.

        Parameters
        ----------
        amps : float or np.ndarray, optional
            Amplitude(s) for the actuator commands. If a single float is provided,
            it is applied to all actuators. Default is 1.0.

        Returns
        -------
        np.ma.MaskedArray
            A masked cube of influence functions with shape (height, width, nActs).
        """
        # Get the number of actuators from the coordinates array.
        n_acts = self.actCoords.shape[1]
        max_x, max_y = self.mask.shape
        # Create pixel grid coordinates.
        pix_coords = np.zeros((max_x * max_y, 2))
        pix_coords[:, 0] = np.repeat(np.arange(max_x), max_y)
        pix_coords[:, 1] = np.tile(np.arange(max_y), max_x)
        # Convert actuator coordinates to pixel coordinates.
        # Note: self.coords is of shape (2, nActs) where first row is x and second is y.
        act_coords = self.actCoords.T  # shape: (n_acts, 2)
        act_pix_coords = np.zeros((n_acts, 2), dtype=int)
        act_pix_coords[:, 0] = (
            act_coords[:, 1] / np.max(act_coords[:, 1]) * max_x
        ).astype(int)
        act_pix_coords[:, 1] = (
            act_coords[:, 0] / np.max(act_coords[:, 0]) * max_y
        ).astype(
            int
        )  # corrected to use act_coords[:, 0]
        # Prepare an image cube to store the influence functions.
        img_cube = np.zeros((max_x, max_y, n_acts))
        amps = np.ones(n_acts)
        # For each actuator, compute the influence function with a TPS interpolation.

        ## prove con multiprocessing ##
        # import multiprocessing as mp
        # _tps = _TPS(alpha=0.0, act_pix_coords=act_pix_coords, pix_coords=pix_coords)
        # ncores = mp.cpu_count()
        # with mp.Pool(ncores) as pool:
        #     img_cube = pool.map(_tps.fit, range(n_acts))
        # img_cube = np.ma.dstack(img_cube)

        ## ##

        for k in range(n_acts):
            print(f"{k+1}/{n_acts}", end="\r", flush=True)
            # Create a command vector with a single nonzero element.
            act_data = np.zeros(n_acts)
            act_data[k] = amps[k]
            tps = ThinPlateSpline(alpha=0.0)
            tps.fit(act_pix_coords, act_data)
            flat_img = tps.transform(pix_coords)
            img_cube[:, :, k] = flat_img.reshape((max_x, max_y))
        # Create a cube mask that tiles the local mirror mask for each actuator.
        cube_mask = np.tile(self.mask, n_acts).reshape(img_cube.shape, order="F")
        cube = np.ma.masked_array(img_cube, mask=cube_mask)
        # Save the cube to a FITS file.
        fits_file = fp.INFLUENCE_FUNCTIONS_FILE(self.nActs)
        osu.save_fits(fits_file, cube)
        self._iffCube = cube

    def _scaleActCoords(self):
        """
        Scales the actuator coordinates to the mirror's pixel scale.
        """
        max_x, max_y = self.mask.shape
        act_coords = self.actCoords.T  # shape: (n_acts, 2)
        act_pix_coords = np.zeros((self.nActs, 2), dtype=int)
        act_pix_coords[:, 0] = (
            act_coords[:, 1] / np.max(act_coords[:, 1]) * max_x
        ).astype(int)
        act_pix_coords[:, 1] = (
            act_coords[:, 0] / np.max(act_coords[:, 0]) * max_y
        ).astype(int)
        return act_pix_coords


class _TPS:

    def __init__(self, alpha: float, act_pix_coords, pix_coords):
        self.alpha = alpha
        self.act_pix_coords = act_pix_coords
        self.pix_coords = pix_coords

    def fit(self, k: int):
        act_data = np.zeros(self.act_pix_coords.shape[0])
        act_data[k] = 1.0
        tps = ThinPlateSpline(alpha=self.alpha)
        tps.fit(self.act_pix_coords, act_data)
        flat_img = tps.transform(self.pix_coords)
        return flat_img
