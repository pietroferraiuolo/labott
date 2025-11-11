"""
Zernike Generation Library
==========================
This module provides functions and utilities for generating Zernike polynomials,
which are a sequence of polynomials that are orthogonal on the unit disk. These
polynomials are commonly used in optics and wavefront analysis.

Author(s)
---------
- Tim van Werkhoven (t.i.m.vanwerkhoven@xs4all.nl) : Original Author,  Created in 2011-10-12
- Pietro Ferraiuolo (pietro.ferraiuolo@inaf.it) : Adapted in 2024

Functions
---------
- removeZernike(ima, modes=np.array([1, 2, 3, 4])): Remove Zernike modes from an image.
- removeZernikeAuxMask(img, mm, zlist): Remove Zernike modes from an image using an auxiliary mask.
- zernikeFit(img, zernike_index_vector, qpupil=True): Fit Zernike modes to an image.
- zernikeFitAuxmask(img, auxmask, zernike_index_vector): Fit Zernike modes to an image using an auxiliary mask.
- zernikeSurface(img, coef, mat): Generate Zernike surface from coefficients and matrix.
- _surf_fit(xx, yy, zz, zlist, ordering='noll'): Fit surface using Zernike polynomials.
- _getZernike(xx, yy, zlist, ordering='noll'): Get Zernike polynomials.
- _zernike_rad(m, n, rho): Calculate the radial component of Zernike polynomial (m, n).
- _zernike(m, n, rho, phi): Calculate Zernike polynomial (m, n).
- _zernikel(j, rho, phi): Calculate Zernike polynomial with Null coordinate j.
- _l2mn_ansi(j): Convert ANSI index to Zernike polynomial indices.
- _l2mn_noll(j): Convert Noll index to Zernike polynomial indices.

Example
-------
Example usage of the module:

```python
# Create a sample wavefront image (e.g., 256x256 pixels)
size = 256
y, x = np.ogrid[-size/2:size/2, -size/2:size/2]
radius = size / 2

# Create a circular pupil mask
pupil_mask = (x**2 + y**2) <= radius**2

# Generate a simulated wavefront with some aberrations
# Adding defocus (Z4) and astigmatism (Z5, Z6)
wavefront = np.random.normal(0, 0.1, (size, size))
wavefront = np.ma.masked_array(wavefront, mask=~pupil_mask)

# Initialize the Zernike fitter with a circular pupil
fitter = ZernikeFitter(fit_mask=pupil_mask)

# Fit Zernike modes 1-10 to the wavefront
modes_to_fit = list(range(1, 11))
coefficients, fitting_matrix = fitter.fit(wavefront, modes_to_fit)

print(f"Fitted Zernike coefficients: {coefficients}")

# Remove tip-tilt (modes 2 and 3) from the wavefront
corrected_wavefront = fitter.removeZernike(wavefront, zernike_index_vector=[2, 3])

# Generate a pure Zernike surface (e.g., coma, mode 7)
coma_surface = fitter.makeSurface(modes=[7])

# Fit modes on multiple ROIs and get global average
roi_coefficients = fitter.fitOnRoi(wavefront, modes2fit=[1, 2, 3], mode='global')
print(f"ROI-averaged coefficients: {roi_coefficients}")
```
"""

import numpy as _np
import math as _math
from contextlib import contextmanager as _contextmanager
from abc import abstractmethod, ABC
from . import roi as _roi
from opticalib import typings as _t
from arte.utils.zernike_generator import ZernikeGenerator
from arte.types.mask import CircularMask

fac = _math.factorial


class _ModeFitter(ABC):
    """
    Class for fitting Zernike polynomials to an image.

    Parameters
    ----------
    fit_mask : ImageData or CircularMask or np.ndarray, optional
        Mask to be used for fitting. Can be an ImageData, CircularMask, or ndarray.
        If None, a default CircularMask will be created.
    """

    def __init__(self, fit_mask: _t.Optional[_t.ImageData] = None):
        """
        Class for fitting Zernike polynomials to an image.

        Parameters
        ----------
        fit_mask : ImageData or CircularMask or np.ndarray, optional
            Mask to be used for fitting. Can be an ImageData, CircularMask, or ndarray.
            If None, a default CircularMask will be created.
        """
        if fit_mask is not None:
            if isinstance(fit_mask, CircularMask):
                self._fit_mask = fit_mask
            elif isinstance(fit_mask, _np.ndarray):
                self._fit_mask = CircularMask.fromMaskedArray(
                    _np.ma.masked_array(fit_mask, mask=fit_mask == 0),
                    method="COG",
                )
            else:
                self._fit_mask = CircularMask.fromMaskedArray(
                    fit_mask, mask=fit_mask.mask, method="COG"
                )
            self.auxmask = self._fit_mask.as_masked_array()
            self._mgen = self._create_modes_generator(self._fit_mask)
        else:
            self._fit_mask = None
            self.auxmask = None
            self._mgen = None

    @abstractmethod
    def _create_fitting_matrix(self, modes: list[int]) -> _t.MatrixLike:
        """
        Create the fitting matrix for the given basis modes.
        """
        raise NotImplementedError(
            "Subclasses must implement the `_create_fitting_matrix` method."
        )

    @abstractmethod
    def _create_modes_generator(self, mask: _t.MaskData) -> object:
        """
        Create the modes generator.
        """
        raise NotImplementedError(
            "Subclasses must implement the `_create_modes_generator` method."
        )

    @abstractmethod
    def _get_mode_from_generator(self, mode_index: int) -> _t.ImageData:
        """
        Get the mode defined on the mask from the generator.
        """
        raise NotImplementedError(
            "Subclasses must implement the `_get_mode_from_generator` method."
        )

    @property
    def fitMask(self) -> _t.ImageData:
        """
        Get the current fitting mask.

        Returns
        -------
        fit_mask : ImageData
            Current fitting mask.
        """
        return self.auxmask

    def setFitMask(self, fit_mask: _t.ImageData, method: str = "GOC") -> None:
        """
        Set the fitting mask.

        Parameters
        ----------
        fit_mask : ImageData or CircularMask or np.ndarray
            Mask to be used for fitting. Can be an ImageData, CircularMask, or ndarray.
        method : str, optional
            Method used my the `CircularMask.fromMaskedArray` function. Default is 'COG'.
        """
        if isinstance(fit_mask, CircularMask):
            self._fit_mask = fit_mask
        elif isinstance(fit_mask, _np.ndarray):
            self._fit_mask = CircularMask.fromMaskedArray(
                _np.ma.masked_array(fit_mask, mask=fit_mask == 0),
                method=method,
            )
        else:
            self._fit_mask = CircularMask.fromMaskedArray(fit_mask, method="COG")
        self.auxmask = self._fit_mask.as_masked_array()
        self._mgen = self._create_modes_generator(self._fit_mask)

    def fit(
        self, image: _t.ImageData, zernike_index_vector: list[int]
    ) -> tuple[_t.ArrayLike, _t.ArrayLike]:
        """
        Fit Zernike modes to an image.

        Parameters
        ----------
        image : ImageData
            Image for Zernike fit.
        zernike_index_vector : list[int]
            List containing the index of Zernike modes to be fitted starting from 1.

        Returns
        -------
        coeff : numpy array
            Vector of Zernike coefficients.
        mat : numpy array
            Matrix of Zernike polynomials.
        """
        image = self._make_sure_on_cpu(image)

        with self._temporary_zgen_from_image(image) as (pimage, _):
            mask = pimage.mask == 0
            mat = self._create_fitting_matrix(zernike_index_vector, mask)

            A = mat.T
            B = _np.transpose(pimage.compressed())
            coeffs = _np.linalg.lstsq(A, B, rcond=None)[0]
            return coeffs, A

    def fitOnRoi(
        self,
        image: _t.ImageData,
        modes2fit: _t.Optional[list[int]] = None,
        mode: str = "global",
    ) -> tuple[_t.ArrayLike, _t.ArrayLike]:
        """
        Computes modal coefficients over a segmented fitting area, i.e. a pupil
        mask divided into Regions Of Interest (ROI). The computation is based on
        the fitting of modes independently on each ROI.

        Parameters
        ----------
        image : ImageData
            Image for modal fit.
        modes2fit : list[int], optional
            List containing the index of  modes to be fitted starting from 1.
            Default is [1,2,3].
        mode : str, optional
            Mode of fitting.
            - `global` will return the mean of the fitted coefficient of each ROI
            - `local` will return the vector of fitted coefficient for each ROI
            Default is 'global'.

        Returns
        -------
        coeff : numpy array
            Vector of modal coefficients.
        mat : numpy array
            Matrix of modal polynomials.
        """
        if mode not in ["global", "local"]:
            raise ValueError("mode must be 'global' or 'local'")
        if modes2fit is None:
            modes2fit = [1, 2, 3]
        roiimg = _roi.roiGenerator(image)
        nroi = len(roiimg)
        print("Found " + str(nroi) + " ROI")
        coeff = _np.zeros([nroi, len(modes2fit)])
        for i in range(nroi):
            img2fit = _np.ma.masked_array(image.data, mask=roiimg[i])
            cc, _ = self.fit(img2fit, modes2fit)
            coeff[i, :] = cc
        if mode == "global":
            coeff = coeff.mean(axis=0)
        return coeff

    def makeSurface(self, modes: list[int], image: _t.ImageData = None) -> _t.ImageData:
        """
        Generate modal surface from image.

        Parameters
        ----------
        image : ImageData, optional
            Image for fit. If no image is provided, it will be generated a surface,
            defined on a circular mask with amplitude 1.
        modes : list[int], optional
            List of modes indices. Defaults to [1].

        Returns
        -------
        surface : ImageData
            Generated modal surface.
        """
        if image is None and self._mgen is None:
            raise ValueError(
                "Either an image must be provided or a fitting mask must be set."
            )
        elif image is not None:
            image = self._make_sure_on_cpu(image)
            mm = _np.where(image.mask == 0)
            zernike_surface = _np.zeros(image.shape)
            coeff, mat = self.fit(image, modes)
            zernike_surface[mm] = _np.dot(mat, coeff)
            surface = _np.ma.masked_array(zernike_surface, mask=image.mask)
        elif image is None and self._mgen is not None:
            surface = self._get_mode_from_generator(modes[0])
            if len(modes) > 1:
                for mode in modes[1:]:
                    surface += self._get_mode_from_generator(mode)
        return surface

    @_contextmanager
    def no_mask(self):
        """
        Context manager to temporarily clear the fitting mask and Zernike generator.

        Usage
        -----
        with zfitter.no_mask():
            coeffs, mat = zfitter.fit(image, modes)

        Within the context, ``self._fit_mask``, ``self._zgen`` and ``self.auxmask``
        are set to ``None`` so that ``fit`` will lazily create a temporary mask
        from the provided image. On exit, the previous values are restored.
        """
        prev_fit_mask = self._fit_mask
        prev_mgen = self._mgen
        prev_auxmask = self.auxmask.copy()
        try:
            self._fit_mask = None
            self._mgen = None
            self.auxmask = None
            yield self
        finally:
            self._fit_mask = prev_fit_mask
            self._mgen = prev_mgen
            self.auxmask = prev_auxmask

    @_contextmanager
    def _temporary_zgen_from_image(self, image: _t.ImageData):
        """
        Context manager to temporarily create a ZernikeGenerator from an image
        when self._zgen is None, and restore the original state afterwards.

        Parameters
        ----------
        image : ImageData
            Image from which to create a temporary ZernikeGenerator

        Yields
        ------
        tuple
            (modified_image, was_temporary) where was_temporary indicates if a temp generator was created
        """
        prev_zgen = self._mgen
        was_temporary = False

        try:
            if self._mgen is None:
                self._mgen = self._create_fit_mask_from_img(image)
                image = _np.ma.masked_array(
                    image.data, mask=self._mgen._boolean_mask.copy()
                )
                was_temporary = True
            yield image, was_temporary
        finally:
            if was_temporary:
                self._mgen = prev_zgen

    def _create_fit_mask_from_img(self, image: _t.ImageData) -> CircularMask:
        """
        Create a default CircularMask for fitting.

        Returns
        -------
        fit_mask : CircularMask
            Default fitting mask.
        """
        if not isinstance(image, _np.ma.masked_array):
            try:
                image = _np.ma.masked_array(image, mask=image == 0)
            except Exception as e:
                raise ValueError(
                    "Input image must be a numpy masked array or convertible to one."
                ) from e
        cmask = CircularMask(image.shape)
        cmask._mask = image.mask
        mgen = self._create_modes_generator(cmask)
        return mgen

    def _make_sure_on_cpu(self, img: _t.ImageData) -> _t.ImageData:
        """
        Ensure the image is on CPU.

        Parameters
        ----------
        img : ImageData
            Input image.

        Returns
        -------
        img_cpu : ImageData
            Image on CPU.
        """
        if isinstance(img, _np.ma.MaskedArray):
            return img
        else:
            import xupy as xp

            if isinstance(img, xp.ma.MaskedArray):
                img = img.asmarray()
            elif isinstance(img, xp.ndarray):
                img = img.get()
        return img


class ZernikeFitter(_ModeFitter):
    """
    Class for fitting Zernike polynomials to an image.

    Parameters
    ----------
    fit_mask : ImageData or CircularMask or np.ndarray, optional
        Mask to be used for fitting. Can be an ImageData, CircularMask, or ndarray.
        If None, a default CircularMask will be created.
    """

    def __init__(self, fit_mask: _t.Optional[_t.ImageData] = None):
        """The Initiator."""
        # Defines the auxmask (if any) mask from the Parent class
        super().__init__(fit_mask)

    def removeZernike(
        self, image: _t.ImageData, zernike_index_vector: list[int] = None
    ) -> _t.ImageData:
        """
        Remove Zernike modes from the image using the current fit mask.

        Parameters
        ----------
        image : ImageData
            Image from which to remove Zernike modes.
        zernike_index_vector : list[int], optional
            List of Zernike mode indices to be removed. Default is [1, 2, 3].

        Returns
        -------
        new_ima : ImageData
            Image with Zernike modes removed.
        """
        if zernike_index_vector is None:
            zernike_index_vector = [1, 2, 3]
        surf = self.makeSurface(zernike_index_vector, image)
        return _np.ma.masked_array(image - surf, image.mask)

    def _create_fitting_matrix(
        self, modes: list[int], mask: _t.MaskData
    ) -> _t.MatrixLike:
        """
        Create the fitting matrix for the given Zernike modes.
        
        Parameters
        ----------
        modes : list[int]
            List of Zernike mode indices.
        mask : MaskData
            Boolean mask defining the fitting area.
        
        Returns
        -------
        mat : MatrixLike
            Fitting matrix for the specified Zernike modes.
        """
        mat = []
        for zmode in modes:
            vv = self._mgen.getZernike(zmode)
            mat.append(vv[mask])
        mat = _np.array(mat)
        return mat

    def _create_modes_generator(self, mask: CircularMask) -> CircularMask:
        """
        Create a default CircularMask for fitting.

        Returns
        -------
        zgen : ZernikeGenerator
            The Zernike Generator defined on the created Circular Mask.
        """
        return ZernikeGenerator(mask)

    def _get_mode_from_generator(self, mode_index: int) -> _t.ImageData:
        """
        Get the mode defined on the mask from the generator.

        Parameters
        ----------
        mode_index : int
            Index of the Zernike mode to retrieve.

        Returns
        -------
        mode_image : ImageData
            The Zernike mode image corresponding to the given index.
        """
        return self._mgen.getZernike(mode_index)
