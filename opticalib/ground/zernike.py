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
import numpy.ma as ma
# Create a sample image with a mask
image_data = np.random.random((100, 100))
mask = np.zeros((100, 100), dtype=bool)
mask[30:70, 30:70] = True
masked_image = ma.masked_array(image_data, mask=mask)
# Define Zernike modes to be removed
zernike_modes = np.array([1, 2, 3, 4])
# Remove Zernike modes from the image
cleaned_image = zernike.removeZernike(masked_image, zernike_modes)
# Display the original and cleaned images
import matplotlib.pyplot as plt
plt.figure()
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(masked_image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Cleaned Image")
plt.imshow(cleaned_image, cmap='gray')
plt.show()
```
"""

import numpy as _np
import math as _math
from contextlib import contextmanager as _contextmanager
from . import geo as _geo, roi as _roi
from opticalib import typings as _t
from arte.utils.zernike_generator import ZernikeGenerator
from arte.types.mask import CircularMask

fac = _math.factorial


class ZernikeFitter:
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
            self._zgen = ZernikeGenerator(self._fit_mask)
            self.auxmask = self._fit_mask.as_masked_array()
        else:
            self._fit_mask = None
            self._zgen = None
            self.auxmask = None

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
        self._zgen = ZernikeGenerator(self._fit_mask)
        self.auxmask = self._fit_mask.as_masked_array()

    def getFitMask(self) -> _t.ImageData:
        """
        Get the current fitting mask.

        Returns
        -------
        fit_mask : ImageData
            Current fitting mask.
        """
        return self.auxmask

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
        prev_zgen = self._zgen
        prev_auxmask = self.auxmask.copy()
        try:
            self._fit_mask = None
            self._zgen = None
            self.auxmask = None
            yield self
        finally:
            self._fit_mask = prev_fit_mask
            self._zgen = prev_zgen
            self.auxmask = prev_auxmask

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
        coeff, mat = self.fit(image, zernike_index_vector)
        surf = self.makeZernikeSurface(image, coeff, mat)
        return _np.ma.masked_array(image - surf, image.mask)

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
        if self._fit_mask is None and self._zgen is None:
            zgen = self._create_fit_mask_from_img(image)
            image = _np.ma.masked_array(image.data, mask=zgen._boolean_mask.copy())
        else:
            zgen = self._zgen
            image = image
        tmp_mask = image.mask == 0
        mat = []
        for zmode in zernike_index_vector:
            vv = zgen.getZernike(zmode)
            mat.append(vv[tmp_mask])
        mat = _np.array(mat)
        A = mat.T
        B = _np.transpose(image.compressed())
        coeffs = _np.linalg.lstsq(A, B, rcond=None)[0]
        return coeffs, A

    def fitOnROi(
        self,
        image: _t.ImageData,
        z2fit: _t.Optional[list[int]] = None,
        mode: str = "global",
    ) -> tuple[_t.ArrayLike, _t.ArrayLike]:
        """
        Computes Zernike coefficients over a segmented fitting area, i.e. a pupil
        mask divided into Regions Of Interest (ROI). The computation is based on
        the fitting of Zernike modes independently on each ROI.

        Parameters
        ----------
        image : ImageData
            Image for Zernike fit.
        z2fit : list[int], optional
            List containing the index of Zernike modes to be fitted starting from 1.
            Default is [1,2,3].
        mode : str, optional
            Mode of fitting.
            - `global` will return the mean of the fitted zernike coefficient of each ROI
            - `local` will return the vector of fitted zernike coefficient for each ROI
            Default is 'global'.

        Returns
        -------
        coeff : numpy array
            Vector of Zernike coefficients.
        mat : numpy array
            Matrix of Zernike polynomials.
        """
        if mode not in ["global", "local"]:
            raise ValueError("mode must be 'global' or 'local'")
        if z2fit is None:
            z2fit = [1, 2, 3]
        roiimg = _roi.roiGenerator(image)
        nroi = len(roiimg)
        print("Found " + str(nroi) + " ROI")
        zcoeff = _np.zeros([nroi, len(z2fit)])
        for i in range(nroi):
            # TODO: Da pensare se qui va fittato sulla roi o, se c'è un'auxmas, su quella.
            img2fit = _np.ma.masked_array(image.data, mask=roiimg[i])
            cc, _ = self.fit(img2fit, z2fit)
            zcoeff[i, :] = cc
        if mode == "global":
            zcoeff = zcoeff.mean(axis=0)
        return zcoeff

    def makeZernikeSurface(
        self, image: _t.ImageData, coeff: _t.ArrayLike, mat: _t.ArrayLike
    ) -> _t.ImageData:
        """
        Generate Zernike surface from coefficients and matrix.

        Parameters
        ----------
        image : ImageData
            Image for Zernike fit.
        coeff : numpy array
            Vector of Zernike coefficients.
        mat : numpy array
            Matrix of Zernike polynomials.

        Returns
        -------
        zernike_surface : ImageData
            Generated Zernike surface.
        """
        image = self._make_sure_on_cpu(image)
        mm = _np.where(image.mask == 0)
        zernike_surface = _np.zeros(image.shape)
        zernike_surface[mm] = _np.dot(mat, coeff)
        return _np.ma.masked_array(zernike_surface, mask=image.mask)

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
        zgen = ZernikeGenerator(cmask)
        return zgen

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

