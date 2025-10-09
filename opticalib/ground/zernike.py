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
                self._fit_mask = CircularMask.fromMaskedArray(fit_mask, mask=fit_mask.mask, method="COG")
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
        if self._fit_mask is None and self._zgen is None:
            zgen = self._create_fit_mask_from_img(image)
            img2fit = _np.ma.masked_array(image.data, mask=zgen._boolean_mask.copy())
        else:
            zgen = self._zgen
            img2fit = image
        coeffs, mat = _surf_fit(img2fit, zgen, zernike_index_vector)
        return coeffs, mat

    def fitOnROi(
        self,
        image: _t.ImageData,
        z2fit: _t.Optional[list[int]] = None,
        mode: str = "global",
    ) -> tuple[_t.ArrayLike, _t.ArrayLike]:
        """
        Fit Zernike modes to an image or to an image using the current fit mask.

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


def generateZernMat(noll_ids: list[int], img_mask: _t.ImageData) -> _t.MatrixLike:
    """
    Generates the interaction matrix of the Zernike modes with Noll index
    in noll_ids on the mask in input

    Parameters
    ----------
    noll_ids : ArrayLike
        List of (Noll) mode indices to fit.
    img_mask : matrix bool
        Mask of the desired image.

    Returns
    -------
    ZernMat : MatrixLike [n_pix,n_zern]
        The Zernike interaction matrix of the given indices on the given mask.
    """

    zgen = ZernikeGenerator(CircularMask.fromMaskedArray(img_mask))
    mat = []
    for zmode in noll_ids:
        mat.append(zgen.getZernike(zmode).compressed())
    A = _np.array(mat).T
    return A


def zernikeFitOnRoi(
    img: _t.ImageData,
    auxmask: _t.Optional[_t.ImageData] = None,
    z2fit: _t.Optional[list[int]] = None,
    mode: str = "global",
) -> tuple[_t.ArrayLike, _t.ArrayLike]:
    """
    Fit Zernike modes to an image or to an image using an auxiliary mask.

    Parameters
    ----------
    img : numpy masked array
        Image for Zernike fit.
    auxmask : numpy array, optional
        Auxiliary mask. Default is the image mask.
    z2fit : numpy array, optional
        Vector containing the index of Zernike modes to be fitted starting from 1.
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
    roiimg = _roi.roiGenerator(img)
    nroi = len(roiimg)
    print("Found " + str(nroi) + " ROI")
    if auxmask is None:
        auxmask2use = img.mask
    else:
        auxmask2use = auxmask
    zcoeff = _np.zeros([nroi, len(z2fit)])
    for i in range(nroi):
        img2fit = _np.ma.masked_array(img.data, roiimg[i])
        cc, _ = zernikeFitAuxmask(img2fit, auxmask2use, z2fit)
        zcoeff[i, :] = cc
    if mode == "global":
        zcoeff = zcoeff.mean(axis=0)
    return zcoeff


def removeZernike(
    image: _t.ImageData, modes: _t.Optional[list[int]] = None
) -> _t.ImageData:
    """
    Remove Zernike modes from an image.

    Parameters
    ----------
    image : numpy masked array
        Image from which Zernike modes are to be removed.
    modes : numpy array, optional
        Zernike modes to be removed. Default is np.array([1, 2, 3, 4]).

    Returns
    -------
    new_ima : numpy masked array
        Image with Zernike modes removed.
    """
    if modes is None:
        modes = _np.array([1, 2, 3, 4])
    coeff, mat = zernikeFit(image, modes)
    surf = zernikeSurface(image, coeff, mat)
    return image - surf


def removeZernikeAuxMask(
    image: _t.ImageData, mask: _t.ImageData, zlist: list[int]
) -> _t.ImageData:
    """
    Remove Zernike modes from an image using an auxiliary mask.

    Parameters
    ----------
    image : numpy masked array
        Image from which Zernike modes are to be removed.
    mask : numpy array
        Auxiliary mask.
    zlist : numpy array
        List of Zernike modes to be removed.

    Returns
    -------
    new_ima : numpy masked array
        Image with Zernike modes removed.
    """
    coeff, mat = zernikeFitAuxmask(image, mask, zlist)
    img2 = _np.ma.masked_array(image.data, mask=mask == 0)
    surf = zernikeSurface(img2, coeff, mat)
    return _np.ma.masked_array(image - surf, image.mask)


def zernikeFit(
    image: _t.ImageData, zernike_index_vector: list[int], qpupil: bool = True
) -> tuple[_t.ArrayLike, _t.ArrayLike]:
    """
    Fit Zernike modes to an image.

    Parameters
    ----------
    img : numpy masked array
        Image for Zernike fit.
    zernike_index_vector : numpy array
        Vector containing the index of Zernike modes to be fitted starting from 1.
    qpupil : bool, optional
        If True, use a pupil mask; otherwise, use a circular pupil. Default is True.

    Returns
    -------
    coeff : numpy array
        Vector of Zernike coefficients.
    mat : numpy array
        Matrix of Zernike polynomials.
    """
    img1 = image.data
    mask = _np.invert(image.mask).astype(int)
    _, _, _, xx, yy = _geo.qpupil(mask) if qpupil else _geo.qpupil_circle(image)
    sx, sy = img1.shape
    pixsc = xx[1, 0] - xx[0, 0]
    rpix = 1 / pixsc
    cx = -(_np.min(xx) * rpix - 0.5)
    cy = -(_np.min(yy) * rpix - 0.5)
    cmask = CircularMask((sx, sy), rpix, (cx, cy))
    cmask._mask = image.mask
    zgen = ZernikeGenerator(cmask)
    mm = zgen._boolean_mask.copy()
    img2 = _np.ma.masked_array(img1, mask=mm)
    coeffs, mat = _surf_fit(img2, zgen, zernike_index_vector)
    return coeffs, mat


def zernikeFitAuxmask(
    image: _t.ImageData, auxmask: _t.ImageData, zernike_index_vector: list[int]
) -> tuple[_t.ArrayLike, _t.ArrayLike]:
    """
    Fit Zernike modes to an image using an auxiliary mask.

    Parameters
    ----------
    img : numpy masked array
        Image for Zernike fit.
    auxmask : numpy array
        Auxiliary mask.
    zernike_index_vector : numpy array
        Vector containing the index of Zernike modes to be fitted starting from 1.

    Returns
    -------
    coeff : numpy array
        Vector of Zernike coefficients.
    mat : numpy array
        Matrix of Zernike polynomials.
    """
    if isinstance(auxmask, _np.ndarray):
        tmp = _np.ma.masked_array(auxmask, mask=auxmask == 0)
    else:
        tmp = auxmask
    cmask = CircularMask.fromMaskedArray(tmp, mask=tmp.mask)
    zgen = ZernikeGenerator(cmask)
    # _ = zgen.getZernike(1)  # to initialize the ZGen <- should not be needed
    coeffs, mat = _surf_fit(image, zgen, zernike_index_vector)
    return coeffs, mat


def zernikeSurface(
    image: _t.ImageData, coeff: _t.ArrayLike, mat: _t.ArrayLike
) -> _t.ImageData:
    """
    Generate Zernike surface from coefficients and matrix.

    Parameters
    ----------
    img : numpy masked array
        Image for Zernike fit.
    coeff : numpy array
        Vector of Zernike coefficients.
    mat : numpy array
        Matrix of Zernike polynomials.

    Returns
    -------
    surf : numpy masked array
        Zernike surface generated by coefficients.
    """
    mm = _np.where(image.mask == 0)
    zernike_surface = _np.zeros(image.shape)
    zernike_surface[mm] = _np.dot(mat, coeff)
    return _np.ma.masked_array(zernike_surface, mask=image.mask)


def _surf_fit(
    zz: _np.ma.masked_array,
    zgen: ZernikeGenerator,
    zlist: list[int],
    ordering: str = "noll",
) -> tuple[_t.ArrayLike, _t.ArrayLike]:
    """
    Fit surface using Zernike polynomials.

    Parameters
    ----------
    zz : numpy array
        Surface data.
    zgen : ZernikeGenerator
        Zernike generator instance.
    zlist : list[int]
        List of Zernike modes.
    ordering : str, optional
        Ordering of Zernike modes. Default is 'noll'.

    Returns
    -------
    coeffs : numpy array
        Zernike coefficients.
    mat : numpy array
        Matrix of Zernike polynomials.
    """
    # tmp = _np.ma.masked_array(zz.data, zgen._boolean_mask)
    tmp = zz.copy()  # _np.ma.masked_array(zz.data, zgen._boolean_mask)

    tmp_mask = tmp.mask == 0

    if ordering not in ["noll"]:
        raise ValueError("ordering currently supported is only 'noll'")
    mat = []
    for zmode in zlist:
        vv = zgen.getZernike(zmode)
        mat.append(vv[tmp_mask])
    mat = _np.array(mat)
    A = mat.T
    B = _np.transpose(tmp.compressed())
    coeffs = _np.linalg.lstsq(A, B, rcond=None)[0]
    return coeffs, A
