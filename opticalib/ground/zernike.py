"""
Zernike Generation Library
==========================
Author(s)
---------
    - Tim van Werkhoven (t.i.m.vanwerkhoven@xs4all.nl) : Original Author
        Created in 2011-10-12
    - Pietro Ferraiuolo (pietro.ferraiuolo@inaf.it) : Adapted in 2024

Description
-----------
This module provides functions and utilities for generating Zernike polynomials,
which are a sequence of polynomials that are orthogonal on the unit disk. These
polynomials are commonly used in optics and wavefront analysis.

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
    
    >>> import numpy.ma as ma
    >>> # Create a sample image with a mask
    >>> image_data = np.random.random((100, 100))
    >>> mask = np.zeros((100, 100), dtype=bool)
    >>> mask[30:70, 30:70] = True
    >>> masked_image = ma.masked_array(image_data, mask=mask)
    >>> # Define Zernike modes to be removed
    >>> zernike_modes = np.array([1, 2, 3, 4])
    >>> # Remove Zernike modes from the image
    >>> cleaned_image = zernike.removeZernike(masked_image, zernike_modes)
    >>> # Display the original and cleaned images
    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(1, 2, 1)
    >>> plt.title("Original Image")
    >>> plt.imshow(masked_image, cmap='gray')
    >>> plt.subplot(1, 2, 2)
    >>> plt.title("Cleaned Image")
    >>> plt.imshow(cleaned_image, cmap='gray')
    >>> plt.show()
"""

import numpy as _np
import math as _math
from . import geo as _geo

fac = _math.factorial


def removeZernike(ima, modes=_np.array([1, 2, 3, 4])):
    """
    Remove Zernike modes from an image.

    Parameters
    ----------
    ima : numpy masked array
        Image from which Zernike modes are to be removed.
    modes : numpy array, optional
        Zernike modes to be removed. Default is np.array([1, 2, 3, 4]).

    Returns
    -------
    new_ima : numpy masked array
        Image with Zernike modes removed.
    """
    coeff, mat = zernikeFit(ima, modes)
    surf = zernikeSurface(ima, coeff, mat)
    return ima - surf


def removeZernikeAuxMask(img, mm, zlist):
    """
    Remove Zernike modes from an image using an auxiliary mask.

    Parameters
    ----------
    img : numpy masked array
        Image from which Zernike modes are to be removed.
    mm : numpy array
        Auxiliary mask.
    zlist : numpy array
        List of Zernike modes to be removed.

    Returns
    -------
    new_ima : numpy masked array
        Image with Zernike modes removed.
    """
    coef, mat = zernikeFitAuxmask(img, mm, zlist)
    surf = zernikeSurface(img, coef, mat)
    return _np.ma.masked_array(img - surf, img.mask)


def zernikeFit(img, zernike_index_vector, qpupil: bool = True):
    """
    Fit Zernike modes to an image.

    Parameters
    ----------
    img : numpy masked array
        Image for Zernike fit.
    zernike_index_vector : numpy array
        Vector containing the index of Zernike modes to be fitted starting from 1.

    Returns
    -------
    coeff : numpy array
        Vector of Zernike coefficients.
    mat : numpy array
        Matrix of Zernike polynomials.
    """
    img1 = img.data
    mask = _np.invert(img.mask).astype(int)
    xx, yy = _geo.qpupil(mask) if qpupil else _geo.qpupil_circle(img)
    mm = mask == 1
    coeff = _surf_fit(xx[mm], yy[mm], img1[mm], zernike_index_vector)
    mat = _getZernike(xx[mm], yy[mm], zernike_index_vector)
    return coeff, mat


def zernikeFitAuxmask(img, auxmask, zernike_index_vector):
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
    img1 = img.data
    mask = _np.invert(img.mask).astype(int)
    xx, yy = _geo.qpupil(auxmask)
    mm = mask == 1
    coeff = _surf_fit(xx[mm], yy[mm], img1[mm], zernike_index_vector)
    mat = _getZernike(xx[mm], yy[mm], zernike_index_vector)
    return coeff, mat


def zernikeSurface(img, coef, mat):
    """
    Generate Zernike surface from coefficients and matrix.

    Parameters
    ----------
    img : numpy masked array
        Image for Zernike fit.
    coef : numpy array
        Vector of Zernike coefficients.
    mat : numpy array
        Matrix of Zernike polynomials.

    Returns
    -------
    surf : numpy masked array
        Zernike surface generated by coefficients.
    """
    mm = _np.where(img.mask == 0)
    zernike_surface = _np.zeros(img.shape)
    zernike_surface[mm] = _np.dot(mat, coef)
    return _np.ma.masked_array(zernike_surface, mask=img.mask)


def _surf_fit(xx, yy, zz, zlist, ordering="noll"):
    """
    Fit surface using Zernike polynomials.

    Parameters
    ----------
    xx, yy : numpy arrays
        Coordinates.
    zz : numpy array
        Surface data.
    zlist : numpy array
        List of Zernike modes.
    ordering : str, optional
        Ordering of Zernike modes. Default is 'noll'.

    Returns
    -------
    coeff : numpy array
        Zernike coefficients.
    """
    A = _getZernike(xx, yy, zlist, ordering)
    B = _np.transpose(zz.copy())
    return _np.linalg.lstsq(A, B, rcond=None)[0]


def _getZernike(xx, yy, zlist, ordering="noll"):
    """
    Get Zernike polynomials.

    Parameters
    ----------
    xx, yy : numpy arrays
        Coordinates.
    zlist : numpy array
        List of Zernike modes.
    ordering : str, optional
        Ordering of Zernike modes. Default is 'noll'.

    Returns
    -------
    zkm : numpy array
        Zernike polynomials.
    """
    if min(zlist) == 0:
        raise ValueError("Zernike index must be greater or equal to 1")
    rho = _np.sqrt(yy**2 + xx**2)
    phi = _np.arctan2(yy, xx)
    zkm = []
    for j in zlist:
        if ordering == "noll":
            m, n = _l2mn_noll(j)
            cnorm = _np.sqrt(n + 1) if m == 0 else _np.sqrt(2.0 * (n + 1))
        elif ordering == "ansi":
            m, n = _l2mn_ansi(j)
            cnorm = 1
        zkm.append(cnorm * _zernike(m, n, rho, phi))
    return _np.transpose(_np.array(zkm))


def _zernike_rad(m, n, rho):
    """
    Calculate the radial component of Zernike polynomial (m, n).

    Parameters
    ----------
    m, n : int
        Zernike polynomial indices.
    rho : numpy array
        Radial coordinates.

    Returns
    -------
    rad : numpy array
        Radial component of Zernike polynomial.
    """
    if n < 0 or m < 0 or abs(m) > n:
        raise ValueError("Invalid Zernike polynomial indices")
    if (n - m) % 2:
        return rho * 0.0
    pre_fac = (
        lambda k: (-1.0) ** k
        * fac(n - k)
        / (fac(k) * fac((n + m) // 2 - k) * fac((n - m) // 2 - k))
    )
    return sum(pre_fac(k) * rho ** (n - 2 * k) for k in range((n - m) // 2 + 1))


def _zernike(m, n, rho, phi):
    """
    Calculate Zernike polynomial (m, n).

    Parameters
    ----------
    m, n : int
        Zernike polynomial indices.
    rho, phi : numpy arrays
        Radial and azimuthal coordinates.

    Returns
    -------
    zernike : numpy array
        Zernike polynomial.
    """
    if m > 0:
        rad = _zernike_rad(m, n, rho)
        return rad * _np.cos(m * phi)
    if m < 0:
        rad = _zernike_rad(-m, n, rho)
        return rad * _np.sin(-m * phi)
    return _zernike_rad(0, n, rho)


def _zernikel(j, rho, phi):
    """
    Calculate Zernike polynomial with Null coordinate j.

    Parameters
    ----------
    j : int
        Null coordinate.
    rho, phi : numpy arrays
        Radial and azimuthal coordinates.

    Returns
    -------
    zernike : numpy array
        Zernike polynomial.
    """
    n = 0
    while j > n:
        n += 1
        j -= n
    m = -n + 2 * j
    return _zernike(m, n, rho, phi)


def _l2mn_ansi(j):
    """
    Convert ANSI index to Zernike polynomial indices.

    Parameters
    ----------
    j : int
        ANSI index.

    Returns
    -------
    m, n : int
        Zernike polynomial indices.
    """
    n = 0
    while j > n:
        n += 1
        j -= n
    m = -n + 2 * j
    return m, n


def _l2mn_noll(j):
    """
    Convert Noll index to Zernike polynomial indices.

    Parameters
    ----------
    j : int
        Noll index.

    Returns
    -------
    m, n : int
        Zernike polynomial indices.
    """
    n = int((-1.0 + _np.sqrt(8 * (j - 1) + 1)) / 2.0)
    p = j - (n * (n + 1)) / 2
    k = n % 2
    m = int((p + k) / 2.0) * 2 - k
    if m != 0:
        m *= 1 if j % 2 == 0 else -1
    return m, n
