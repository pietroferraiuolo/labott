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
from . import geo as _geo
from opticalib import typings as _t

fac = _math.factorial


def generateZernMat(noll_ids: list[int], img_mask: _t.ImageData, scale_length:float = None) -> _t.MatrixLike:
    """
    Generates the interaction matrix of the Zernike modes with Noll index
    in noll_ids on the mask in input

    Parameters
    ----------
    noll_ids : ArrayLike
        List of (Noll) mode indices to fit.
    img_mask : matrix bool
        Mask of the desired image.
    scale_length : float, optional
        The scale length to use for the Zernike fit.
        The default is the maximum of the image mask shape.

    Returns
    -------
    ZernMat : MatrixLike [n_pix,n_zern]
        The Zernike interaction matrix of the given indices on the given mask.
    """
    n_pix = int(_np.sum(1-img_mask))
    n_zern = len(noll_ids)
    ZernMat = _np.zeros([n_pix,n_zern])
    for i in range(n_zern):
        ZernMat[:,i] = _project_zernike_on_mask(noll_ids[i], img_mask, scale_length)
    return ZernMat


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
    coef, mat = zernikeFitAuxmask(image, mask, zlist)
    surf = zernikeSurface(image, coef, mat)
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
    xx, yy = _geo.qpupil(mask) if qpupil else _geo.qpupil_circle(image)
    mm = mask == 1
    coeff = _surf_fit(xx[mm], yy[mm], img1[mm], zernike_index_vector)
    mat = _getZernike(xx[mm], yy[mm], zernike_index_vector)
    return coeff, mat


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
    img1 = image.data
    mask = _np.invert(image.mask).astype(int)
    xx, yy = _geo.qpupil(auxmask)
    mm = mask == 1
    coeff = _surf_fit(xx[mm], yy[mm], img1[mm], zernike_index_vector)
    mat = _getZernike(xx[mm], yy[mm], zernike_index_vector)
    return coeff, mat


def zernikeSurface(
    image: _t.ImageData, coef: _t.ArrayLike, mat: _t.ArrayLike
) -> _t.ImageData:
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
    mm = _np.where(image.mask == 0)
    zernike_surface = _np.zeros(image.shape)
    zernike_surface[mm] = _np.dot(mat, coef)
    return _np.ma.masked_array(zernike_surface, mask=image.mask)


def _surf_fit(
    xx: _t.ArrayLike,
    yy: _t.ArrayLike,
    zz: _t.ArrayLike,
    zlist: list[int],
    ordering: str = "noll",
) -> _t.ArrayLike:
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


def _getZernike(
    xx: _t.ArrayLike, yy: _t.ArrayLike, zlist: list[int], ordering: str = "noll"
) -> _t.ArrayLike:
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


def _project_zernike_on_mask(noll_number:int, mask: _t.ImageData, scale_length: float = None) -> _t.ArrayLike:
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
    X,Y = _np.shape(mask)
    # Determine circle radius on to which define the Zernike
    if scale_length is not None:
        r = scale_length
    else:
        r = _np.max([X,Y])/2
    # Conversion to polar coordinates on circle of radius r 
    phi = lambda i,j: _np.arctan2((j-Y/2.)/r,(i-X/2.)/r)
    rho = lambda i,j: _np.sqrt(((j-Y/2.)/r)**2+((i-X/2.)/r)**2)
    mode = _np.fromfunction(lambda i,j: _zernikel(noll_number, rho(i,j), phi(i,j)), [X,Y])
    #masked_data = mode[1-mask]
    masked_data = mode.flatten()[mask.flatten()<1]
    # Normalization of the masked data: null mean and unit STD
    if noll_number > 1:
        masked_data = (masked_data - _np.mean(masked_data))/_np.std(masked_data)
    return masked_data


def _zernike_rad(m: int, n: int, rho: _t.ArrayLike) -> _t.ArrayLike:
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


def _zernike(m: int, n: int, rho: _t.ArrayLike, phi: _t.ArrayLike) -> _t.ArrayLike:
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


def _zernikel(j: int, rho: _t.ArrayLike, phi: _t.ArrayLike) -> _t.ArrayLike:
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


def _l2mn_ansi(j: int) -> tuple[int, int]:
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


def _l2mn_noll(j: int) -> tuple[int, int]:
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
