"""
Author(s):
----------
- Chiara Selmi: written in 2019 | rewritten in 2022
- Pietro Ferraiuolo: modified in 2024
"""

import numpy as _np
from skimage import measure as _meas
from opticalib import typings as _ot


def roiGenerator(img: _ot.ImageData, n_masks: int = 2) -> list[_ot.ImageData]:
    """
    This function generates a list of `n_masks` roi from the input image.

    Parameters
    ----------
    img: ImageData | np.ma.maskedArray
        input image from which the roi are generated.

    Returns
    -------
    roiList: list
        List of the first `n_masks` roi found in the image.
    """
    labels = _meas.label(_np.invert(img.mask))
    roiList = []
    for i in range(1, n_masks + 1):
        maski = _np.zeros(labels.shape, dtype=bool)
        maski[_np.where(labels == i)] = 1
        final_roi = _np.ma.mask_or(_np.invert(maski), img.mask)
        roiList.append(final_roi)
    return roiList


def imgCut(img: _ot.ImageData):
    """
    Cuts the image to the bounding box of the finite (non-NaN) pixels in the masked image.

    Parameters
    ----------
    image : np.ma.maskedArray
        The original masked image array.

    Returns
    -------
    cutImg = np.ma.maskedArray
        The cut image within the bounding box of finite pixels.
    """
    # Find indices of finite (non-NaN) pixels
    finite_coords = _np.argwhere(_np.isfinite(img))
    # If there are no finite pixels, return the original image
    if finite_coords.size == 0:
        return img
    top_left = finite_coords.min(axis=0)
    bottom_right = finite_coords.max(axis=0)
    cutImg = img[top_left[0] : bottom_right[0] + 1, top_left[1] : bottom_right[1] + 1]
    return cutImg
