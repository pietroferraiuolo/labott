# -*- coding: utf-8 -*-
"""
Autor(s)
---------
    - Federico Miceli : added funcitonality on march 2022
    - Runa Briguglio : created Mar 2020
    - Pietro Ferraiuolo : polished on 2024 for this Alignment Software

Description
-----------
This module contains functions for geometric operations on images.
"""
import numpy as np
from skimage.measure import CircleModel
from skimage.draw import disk


def qpupil_circle(image, pixel_dir=0):
    """
    Function for...
    Created by Federico
    NOTA: la funzione usa come standard la direzione y per determinare la dimensione dei pixel

    pixel_dir: int
        indicates which direction to use for counting the number of pixels in the image.
        Y direction as standard
    """
    aa = np.shape(image)
    imagePixels = aa[pixel_dir]  # standard dir y
    circ = CircleModel()
    cnt = _find_img_borders(image, imagePixels)
    circ.estimate(cnt)
    xc, yc, radius = np.array(circ.params, dtype=int)
    maskedd = np.zeros((imagePixels, imagePixels), dtype=np.uint8)
    rr, cc = disk((xc, yc), int(radius))
    maskedd[rr, cc] = 1
    idx = np.where(maskedd == 1)
    ss = np.shape(maskedd)
    x = np.arange(ss[0]).astype(float)
    x = np.transpose(np.tile(x, [ss[1], 1]))
    y = np.arange(ss[1]).astype(float)
    y = np.tile(y, [ss[0], 1])
    xx = x
    yy = y
    xx = xx - xc
    xx = xx / radius
    yy = yy - yc
    yy = yy / radius
    return xx, yy


def qpupil(mask, xx=None, yy=None, nocircle=0):
    """
    Function for....
    created by Runa

    Parameters
    ----------
    mask: numpy array

    Returns
    ------
    x0:
    y0:
    r:
    xx: numpy array
        grid of coordinates of the same size as input mask
    yy: numpy array
        grid of coordinates of the same size as input mask
    """
    idx = np.where(mask == 1)
    ss = np.shape(mask)
    x = np.arange(ss[0]).astype(float)
    x = np.transpose(np.tile(x, [ss[1], 1]))
    y = np.arange(ss[1]).astype(float)
    y = np.tile(y, [ss[0], 1])
    xx = x
    yy = y
    x0 = 0
    y0 = 0
    r = 0
    if nocircle == 0:
        maxv = max(xx[idx])
        minv = min(xx[idx])
        r1 = (maxv - minv) / 2
        x0 = r1 + minv
        xx = xx - (minv + maxv) / 2
        xx = xx / ((maxv - minv) / 2)
        mx = [minv, maxv]
        maxv = max(yy[idx])
        minv = min(yy[idx])
        r2 = (maxv - minv) / 2
        y0 = r2 + minv
        yy = yy - (minv + maxv) / 2
        yy = yy / ((maxv - minv) / 2)
        r = np.mean([r1, r2])
        my = [minv, maxv]
    return xx, yy


def _find_img_borders(image, imagePixels):
    """
    Function for...
    Created by Federico
    """
    x = image
    val = []
    i = 0
    while i < imagePixels:
        a = x[i, :]
        aa = np.where(a.mask.astype(int) == 0)
        q = np.asarray(aa)
        if q.size < 2:
            i = i + 1
        else:
            val.append(np.array([[i, q[0, 0]], [i, q[0, q.size - 1]]]))
            i = i + 1
    cut = np.concatenate(val)
    return cut
