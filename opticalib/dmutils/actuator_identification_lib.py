import numpy as np

# from m4.ground import geo
from photutils.centroids import centroid_2dg

# from m4.utils import image_registration_lib as imgreg

center_act = 313


def findFrameCoord(imglist, actlist, actcoord):
    """
    returns the position of given actuators from a list of frames
    """
    pos = []
    for i in imglist:
        pos.append(findActuator(i))
    pos = (np.array(pos)).T

    frameCenter = imgreg.marker_general_remap(
        actcoord[:, actlist], pos, actcoord[:, (center_act, center_act)]
    )
    # the last variable has been vectorized (by adding a second element) don't know why but so it works
    frameCenter = frameCenter[:, 0]
    return frameCenter


def findActuator(img):
    """
    Finds the coordinates of an actuator, given the image with the InfFunction masked around the act.
    img: masked array
        image where the act is to be searched
    Return
    imgout: array
        coordinates of the act
    """
    imgw = extractPeak(img, radius=50)
    pos = centroid_2dg(imgw)
    return pos


def extractPeak(img, radius=50):
    """
    Extract a circular area around the peak in the image
    """
    yp, xp = np.where(img == np.max(abs(img)))
    img1 = img * np.invert(img.mask)
    m = np.invert(geo.draw_mask(img.mask, yp, xp, radius))
    imgout = np.ma.masked_array(img1, m)
    return imgout


def combineMasks(imglist):  #!!! deve sparire
    """
    combine masks layers of masked arrays, or a list of masks, to produce the intersection masks: not masked here AND not mnaked there
    masks are expected as in the np.ma convention: True when not masked
    return:
        intersection mask

    """
    imglistIsMaskedArray = True
    imglistIsmasksList = False
    mm = []
    for i in imglist:
        if imglistIsMaskedArray:
            mm.append(np.invert(i.mask).astype(int))
        if imglistIsmasksList:
            mm.append(np.invert(i).astype(int))
    mmask = product(mm, 0)
    return mmask


# from imgreg and from parabolaFootprintRegistration
def marker_general_remap(cghf, ottf, pos2t):
    """
    transforms the pos2t coordinates, using the cghf and ottf coordinates to create the trnasformation
    """
    polycoeff = pfr.fit_trasformation_parameter(cghf, ottf)
    base_cgh = pfr._expandbase(pos2t[0, :], pos2t[1, :])
    cghf_tra = np.transpose(np.dot(np.transpose(base_cgh), np.transpose(polycoeff)))
    return cghf_tra
