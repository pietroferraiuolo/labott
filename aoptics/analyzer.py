"""
Author(s)
---------
- Runa Briguglio: created 2020
- Pietro Ferraiuolo: modified 2024

Description
-----------

"""

import os as _os
import numpy as _np
import jdcal as _jdcal
import matplotlib.pyplot as _plt
from .ground import zernike as zern
from .ground import osutils as osu
from .core import root as _foldname
from .ground.geo import qpupil as _qpupil
from .ground.osutils import InterferometerConverter
from scipy import stats as _stats, fft as _fft, ndimage as _ndimage

_OPDIMG = _foldname.OPD_IMAGES_ROOT_FOLDER
_OPDSER = _foldname.OPD_SERIES_ROOT_FOLDER


def averageFrames(
    tn: str,
    first: int = None,
    last: int = None,
    file_selector: list = None,
    thresh: bool = False,
):
    """
    Perform the average of a list of images, retrievable through a tracking
    number.

    Parameters
    ----------
    tn : str
        Data Tracking Number.
    first : int, optional
        Index number of the first file to consider. If None, the first file in
        the list is considered.
    last : int, optional
        Index number of the last file to consider. If None, the last file in
        list is considered.
    file_selector : list, optional
        A list of integers, representing the specific files to load. If None,
        the range (first->last) is considered.
    thresh : bool, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    aveimg : ndarray
        Final image of averaged frames.

    """
    fileList = osu.getFileList(tn, fold=_OPDSER, key="20")
    if first is not None and last is not None:
        fl = [
            fileList[x]
            for x in _np.arange(first, last, 1)
            if file_selector is None or x in file_selector
        ]
    else:
        first = 0
        last = len(fileList)
        fl = [
            fileList[x]
            for x in _np.arange(first, last, 1)
            if file_selector is None or x in file_selector
        ]
    imcube = createCube(fl)
    if thresh is False:
        aveimg = _np.ma.mean(imcube, axis=2)
    else:
        img = imcube[:, :, 0].data * 0
        mmask = imcube[:, :, 0].mask
        nn = 0
        for j in range(imcube.shape[2]):
            im = imcube[:, :, j]
            size = im.data.compressed.size
            if size > 1:
                nn += 1
                img += im.data
                mmask = _np.ma.mask_or(im.mask, mmask)
        img = img / nn
        aveimg = _np.ma.masked_array(img, mask=mmask)
    return aveimg


def saveAverage(tn, average_img=None, overwrite: bool = False, **kwargs):
    """
    Saves an averaged frame, in the same folder as the original frames. If no
    averaged image is passed as argument, it will create a new average for the
    specified tracking number, and additional arguments, the same as ''averageFrames''
    can be specified.

    Parameters
    ----------
    tn : str
        Tracking number where to save the average frame file. If average_img is
        None, it is the tracking number of the data that will be averaged
    average_img : ndarray, optional
        Result average image of multiple frames. If it's None, it will be generated
        from data found in the tracking number folder. Additional arguments can
        be passed on
    **kwargs : additional optional arguments
        The same arguments as ''averageFrames'', to specify the averaging method.

        tn : str
            Data Tracking Number.
        first : int, optional
            Index number of the first file to consider. If None, the first file in
            the list is considered.
        last : int, optional
            Index number of the last file to consider. If None, the last file in
            list is considered.
        file_selector : list, optional
            A list of integers, representing the specific files to load. If None,
            the range (first->last) is considered.
        thresh : bool, optional
            DESCRIPTION. The default is None.
    """
    fname = _os.path.join(_OPDSER, tn, "average.fits")
    if _os.path.isfile(fname):
        print(f"Average '{fname}' already exists")
    else:
        if average_img is None:
            first = kwargs.get("first", None)
            last = kwargs.get("last", None)
            fsel = kwargs.get("file_selector", None)
            thresh = kwargs.get("tresh", False)
            average_img = averageFrames(
                tn, first=first, last=last, file_selector=fsel, thresh=thresh
            )
        osu.save_fits(fname, average_img, overwrite=overwrite)
        print(f"Saved average at '{fname}'")


def openAverage(tn):
    """
    Loads an averaged frame from an 'average.fits' file, found inside the input
    tracking number

    Parameters
    ----------
    tn : str
        Tracking number of the averaged frame.

    Returns
    -------
    image : ndarray
        Averaged image.

    Raises
    ------
    FileNotFoundError
        Raised if the file does not exist.
    """
    fname = _os.path.join(_OPDSER, tn, "average.fits")
    try:
        image = osu.load_fits(fname)
        print(f"Average loaded: '{fname}'")
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Average file '{fname}' does not exist!") from err
    return image


def runningDiff(tn, gap=2):
    """


    Parameters
    ----------
    tn : TYPE
        DESCRIPTION.
    gap : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    svec : TYPE
        DESCRIPTION.

    """
    llist = osu.getFileList(tn)
    nfile = len(llist)
    npoints = int(nfile / gap) - 2
    slist = []
    for i in range(0, npoints):
        q0 = frame(i * gap, llist)
        q1 = frame(i * gap + 1, llist)
        diff = q1 - q0
        diff = zern.removeZernike(diff)
        slist.append(diff.std())
    svec = _np.array(slist)
    return svec


def frame(idx, mylist):
    """


    Parameters
    ----------
    id : TYPE
        DESCRIPTION.
    mylist : TYPE
        DESCRIPTION.

    Returns
    -------
    img : TYPE
        DESCRIPTION.

    """
    mytype = type(mylist)
    if mytype is list:
        img = osu.read_phasemap(mylist[idx])
    if mytype is _np.ma.core.MaskedArray:
        img = mylist[idx]
    return img


def spectrum(signal, dt=1, show=None):
    """


    Parameters
    ----------
    signal : ndarray
        DESCRIPTION.
    dt : float, optional
        DESCRIPTION. The default is 1.
    show : bool, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    spe : float | ndarray
        DESCRIPTION.
    freq : float | ArrayLike
        DESCRIPTION.

    """
    # https://numpy.org/doc/stable/reference/generated/numpy.angle - Spectrum phase
    nsig = signal.shape
    if _np.size(nsig) == 1:
        thedim = 0
    else:
        thedim = 1
    if _np.size(nsig) == 1:
        spe = _np._fft.rfft(signal, norm="ortho")
        nn = _np.sqrt(spe.shape[thedim])  # modRB
    else:
        spe = _np._fft.rfft(signal, axis=1, norm="ortho")
        nn = _np.sqrt(spe.shape[thedim])  # modRB
    spe = (_np.abs(spe)) / nn
    freq = _np._fft.rfftfreq(signal.shape[thedim], d=dt)
    if _np.size(nsig) == 1:
        spe[0] = 0
    else:
        spe[:, 0] = 0
    if show is not None:
        _plt.figure()
        for i in range(0, len(spe)):
            _plt.plot(freq, spe[i, :], label=f"Channel {i}")
        _plt.xlabel(r"Frequency [$Hz$]")
        _plt.ylabel("PS Amplitude")
        _plt.legend(loc="best")
        _plt.show()
    return spe, freq


def frame2ottFrame(img, croppar, flipOffset=True):
    """


    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    croppar : TYPE
        DESCRIPTION.
    flipOffset : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    fullimg : TYPE
        DESCRIPTION.

    """
    off = croppar.copy()
    if flipOffset is True:
        off = _np.flip(croppar)
        print(f"Offset values flipped: {str(off)}")
    nfullpix = _np.array([2048, 2048])
    fullimg = _np.zeros(nfullpix)
    fullmask = _np.ones(nfullpix)
    offx = off[0]
    offy = off[1]
    sx = _np.shape(img)[0]  # croppar[2]
    sy = _np.shape(img)[1]  # croppar[3]
    fullimg[offx : offx + sx, offy : offy + sy] = img.data
    fullmask[offx : offx + sx, offy : offy + sy] = img.mask
    fullimg = _np.ma.masked_array(fullimg, fullmask)
    return fullimg


def timevec(tn):
    """


    Parameters
    ----------
    tn : TYPE
        DESCRIPTION.

    Returns
    -------
    timevector : TYPE
        DESCRIPTION.

    """
    fold = osu.findTracknum(tn)
    flist = osu.getFileList(tn)
    nfile = len(flist)
    if "OPDImages" in fold:
        tspace = 1.0 / 28.57
        timevector = range(nfile) * tspace
    elif "OPD_series" in fold:
        timevector = []
        for i in flist:
            pp = i.split(".")[0]
            tni = pp.split("/")[-1]
            y = tni[0:4]
            mo = tni[4:6]
            d = tni[6:8]
            h = float(tni[9:11])
            mi = float(tni[11:13])
            s = float(tni[13:15])
            jdi = sum(_jdcal.gcal2jd(y, mo, d)) + h / 24 + mi / 1440 + s / 86400
            timevector.append(jdi)
        timevector = _np.array(timevec)
    return timevector


def track2jd(tni):
    """


    Parameters
    ----------
    tni : TYPE
        DESCRIPTION.

    Returns
    -------
    jdi : TYPE
        DESCRIPTION.

    """
    t = track2date(tni)
    jdi = sum(_jdcal.gcal2jd(t[0], t[1], t[2])) + t[3] / 24 + t[4] / 1440 + t[5] / 86400
    return jdi


def track2date(tni):
    """
    Converts a tracing number into a list containing year, month, day, hour,
    minutes and seconds, divied.

    Parameters
    ----------
    tni : str
        Tracking number to be converted.

    Returns
    -------
    time : list
        List containing the date element by element.
        [0] y : str
            Year.
        [1] mo : str
            Month.
        [2] d : str
            Day.
        [3] h : float
            Hour.
        [4] mi : float
            Minutes.
        [5] s : float
            Seconds.
    """
    y = tni[0:4]
    mo = tni[4:6]
    d = tni[6:8]
    h = float(tni[9:11])
    mi = float(tni[11:13])
    s = float(tni[13:15])
    time = [y, mo, d, h, mi, s]
    return time


def runningMean(vec, npoints):
    """


    Parameters
    ----------
    vec : TYPE
        DESCRIPTION.
    npoints : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return _np.convolve(vec, _np.ones(npoints), "valid") / npoints


def readTemperatures(tn):
    """


    Parameters
    ----------
    tn : TYPE
        DESCRIPTION.

    Returns
    -------
    temperatures : TYPE
        DESCRIPTION.

    """
    fold = osu.findTracknum(tn, complete_path=True)
    fname = _os.path.join(fold, tn, "temperature.fits")
    temperatures = osu.load_fits(fname)
    return temperatures


def readZernike(tn):
    """


    Parameters
    ----------
    tn : TYPE
        DESCRIPTION.

    Returns
    -------
    temperatures : TYPE
        DESCRIPTION.

    """
    fold = osu.findTracknum(tn, complete_path=True)
    fname = _os.path.join(fold, tn, "zernike.fits")
    zernikes = osu.load_fits(fname)
    return zernikes


def zernikePlot(mylist, modes=_np.array(range(1, 11))):
    """


    Parameters
    ----------
    mylist : TYPE
        DESCRIPTION.
    modes : TYPE, optional
        DESCRIPTION. The default is _np.array(range(1, 11)).

    Returns
    -------
    zcoeff : TYPE
        DESCRIPTION.

    """
    mytype = type(mylist)
    if mytype is list:
        imgcube = createCube(mylist)
    if mytype is _np.ma.core.MaskedArray:
        imgcube = mylist
    zlist = []
    for i in range(imgcube.shape[-1]):
        print(i)
        coeff, _ = zern.zernikeFit(imgcube[:,:,i], modes)
        zlist.append(coeff)
    zcoeff = _np.array(zlist)
    zcoeff = zcoeff.T
    return zcoeff


def strfunct(vect, gapvect):
    """
    vect shall be npoints x m
    the strfunct is calculate m times over the npoints time series
    returns stf(n_timeseries x ngaps)
    """
    nn = _np.shape(vect)
    maxgap = _np.max(gapvect)
    ngap = len(gapvect)
    n2ave = int(nn / (maxgap)) - 1  # or -maxgap??
    jump = maxgap
    st = _np.zeros(ngap)
    for j in range(ngap):
        tx = []
        for k in range(n2ave):
            print("Using positions:")
            print(k * jump, k * jump + gapvect[j])
            tx.append((vect[k * jump] - vect[k * jump + gapvect[j]]) ** 2)
        st[j] = _np.mean(_np.sqrt(tx))
    return st


def comp_filtered_image(imgin, verbose=False, disp=False, d=1, freq2filter=None):
    """


    Parameters
    ----------
    imgin : TYPE
        DESCRIPTION.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.
    disp : TYPE, optional
        DESCRIPTION. The default is False.
    d : TYPE, optional
        DESCRIPTION. The default is 1.
    freq2filter : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    imgout : TYPE
        DESCRIPTION.
    """
    img = imgin.copy()
    sx = (_np.shape(img))[0]
    mask = _np.invert(img.mask)
    img[mask == 0] = 0
    norm = "ortho"
    tf2d = _fft.fft2(img.data, norm=norm)
    kfreq = _np._fft.fftfreq(sx, d=d)  # frequency in cicles
    kfreq2D = _np.meshgrid(kfreq, kfreq)  # frequency grid x,y
    knrm = _np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)  # freq. grid distance
    # TODO optional mask to get the circle and not the square
    fmask1 = 1.0 * (knrm > _np.max(kfreq))
    if freq2filter is None:
        fmin = -1
        fmax = _np.max(kfreq)
    else:
        fmin, fmax = freq2filter
    fmask2 = 1.0 * (knrm > fmax)
    fmask3 = 1.0 * (knrm < fmin)
    fmask = (fmask1 + fmask2 + fmask3) > 0
    tf2d_filtered = tf2d.copy()
    tf2d_filtered[fmask] = 0
    imgf = _fft.ifft2(tf2d_filtered, norm=norm)
    imgout = _np.ma.masked_array(_np.real(imgf), mask=imgin.mask)
    if disp:
        _plt.figure()
        _plt.imshow(knrm)
        _plt.title("freq")
        _plt.figure()
        _plt.imshow(fmask1)
        _plt.title("fmask1")
        _plt.figure()
        _plt.imshow(fmask2)
        _plt.title("fmask2")
        _plt.figure()
        _plt.imshow(fmask3)
        _plt.title("fmask3")
        _plt.figure()
        _plt.imshow(fmask)
        _plt.title("fmask")
        _plt.figure()
        _plt.imshow(_np.abs(tf2d))
        _plt.title("Initial spectrum")
        _plt.figure()
        _plt.imshow(_np.abs(tf2d_filtered))
        _plt.title("Filtered spectrum")
        _plt.figure()
        _plt.imshow(imgin)
        _plt.title("Initial image")
        _plt.figure()
        _plt.imshow(imgout)
        _plt.title("Filtered image")
    e1 = _np.sqrt(_np.sum(img[mask] ** 2) / _np.sum(mask)) * 1e9
    e2 = _np.sqrt(_np.sum(imgout[mask] ** 2) / _np.sum(mask)) * 1e9
    e3 = _np.sqrt(_np.sum(_np.abs(tf2d) ** 2) / _np.sum(mask)) * 1e9
    e4 = _np.sqrt(_np.sum(_np.abs(tf2d_filtered) ** 2) / _np.sum(mask)) * 1e9
    if verbose:
        print(f"RMS image [nm]            {e1:.2f}")
        print(f"RMS image filtered [nm]   {e2:.2f}")
        print(f"RMS spectrum              {e3:.2f}")
        print(f"RMS spectrum filtered     {e4:.2f}")
    return imgout


def comp_psd(
    imgin,
    nbins=None,
    norm="backward",
    verbose=False,
    disp=False,
    d=1,
    sigma=None,
    crop=True,
):
    """


    Parameters
    ----------
    imgin : TYPE
        DESCRIPTION.
    nbins : TYPE, optional
        DESCRIPTION. The default is None.
    norm : TYPE, optional
        DESCRIPTION. The default is "backward".
    verbose : TYPE, optional
        DESCRIPTION. The default is False.
    disp : TYPE, optional
        DESCRIPTION. The default is False.
    d : TYPE, optional
        DESCRIPTION. The default is 1.
    sigma : TYPE, optional
        DESCRIPTION. The default is None.
    crop : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    fout : TYPE
        DESCRIPTION.
    Aout : TYPE
        DESCRIPTION.

    """
    if crop:
        cir = _qpupil(-1 * imgin.mask + 1)
        cir = _np.array(cir[0:3]).astype(int)
        img = imgin.data[
            cir[0] - cir[2] : cir[0] + cir[2], cir[1] - cir[2] : cir[1] + cir[2]
        ]
        m = imgin.mask[
            cir[0] - cir[2] : cir[0] + cir[2], cir[1] - cir[2] : cir[1] + cir[2]
        ]
        img = _np.ma.masked_array(img, m)
    else:
        img = imgin.copy()
    sx = (_np.shape(img))[0]
    if nbins is None:
        nbins = sx // 2
    img = img - _np.mean(img)
    mask = _np.invert(img.mask)
    img[mask == 0] = 0
    if sigma is not None:
        img = _ndimage.fourier_gaussian(img, sigma=sigma)
    tf2d = _fft.fft2(img, norm=norm)
    tf2d[0, 0] = 0
    tf2d_power_spectrum = _np.abs(tf2d) ** 2
    kfreq = _np._fft.fftfreq(sx, d=d)  # frequency in cicles
    kfreq2D = _np.meshgrid(kfreq, kfreq)  # freq. grid
    knrm = _np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)  # freq. grid distance
    fmask = knrm < _np.max(kfreq)
    knrm = knrm[fmask].flatten()
    fourier_amplitudes = tf2d_power_spectrum[fmask].flatten()
    Abins, _, _ = _stats.binned_statistic(
        knrm, fourier_amplitudes, statistic="sum", bins=nbins
    )
    e1 = _np.sum(img[mask] ** 2 / _np.sum(mask))
    e2 = _np.sum(Abins) / _np.sum(mask)
    ediff = _np.abs(e2 - e1) / e1
    fout = kfreq[0 : sx // 2]
    Aout = Abins / _np.sum(mask)
    if verbose:
        print(f"Sampling          {d:}")
        print(f"Energy signal     {e1}")
        print(f"Energy spectrum   {e2}")
        print(f"Energy difference {ediff}")
        print(f"RMS from spectrum {_np.sqrt(e2)}")
        print(f"RMS [nm]          {(_np.std(img[mask])*1e9):.2f}")
        print(kfreq[0:4])
        print(kfreq[-4:])
    else:
        print(f"RMS from spectrum {_np.sqrt(e2)}")
        print(f"RMS [nm]          {(_np.std(img[mask])*1e9):.2f}")
    if disp is True:
        _plt.figure()
        _plt.plot(fout[1:], Aout[1:] * fout[1:], ".")
        _plt.yscale("log")
        _plt.xscale("log")
        _plt.title("Power spectrum")
        _plt.xlabel("Frequency [Hz]")
        _plt.ylabel("Amplitude [A^2]")
    return fout, Aout


def integrate_psd(y, img):
    nn = _np.sqrt(_np.sum(-1 * img.mask + 1))
    yint = _np.sqrt(_np.cumsum(y)) / nn
    return yint


def getDataFileList(tn):
    pass


def createCube(filelist, register=False):
    """
    Creates a cube of images from an images file list

    Parameters
    ----------
    filelist : list of str
        List of file paths to the images/frames to be stacked into a cube.
    register : int or tuple, optional
        If not False, and int or a tuple of int must be passed as value, and
        the registration algorithm is performed on the images before stacking them
        into the cube. Default is False.

    Returns
    -------
    cube : ndarray
        Data cube containing the images/frames stacked.
    """
    cube_list = []
    for imgfits in filelist:
        image = osu.read_phasemap(imgfits)
        if register:
            image = _np.roll(image, register)
        cube_list.append(image)
    cube = _np.ma.dstack(cube_list)
    return cube


def modeRebinner(img, rebin):
    """
    Image rebinner

    Rebins a masked array image by a factor rebin.

    Parameters
    ----------
    img : masked_array
        Image to rebin.
    rebin : int
        Rebinning factor.

    Returns
    -------
    newImg : masked_array
        Rebinned image.
    """
    shape = img.shape
    new_shape = (shape[0] // rebin, shape[1] // rebin)
    newImg = _rebin2DArray(img, new_shape)
    return newImg

def cubeRebinner(cube, rebin):
    """
    Cube rebinner

    Parameters
    ----------
    cube : ndarray
        Cube to rebin.
    rebin : int
        Rebinning factor.
    
    Returns
    -------
    newCube : ndarray
        Rebinned cube.
    """
    newCube = []
    for i in range(cube.shape[-1]):
        newCube.append(modeRebinner(cube[:,:,i], rebin))
    return _np.ma.dstack(newCube)

# From ARTE #
def _rebin2DArray(a, new_shape, sample=False):
    """
    Replacement of IDL's rebin() function for 2d arrays.
    Resizes a 2d array by averaging or repeating elements.
    New dimensions must be integral factors of original dimensions,
    otherwise a ValueError exception will be raised.
    Parameters
    ----------
    a : ndarray
        Input array.
    new_shape : 2-elements sequence
        Shape of the output array
    sample : bool
        if True, when reducing the array side elements are set
        using a nearest-neighbor algorithm instead of averaging.
        This parameter has no effect when enlarging the array.
    Returns
    -------
    rebinned_array : ndarray
        If the new shape is smaller of the input array  the data are averaged,
        unless the sample parameter is set.
        If the new shape is bigger array elements are repeated.
    Raises
    ------
    ValueError
        in the following cases:
         - new_shape is not a sequence of 2 values that can be converted to int
         - new dimensions are not an integral factor of original dimensions
    NotImplementedError
         - one dimension requires an upsampling while the other requires
           a downsampling
    Examples
    --------
    >>> a = np.array([[0, 1], [2, 3]])
    >>> b = rebin(a, (4, 6)) #upsize
    >>> b
    array([[0, 0, 0, 1, 1, 1],
           [0, 0, 0, 1, 1, 1],
           [2, 2, 2, 3, 3, 3],
           [2, 2, 2, 3, 3, 3]])
    >>> rebin(b, (2, 3)) #downsize
    array([[0. , 0.5, 1. ],
           [2. , 2.5, 3. ]])
    >>> rebin(b, (2, 3), sample=True) #downsize
    array([[0, 0, 1],
           [2, 2, 3]])
    """

    # unpack early to allow any 2-length type for new_shape
    m, n = map(int, new_shape)

    if a.shape == (m, n):
        return a

    M, N = a.shape

    if m <= M and n <= M:
        if (M // m != M / m) or (N // n != N / n):
            raise ValueError("Cannot downsample by non-integer factors")

    elif M <= m and M <= m:
        if (m // M != m / M) or (n // N != n / N):
            raise ValueError("Cannot upsample by non-integer factors")

    else:
        raise NotImplementedError(
            "Up- and down-sampling in different axes " "is not supported"
        )

    if sample:
        slices = [slice(0, old, float(old) / new) for old, new in zip(a.shape, (m, n))]
        idx = _np.mgrid[slices].astype(int)
        return a[tuple(idx)]
    else:
        if m <= M and n <= N:
            return a.reshape((m, M // m, n, N // n)).mean(3).mean(1)
        elif m >= M and n >= M:
            return _np.repeat(_np.repeat(a, m / M, axis=0), n / N, axis=1)
