import os as _os
import numpy as _np
import time as _time
import shutil as _sh
from labott.ground import logger as _logger
from labott.ground.osutils import (
    newtn as _newtn,
    InterferometerConverter,
    rename4D,
)
from labott.core.root import folders as _folds
from labott.analyzer import modeRebinner as _modeRebinner

_confReader = _folds.ConfSettingReader4D

class PhaseCam():
    """
    Class for the 4D Twyman-Green PhaseCam Laser Interferometer.
    """

    def __init__(self, ip: str = None, port: int = None):
        """The constructor"""
        from labott.core.i4d import I4D

        if (ip and port) is None:
            from labott.core.root import I4D_IP, I4D_PORT
            ip = I4D_IP
            port = I4D_PORT

        self._i4d = I4D(ip, port)
        self._ic = InterferometerConverter()
        self._logger = _logger.set_up_logger(_folds.LOGGING_FILE_PATH, 20)
        self._ts = _newtn

    def acquire_map(self, nframes=1, delay=0, rebin: int = 1):
        """
        Acquires the interferometer image and returns it as a masked array.

        Parameters
        ----------
        nframes: int
            Number of frames to be averaged that produce the measurement.
        delay: int
            Delay between images in seconds.
        rebin: int
            Rebin factor for the image.

        Returns
        -------
        masked_ima: numpy masked array
            Interferometer image.
        """
        if nframes == 1:
            width, height, _, data_array = self._i4d.takeSingleMeasurement()
            masked_ima = self._fromDataArrayToMaskedArray(
                width, height, data_array * 632.8e-9
            )
            masked_ima = _modeRebinner(masked_ima, rebin)
        else:
            image_list = []
            for i in range(nframes):
                width, height, _, data_array = self._i4d.takeSingleMeasurement()
                masked_ima = self._fromDataArrayToMaskedArray(
                    width, height, data_array * 632.8e-9
                )
                image_list.append(masked_ima)
                _time.sleep(delay)
            images = _np.ma.dstack(image_list)
            masked_ima = _np.ma.mean(images, 2)
            masked_ima = _modeRebinner(masked_ima, rebin)
        return masked_ima

    def acquire_detector(self, nframes=1, delay=0):
        """
        Parameters
        ----------
            nframes: int
                number of frames
            delay: int [s]
                delay between images

        Returns
        -------
            data2d: numpy masked array
                    detector interferometer image
        """
        self.acquire_phasemap()
        if nframes == 1:
            data, height, _, width = self._i4d.getFringeAmplitudeData()
            data2d = _np.reshape(data, (width, height))
        else:
            image_list = []
            for i in range(nframes):
                data, height, _, width = self._i4d.getFringeAmplitudeData()
                data2d_t = _np.reshape(data, (width, height))
                image_list.append(data2d_t)
                _time.sleep(delay)
            images = _np.ma.dstack(image_list)
            data2d = _np.ma.mean(images, 2)
        return data2d

    def _fromDataArrayToMaskedArray(self, width, height, data_array):
        # data = np.reshape(data_array, (width, height))
        data = _np.reshape(
            data_array, (height, width)
        )  # mod20231002, rectangular frames were bad. now fixed

        idx, idy = _np.where(_np.isnan(data))
        mask = _np.zeros((data.shape[0], data.shape[1]))
        mask[idx, idy] = 1
        masked_ima = _np.ma.masked_array(data, mask=mask.astype(bool))
        return masked_ima

    def capture(self, numberOfFrames, folder_name=None):
        """
        Parameters
        ----------
        numberOfFrames: int
            number of frames to acquire

        Other parameters
        ---------------
        folder_name: string
            if None a tacking number is generate

        Returns
        -------
        folder_name: string
            name of folder measurements
        """
        if folder_name is None:
            folder_name = self._ts()
        print(folder_name)

        self._i4d.burstFramesToSpecificDirectory(
            _os.path.join(_folds.CAPTURE_FOLDER_NAME_4D_PC, folder_name), numberOfFrames
        )
        return folder_name

    def produce(self, folder_name):
        """
        Parameters
        ----------
        folder_name: string
            name of folder measurements to convert
        """
        self._i4d.convertRawFramesInDirectoryToMeasurementsInDestinationDirectory(
            _os.path.join(_folds.PRODUCE_FOLDER_NAME_4D_PC, folder_name),
            _os.path.join(_folds.CAPTURE_FOLDER_NAME_4D_PC, folder_name),
        )

        _sh.move(
            _os.path.join(_folds.PRODUCE_FOLDER_NAME_LOCAL_PC, folder_name),
            _folds.OPD_IMAGES_ROOT_FOLDER,
        )
        rename4D(folder_name)

    def loadConfiguration(self, conffile):
        self._i4d.loadConfiguration(conffile)

    def getCameraSettings(self):
        """
        Return
        ----------
        output: list
        the output is a 4 elements list with width_pixel, height_pixel, offset_x, offset_y, as read from the local copy of the 4D camera settings file
        """

        file_path = _folds.SETTINGS_CONF_FILE
        setting_reader = _confReader(file_path)
        width_pixel = setting_reader.getImageWidhtInPixels()
        height_pixel = setting_reader.getImageHeightInPixels()
        offset_x = setting_reader.getOffsetX()
        offset_y = setting_reader.getOffsetY()
        return [width_pixel, height_pixel, offset_x, offset_y]

    def getFrameRate(self):
        """
        Return
        ----------
        frame_rate: float
        frame rate of the interferometer
        """

        file_path = _folds.SETTINGS_CONF_FILE
        setting_reader = _confReader(file_path)
        frame_rate = setting_reader.getFrameRate()
        return frame_rate

    def intoFullFrame(self, img):
        """
        The function fits the passed frame (expected cropped) into the full interferometer frame (2048x2048), after reading the cropping parameters.

        Parameters
        ----------
        img: masked_array

        Return
        ----------
        output: masked_array
        the output is the interferometer full frame
        """
        off = (self.getCameraSettings())[2:4]
        off = _np.flip(off)
        nfullpix = _np.array([2048, 2048])
        fullimg = _np.full(nfullpix, _np.nan)  # was   _np.zeros(nfullpix)
        fullmask = _np.ones(nfullpix)
        offx = off[0]
        offy = off[1]
        sx = _np.shape(img)[0]  # croppar[2]
        sy = _np.shape(img)[1]  # croppar[3]
        fullimg[offx : offx + sx, offy : offy + sy] = img.data
        fullmask[offx : offx + sx, offy : offy + sy] = img.mask
        fullimg = _np.ma.masked_array(fullimg, fullmask)
        return fullimg
