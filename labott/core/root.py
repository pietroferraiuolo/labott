import os as _os
import configparser as _cp
from shutil import copy as _copy

_config = _cp.ConfigParser()

CONFIGURATION_FILE = _os.path.dirname(_os.path.abspath(__file__)) + "/config.conf"
_cpfile = _os.path.expanduser("~") + "/interfConfig.conf"
if not _os.path.exists(_cpfile):
    _copy(CONFIGURATION_FILE, _cpfile)
    CONFIGURATION_FILE = _cpfile
    print(f"Created configuration file at `{_cpfile}`")
else: 
    CONFIGURATION_FILE = _cpfile
    print(f"Reading configuration file at `{_cpfile}`")


_config.read(CONFIGURATION_FILE)
_cc = _config["PATHS"]
_ci = _config["INTERF"]

I4D_IP = str(_ci["i4d_ip"])
I4D_PORT = int(_ci["i4d_port"])
CAPTURE_FOLDER_NAME_4D_PC = str(_cc["capture_4dpc"])
PRODUCE_FOLDER_NAME_4D_PC = str(_cc["produce_4dpc"])
PRODUCE_FOLDER_NAME_LOCAL_PC = str(_cc["produce"])
SETTINGS_CONF_FILE = str(_ci["settings"])

CORE_FOLDER_PATH = _os.path.dirname(CONFIGURATION_FILE)
BASE_PATH = _os.path.dirname(CORE_FOLDER_PATH)
BASE_DATA_PATH = str(_cc["data_path"])
OPD_IMAGES_ROOT_FOLDER = _os.path.join(BASE_DATA_PATH, "OPDImages")
OPD_SERIES_ROOT_FOLDER = _os.path.join(BASE_DATA_PATH, "OPDSeries")
LOGGING_FILE_PATH = _os.path.join(BASE_DATA_PATH, "interf.log")


for p in [BASE_DATA_PATH, OPD_IMAGES_ROOT_FOLDER, OPD_SERIES_ROOT_FOLDER]:
    if not _os.path.exists(p):
        _os.makedirs(p)



class ConfSettingReader4D:
    """
    Class which reads an interferometer configuration settings file '4DSettings.ini'

    Methods
    -------
    getFrameRate() :
        Gets the camera frame rate in Hz.

    getImageWidthInPixels() :
        Get the width of the frame in pixel units.

    getImageHeightInPixels() :
        Get the height of the frame in pixel units.

    getOffsetX() :
        Get the frame offset in x-axis.

    getOffsetY() :
        Get the frame offset in y-axis.

    getPixelFormat() :
        Get the format of the pixels.

    getUserSettingFilePath() :
        Get the path of the configuration file.

    How to Use it
    -------------
    After initializing the class with a file path, just call methods on the defined
    object

    >>> cr = ConfSettingReader(file_path)
    >>> cr.getImageWidhtInPixels()
    2000
    >>> cr.getImageHeightInPixels()
    2000

    Notes
    -----
    Note that there is no need to directly use this module, as the settings information
    retrievement is handled by m4.urils.osutils, with its functions
    ''getConf4DSettingsPath'' and ''getCameraSettings''.
    """

    def __init__(self, file_path):
        self.config = _cp.ConfigParser()
        self.config.read(file_path)
        self.camera_section = "ACA2440"
        self.path_section = "Paths"

    # CAMERA
    def getFrameRate(self):
        """
        Returns the acquisition frame rate of the interferometer in Hz

        Returns
        -------
        frame_rate : float
            The frame rate.
        """
        frame_rate = self.config.get(self.camera_section, "FrameRate")
        return float(frame_rate)

    def getImageWidhtInPixels(self):
        """
        Returns the image widht in pixel scale

        Returns
        -------
        image_wight_in_pixels : int
            Image pixel width.
        """
        image_width_in_pixels = self.config.get(
            self.camera_section, "ImageWidthInPixels"
        )
        return int(image_width_in_pixels)

    def getImageHeightInPixels(self):
        """
        Returns the image height in pixel scale

        Returns
        -------
        image_height_in_pixels : int
            Image pixel height.
        """
        image_height_in_pixels = self.config.get(
            self.camera_section, "ImageHeightInPixels"
        )
        return int(image_height_in_pixels)

    def getOffsetX(self):
        """
        Returns the camera offset, in pixels, along the x-axis.

        Returns
        -------
        offset_x : int
            Pixel offset in the x-axis.
        """
        offset_x = self.config.get(self.camera_section, "OffsetX")
        return int(offset_x)

    def getOffsetY(self):
        """
        Returns the camera offset, in pixels, along the y-axis.

        Returns
        -------
        offset_y : int
            Pixel offset in the y-axis.
        """
        offset_y = self.config.get(self.camera_section, "OffsetY")
        return int(offset_y)

    def getPixelFormat(self):
        """
        Returns the format of the pixel.

        Returns
        -------
        pixel_format : str
            Pixel format.
        """
        pixel_format = self.config.get(self.camera_section, "PixelFormat")
        return pixel_format

    # PATH
    def getUserSettingFilePath(self):
        """
        Returns the complete filepath of the settings configuration file.

        Returns
        -------
        user_setting_file_path : str
            Settings file path.
        """
        user_setting_file_path = self.config.get(
            self.path_section, "UserSettingsFilePath"
        )
        return user_setting_file_path
