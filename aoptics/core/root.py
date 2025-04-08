import os as _os
import yaml as _yml
import configparser as _cp
from shutil import copy as _copy

ROOT_CONFIGURATION_FILE = (
    _os.path.dirname(_os.path.abspath(__file__)) + "/_configurations/configuration.yaml"
)
CONFIGURATION_ROOT_FOLDER = _os.path.dirname(ROOT_CONFIGURATION_FILE)

with open(ROOT_CONFIGURATION_FILE, "r") as _f:
    _root_config = _yml.safe_load(_f)

BASE_DATA_PATH = _root_config["SYSTEM"]["data_path"]
if not BASE_DATA_PATH == "":
    _copied_config = _os.path.join(
        BASE_DATA_PATH, "SysConfigurations", "configuration.yaml"
    )
    if not _os.path.exists(BASE_DATA_PATH):
        res = input(
            f"Base data path folder not found.\nCreate folder tree at {BASE_DATA_PATH}? (Y/n): "
        )
        if res.lower() == "y":
            _os.makedirs(BASE_DATA_PATH)
            _os.makedirs(_os.path.join(BASE_DATA_PATH, "SysConfigurations"))
    try:
        if not _os.path.exists(_copied_config):
            _copy(ROOT_CONFIGURATION_FILE, _copied_config)
            CONFIGURATION_FILE = _copied_config
            _config = _root_config
            print(f"Created configuration file at\n`{_copied_config}`")
        else:
            CONFIGURATION_FILE = _copied_config
            with open(_copied_config, "r") as _f:
                _config = _yml.safe_load(_f)
            if BASE_DATA_PATH != _config["SYSTEM"]["data_path"]:
                BASE_DATA_PATH = _config["SYSTEM"]["data_path"]
            print(f"Reading configuration file at\n`{_copied_config}`")
    except FileNotFoundError:
        BASE_DATA_PATH = _os.path.join(_os.path.expanduser("~"), "aopticsData")
        print(
            f"Failed to copy configuration file and to create folder tree.\nUsing root configuration file\
`{ROOT_CONFIGURATION_FILE}` and `{BASE_DATA_PATH}` as base data path\n"
        )
        CONFIGURATION_FILE = ROOT_CONFIGURATION_FILE
        _config = _root_config
else:
    BASE_DATA_PATH = _os.path.join(_os.path.expanduser("~"), "aopticsData")
    print(
        f"Base data path not set in configuration file.\nUsing root configuration file\
`{ROOT_CONFIGURATION_FILE}` and `{BASE_DATA_PATH}` as base data path\n"
    )
    CONFIGURATION_FILE = ROOT_CONFIGURATION_FILE
    _config = _root_config

####################################
# INTERFEROMETER SECTION READING
####################################
_iconfig = _config["INTERFEROMETER"]
I4D_IP                       = str(_iconfig["i4d_ip"])
I4D_PORT                     = int(_iconfig["i4d_port"])
SETTINGS_CONF_FILE           = str(_iconfig["Paths"]["settings"])
CAPTURE_FOLDER_NAME_4D_PC    = str(_iconfig["Paths"]["capture_4dpc"])
PRODUCE_FOLDER_NAME_4D_PC    = str(_iconfig["Paths"]["produce_4dpc"])
PRODUCE_FOLDER_NAME_LOCAL_PC = str(_iconfig["Paths"]["produce"])

####################################
# FOLDER TREE CREATION
####################################
FLAT_ROOT_FOLDER = _os.path.join(BASE_DATA_PATH, "Flattening")
INTMAT_ROOT_FOLDER = _os.path.join(BASE_DATA_PATH, "IntMatrices")
LOGGING_ROOT_FOLDER = _os.path.join(BASE_DATA_PATH, "Logs")
CONFIGURATION_FOLDER = _os.path.join(BASE_DATA_PATH, "SysConfigurations")
MODALBASE_ROOT_FOLDER = _os.path.join(BASE_DATA_PATH, "ModalBases")
ALIGNMENT_ROOT_FOLDER  = _os.path.join(BASE_DATA_PATH, "Alignment")
OPD_SERIES_ROOT_FOLDER  = _os.path.join(BASE_DATA_PATH, "OPDSeries")
OPD_IMAGES_ROOT_FOLDER   = _os.path.join(BASE_DATA_PATH, "OPDImages")
IFFUNCTIONS_ROOT_FOLDER   = _os.path.join(BASE_DATA_PATH, "IFFunctions")
CONTROL_MATRIX_FOLDER   = _os.path.join(ALIGNMENT_ROOT_FOLDER, "ControlMatrices")

for p in [
    BASE_DATA_PATH,
    LOGGING_ROOT_FOLDER,
    IFFUNCTIONS_ROOT_FOLDER,
    INTMAT_ROOT_FOLDER,
    MODALBASE_ROOT_FOLDER,
    ALIGNMENT_ROOT_FOLDER,
    OPD_IMAGES_ROOT_FOLDER,
    OPD_SERIES_ROOT_FOLDER,
    CONFIGURATION_FOLDER,
    FLAT_ROOT_FOLDER,
    CONTROL_MATRIX_FOLDER,
]:
    if not _os.path.exists(p):
        _os.makedirs(p)


class _folds:
    """Wrapper class for the folder tree of the package"""

    def __init__(self):
        self.I4D_IP = I4D_IP
        self.I4D_PORT = I4D_PORT
        self.BASE_DATA_PATH = BASE_DATA_PATH
        self.CONFIGURATION_FOLDER = CONFIGURATION_FOLDER
        self.SETTINGS_CONF_FILE = SETTINGS_CONF_FILE
        self.LOGGING_FILE_PATH = LOGGING_ROOT_FOLDER
        self.OPD_SERIES_ROOT_FOLDER = OPD_SERIES_ROOT_FOLDER
        self.OPD_IMAGES_ROOT_FOLDER = OPD_IMAGES_ROOT_FOLDER
        self.IFFUNCTIONS_ROOT_FOLDER = IFFUNCTIONS_ROOT_FOLDER
        self.PRODUCE_FOLDER_NAME_4D_PC = PRODUCE_FOLDER_NAME_4D_PC
        self.CAPTURE_FOLDER_NAME_4D_PC = CAPTURE_FOLDER_NAME_4D_PC
        self.PRODUCE_FOLDER_NAME_LOCAL_PC = PRODUCE_FOLDER_NAME_LOCAL_PC
        self.CONFIGURATION_ROOT_FOLDER = CONFIGURATION_ROOT_FOLDER
        self.MODALBASE_ROOT_FOLDER = MODALBASE_ROOT_FOLDER
        self.INTMAT_ROOT_FOLDER = INTMAT_ROOT_FOLDER
        self.ALIGNMENT_ROOT_FOLDER = ALIGNMENT_ROOT_FOLDER
        self.CONFIGURATION_FILE = CONFIGURATION_FILE
        self.FLAT_ROOT_FOLDER = FLAT_ROOT_FOLDER
        self.CONTROL_MATRIX_FOLDER = CONTROL_MATRIX_FOLDER

    @property
    def print_all(self):
        """Print all the folders"""
        attributes = [
            ("Read configuration file", CONFIGURATION_FILE),
            ("Base data path", self.BASE_DATA_PATH),
            ("Configuration folder", self.CONFIGURATION_FOLDER),
            ("OPD series folder", self.OPD_SERIES_ROOT_FOLDER),
            ("OPD images folder", self.OPD_IMAGES_ROOT_FOLDER),
            ("Modal base root folder", self.MODALBASE_ROOT_FOLDER),
            ("Alignment folder", self.ALIGNMENT_ROOT_FOLDER),
            ("Control matrix folder", self.CONTROL_MATRIX_FOLDER),
            ("IFFunctions folder", self.IFFUNCTIONS_ROOT_FOLDER),
            ("Intmat folder", self.INTMAT_ROOT_FOLDER),
            ("Flattening folder", self.FLAT_ROOT_FOLDER),
            ("Logging root folder", self.LOGGING_FILE_PATH),
            ("Interferometer settings file", self.SETTINGS_CONF_FILE),
            ("Produce folder name 4D PC", self.PRODUCE_FOLDER_NAME_4D_PC),
            ("Capture folder name 4D PC", self.CAPTURE_FOLDER_NAME_4D_PC),
            ("Produce folder name local PC", self.PRODUCE_FOLDER_NAME_LOCAL_PC),
            ("Interferometer IP", self.I4D_IP),
            ("Interferometer port", self.I4D_PORT),
        ]
        for name, value in attributes:
            print(f"{name}: {value}")

folders = _folds()


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
