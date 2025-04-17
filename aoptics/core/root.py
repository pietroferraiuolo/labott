import os as _os
import yaml as _yml
import configparser as _cp
from shutil import copy as _copy
from ruamel.yaml import YAML as _YAML

_gyml = _YAML()
_gyml.preserve_quotes = True

global BASE_DATA_PATH
global CONFIGURATION_FILE
global _config
global ROOT_CONFIGURATION_FILE
global FLAT_ROOT_FOLDER
global INTMAT_ROOT_FOLDER
global LOGGING_ROOT_FOLDER
global CONFIGURATION_FOLDER
global MODALBASE_ROOT_FOLDER
global ALIGNMENT_ROOT_FOLDER
global OPD_SERIES_ROOT_FOLDER
global OPD_IMAGES_ROOT_FOLDER
global IFFUNCTIONS_ROOT_FOLDER
global CONTROL_MATRIX_FOLDER
global OPT_DATA_ROOT_FOLDER
global SETTINGS_CONF_FILE
global COPIED_SETTINGS_CONF_FILE
global CAPTURE_FOLDER_NAME_4D_PC
global PRODUCE_FOLDER_NAME_4D_PC
global PRODUCE_FOLDER_NAME_LOCAL_PC

TEMPLATE_CONF_FILE: str = (
    _os.path.dirname(_os.path.abspath(__file__)) + "/_configurations/configuration.yaml"
)
CONFIGURATION_ROOT_FOLDER: str = _os.path.dirname(TEMPLATE_CONF_FILE)

BASE_DATA_PATH: str = None
OPT_DATA_ROOT_FOLDER: str = None
CONFIGURATION_FILE: str = None
_config: dict = None
FLAT_ROOT_FOLDER: str = None
INTMAT_ROOT_FOLDER: str = None
LOGGING_ROOT_FOLDER: str = None
CONFIGURATION_FOLDER: str = None
MODALBASE_ROOT_FOLDER: str = None
ALIGNMENT_ROOT_FOLDER: str = None
OPD_SERIES_ROOT_FOLDER: str = None
OPD_IMAGES_ROOT_FOLDER: str = None
IFFUNCTIONS_ROOT_FOLDER: str = None
CONTROL_MATRIX_FOLDER: str = None
########################
# INTERFEROMETER PATHS #
########################
SETTINGS_CONF_FILE: str = None
COPIED_SETTINGS_CONF_FILE: str = None
CAPTURE_FOLDER_NAME_4D_PC: str = None
PRODUCE_FOLDER_NAME_4D_PC: str = None
PRODUCE_FOLDER_NAME_LOCAL_PC: str = None


def create_configuration_file(
    path: str = "", data_path: str | bool = False, load: bool = False
) -> None:
    """
    Create a configuration file in the specified path.

    Parameters
    ----------
    path : str
        The path to the configuration file.
    data_path : str | bool
        The path to the data folder. If True, it will be set to the same
        directory as the configuration file. If False, it will not be set.
        If a string, a path must be provided, and the `data_path` will be
        set to that path.
    load : bool
        If True, the configuration file will be loaded after creation, the folder
        tree created (if not already) and all the paths updated.
    """
    global TEMPLATE_CONF_FILE
    global CONFIGURATION_FILE
    global _config
    bp = _os.path.expanduser("~")
    if not bp in path:
        if "mnt" in path or "media" in path:
            pass
        else:
            path = _os.path.join(bp, path)
    if not ".yaml" in path:
        file = _os.path.join(path, "configuration.yaml")
        _create_folder(path)
        if not _os.path.isdir(path):
            raise OSError(f"Invalid Path: {path}")
    else:
        file = path
        _create_folder(_os.path.dirname(path))
    _copy(TEMPLATE_CONF_FILE, file)
    if data_path is not False:
        data_path = _os.path.dirname(file) if data_path is True else data_path
        with open(file, "r") as _f:
            config = _gyml.load(_f)
        config["SYSTEM"]["data_path"] = data_path
        with open(file, "w") as _f:
            _gyml.dump(config, _f)
    if load:
        with open(file, "r") as _f:
            _config = _gyml.load(_f)
        CONFIGURATION_FILE = file
        load_configuration_file(CONFIGURATION_FILE)


def load_configuration_file(file_path: str) -> None:
    """
    Load a configuration file and updates the folder tree and system configuration.

    Parameters
    ----------
    file_path : str
        The FULL path to the configuration file, including the file name if a
        custom name has been used upon creating it.
    """
    from .read_config import _update_imports
    global BASE_DATA_PATH
    global CONFIGURATION_FILE
    global _config
    bp = _os.path.expanduser("~")
    if not bp in file_path:
        if "/mnt" in file_path or "/media" in file_path:
            pass
        else:
            file_path = _os.path.join(bp, file_path)
    if not ".yaml" in file_path:
        if not _os.path.isdir(file_path):
            raise OSError(f"Invalid Path: {file_path}.")
        file_path = _os.path.join(file_path, "configuration.yaml")
    with open(file_path, "r") as _f:
        _config = _gyml.load(_f)
    CONFIGURATION_FILE = file_path
    BASE_DATA_PATH = _config["SYSTEM"]["data_path"]
    _create_folder_tree()
    _update_imports()



def _create_folder(path):
    if not _os.path.exists(path):
        _os.makedirs(path)


def _create_folder_tree() -> None:
    """
    Create the folder tree for the package.
    """
    global BASE_DATA_PATH
    global OPT_DATA_ROOT_FOLDER
    global LOGGING_ROOT_FOLDER
    global CONFIGURATION_FOLDER
    global FLAT_ROOT_FOLDER
    global INTMAT_ROOT_FOLDER
    global MODALBASE_ROOT_FOLDER
    global OPD_SERIES_ROOT_FOLDER
    global OPD_IMAGES_ROOT_FOLDER
    global IFFUNCTIONS_ROOT_FOLDER
    global ALIGNMENT_ROOT_FOLDER
    global CONTROL_MATRIX_FOLDER
    OPT_DATA_ROOT_FOLDER = _os.path.join(BASE_DATA_PATH, "OPTData")
    LOGGING_ROOT_FOLDER = _os.path.join(BASE_DATA_PATH, "Logging")
    CONFIGURATION_FOLDER = _os.path.join(BASE_DATA_PATH, "SysConfig")
    FLAT_ROOT_FOLDER = _os.path.join(OPT_DATA_ROOT_FOLDER, "Flattening")
    INTMAT_ROOT_FOLDER = _os.path.join(OPT_DATA_ROOT_FOLDER, "IntMatrices")
    MODALBASE_ROOT_FOLDER = _os.path.join(OPT_DATA_ROOT_FOLDER, "ModalBases")
    OPD_SERIES_ROOT_FOLDER = _os.path.join(OPT_DATA_ROOT_FOLDER, "OPDSeries")
    OPD_IMAGES_ROOT_FOLDER = _os.path.join(OPT_DATA_ROOT_FOLDER, "OPDImages")
    IFFUNCTIONS_ROOT_FOLDER = _os.path.join(OPT_DATA_ROOT_FOLDER, "IFFunctions")
    ALIGNMENT_ROOT_FOLDER = _os.path.join(OPT_DATA_ROOT_FOLDER, "Alignment")
    CONTROL_MATRIX_FOLDER = _os.path.join(ALIGNMENT_ROOT_FOLDER, "ControlMatrices")
    for p in [
        BASE_DATA_PATH,
        OPT_DATA_ROOT_FOLDER,
        LOGGING_ROOT_FOLDER,
        CONFIGURATION_FOLDER,
        FLAT_ROOT_FOLDER,
        INTMAT_ROOT_FOLDER,
        MODALBASE_ROOT_FOLDER,
        OPD_SERIES_ROOT_FOLDER,
        OPD_IMAGES_ROOT_FOLDER,
        IFFUNCTIONS_ROOT_FOLDER,
        ALIGNMENT_ROOT_FOLDER,
        CONTROL_MATRIX_FOLDER,
    ]:
        _create_folder(p)


def _updateInterfPaths(paths: dict) -> None:
    """
    Update the path of the configuration file and the folders.

    This function reads the configuration file and updates the paths of the
    settings file and the folders used in the package.
    """
    global SETTINGS_CONF_FILE
    global CAPTURE_FOLDER_NAME_4D_PC
    global PRODUCE_FOLDER_NAME_4D_PC
    global PRODUCE_FOLDER_NAME_LOCAL_PC
    SETTINGS_CONF_FILE = paths["settings"]
    CAPTURE_FOLDER_NAME_4D_PC = paths["capture_4dpc"]
    PRODUCE_FOLDER_NAME_4D_PC = paths["produce_4dpc"]
    PRODUCE_FOLDER_NAME_LOCAL_PC = paths["produce"]


###############################################################################
# CLASSES DEFINITIONS: THE FOLDER TREE WRAPPER AND THE 4D CONFIGURATION READER
###############################################################################


class _folds:
    """Wrapper class for the folder tree of the package"""

    @property
    def BASE_DATA_PATH(self):
        """
        Returns the base data path.
        """
        global BASE_DATA_PATH
        return BASE_DATA_PATH

    @property
    def OPT_DATA_ROOT_FOLDER(self):
        """
        Returns the OPT data root folder.
        """
        global OPT_DATA_ROOT_FOLDER
        return OPT_DATA_ROOT_FOLDER

    @property
    def CONFIGURATION_FILE(self):
        """
        Returns the configuration file path.
        """
        global CONFIGURATION_FILE
        return CONFIGURATION_FILE

    @property
    def FLAT_ROOT_FOLDER(self):
        """
        Returns the flat root folder.
        """
        global FLAT_ROOT_FOLDER
        return FLAT_ROOT_FOLDER

    @property
    def INTMAT_ROOT_FOLDER(self):
        """
        Returns the INTMAT root folder.
        """
        global INTMAT_ROOT_FOLDER
        return INTMAT_ROOT_FOLDER

    @property
    def LOGGING_ROOT_FOLDER(self):
        """
        Returns the logging root folder.
        """
        global LOGGING_ROOT_FOLDER
        return LOGGING_ROOT_FOLDER

    @property
    def CONFIGURATION_FOLDER(self):
        """
        Returns the configuration folder.
        """
        global CONFIGURATION_FOLDER
        return CONFIGURATION_FOLDER

    @property
    def MODALBASE_ROOT_FOLDER(self):
        """
        Returns the modal base root folder.
        """
        global MODALBASE_ROOT_FOLDER
        return MODALBASE_ROOT_FOLDER

    @property
    def ALIGNMENT_ROOT_FOLDER(self):
        """
        Returns the alignment root folder.
        """
        global ALIGNMENT_ROOT_FOLDER
        return ALIGNMENT_ROOT_FOLDER

    @property
    def OPD_SERIES_ROOT_FOLDER(self):
        """
        Returns the OPD series root folder.
        """
        global OPD_SERIES_ROOT_FOLDER
        return OPD_SERIES_ROOT_FOLDER

    @property
    def OPD_IMAGES_ROOT_FOLDER(self):
        """
        Returns the OPD images root folder.
        """
        global OPD_IMAGES_ROOT_FOLDER
        return OPD_IMAGES_ROOT_FOLDER

    @property
    def IFFUNCTIONS_ROOT_FOLDER(self):
        """
        Returns the IFFunctions root folder.
        """
        global IFFUNCTIONS_ROOT_FOLDER
        return IFFUNCTIONS_ROOT_FOLDER

    @property
    def CONTROL_MATRIX_FOLDER(self):
        """
        Returns the control matrix folder.
        """
        global CONTROL_MATRIX_FOLDER
        return CONTROL_MATRIX_FOLDER

    @property
    def SETTINGS_CONF_FILE(self):
        """
        Returns the settings configuration file path.
        """
        global SETTINGS_CONF_FILE
        return SETTINGS_CONF_FILE

    @property
    def COPIED_SETTINGS_CONF_FILE(self):
        """
        Returns the copied settings configuration file path.
        """
        global COPIED_SETTINGS_CONF_FILE
        return COPIED_SETTINGS_CONF_FILE

    @property
    def CAPTURE_FOLDER_NAME_4D_PC(self):
        """
        Returns the capture folder name for 4D PC.
        """
        global CAPTURE_FOLDER_NAME_4D_PC
        return CAPTURE_FOLDER_NAME_4D_PC

    @property
    def PRODUCE_FOLDER_NAME_4D_PC(self):
        """
        Returns the produce folder name for 4D PC.
        """
        global PRODUCE_FOLDER_NAME_4D_PC
        return PRODUCE_FOLDER_NAME_4D_PC

    @property
    def PRODUCE_FOLDER_NAME_LOCAL_PC(self):
        """
        Returns the produce folder name for the local PC.
        """
        global PRODUCE_FOLDER_NAME_LOCAL_PC
        return PRODUCE_FOLDER_NAME_LOCAL_PC


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
