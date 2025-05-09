from abc import ABC, abstractmethod
from opticalib.ground import logger as _logger
from opticalib.ground.osutils import newtn as _newtn
from opticalib.core.read_config import getInterfConfig
from opticalib.core.root import _updateInterfPaths, folders as _folds


class BaseInterferometer(ABC):
    """
    Base class for all interferometer devices.
    """

    def __init__(self, name: str, ip: str, port: int):
        """
        Initializes the interferometer with a name, in order to retrieve
        all the information from the configuration file.
        """
        self.name = name
        if (ip and port) is None:
            config = getInterfConfig(name)
            ip = config["ip"]
            port = config["port"]
        self.ip = ip
        self.port = port
        self._logger = _logger.set_up_logger(f"{self.name}.log", 20)
        self._logger.info(f"Interferometer {self.name} initialized on addess {self.ip}:{self.port}")
        self._ts = _newtn
        _updateInterfPaths(config['Paths'])
        _folds._update_interf_paths()

    @abstractmethod
    def acquire_map(self):
        """
        Abstract method to measure the interference pattern.
        Must be implemented by subclasses.
        """
        pass


class BaseDeformableMirror(ABC):
    """
    Base class for all deformable mirror devices.
    """

    @abstractmethod
    def set_shape(self):
        """
        Abstract method to set the shape of the deformable mirror.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_shape(self):
        """
        Abstract method to get the shape of the deformable mirror.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def uploadCmdHistory(self):
        """
        Abstract method to upload the command history to the deformable mirror.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def runCmdHistory(self):
        """
        Abstract method to run the command history on the deformable mirror.
        Must be implemented by subclasses.
        """
        pass