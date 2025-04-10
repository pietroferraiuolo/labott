from abc import ABC, abstractmethod
from aoptics.ground import logger as _logger
from aoptics.ground.osutils import newtn as _newtn
from aoptics.core.read_config import getInterfConfig
from aoptics.core.root import _updateInterfPaths, folders as _folds


class BaseInterferometer(ABC):
    """
    Base class for all interferometer devices.
    """

    def __init__(self, name: str, ip: str, port: str):
        """
        Initializes the interferometer with a name, in order to retrieve
        all the information from the configuration file.
        """
        self.name = name
        config = getInterfConfig(name)
        if (ip and port) is None:
            ip = config["ip"]
            port = config["port"]
        self.ip = ip
        self.port = port
        self._logger = _logger.set_up_logger(f"{self.name}.log", 20)
        self._ts = _newtn
        _updateInterfPaths(config['Paths'])

    @abstractmethod
    def acquire_map(self):
        """
        Abstract method to measure the interference pattern.
        Must be implemented by subclasses.
        """
        pass