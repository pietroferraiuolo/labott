class DeviceNotFoundError(Exception):
    """Exception raised when a device is not found in the
        configuration file.
    """
    def __init__(self, device_name: str):
        super().__init__(f"Device '{device_name}' not found in the configuration file.")

class WrongDeviceError(Exception):
    """Exception raised when a device is not found in the
        configuration file.
    """
    def __init__(self, device_name: str, device_type: str):
        super().__init__(f"Device '{device_name}' is not a valid {device_type}.")