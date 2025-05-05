class DeviceNotFoundError(Exception):
    """Exception raised when a device is not found in the
        configuration file.
    """
    def __init__(self, device_name: str):
        super().__init__(f"Device '{device_name}' not found in the configuration file.")
