"""
Author(s)
---------
- Pietro Ferraiuolo : written in 2025

Description
-----------
`opticalib` is a package for the control of the 4D PhaseCam interferometer.

How to Use:
-----------
```python
> import opticalib
> interf = opticalib.PhaseCam('193.206.155.218', 8011)
> img = interf.acquire_map()
```
"""
from .__version__ import __version__
from . import analyzer
from .ground.osutils import load_fits, save_fits, getFileList, read_phasemap
from .core.root import (
    folders,
    create_configuration_file,
)
from .core import read_config
from .devices import *

__all__ = [
    "analyzer",
    "load_fits",
    "save_fits",
    "read_phasemap",
    "getFileList",
    "folders",
    "create_configuration_file",
    "load_configuration_file",
    "read_config",
    "interferometer",
    "deformable_mirrors",
]
