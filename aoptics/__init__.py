"""
Author(s)
---------
- Pietro Ferraiuolo : written in 2025

Description
-----------
Gentle is a package for the control of the 4D PhaseCam interferometer.

How to Use:
-----------
```python
> import gentle
> interf = gentle.PhaseCam('193.206.155.218', 8011)
> img = interf.acquire_map()
```
"""

from . import analyzer
from . import devices
from .ground import zernike as zern
from .ground.osutils import (
    load_fits,
    save_fits,
    getFileList,
)
from .core.root import folders


__all__ = [
    'analyzer',
    'zern',
    'load_fits',
    'save_fits',
    'getFileList',
    'folders',
]
