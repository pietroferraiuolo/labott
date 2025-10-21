# Changes to be made in the `XuPy Transition` branch

```plain
[] GROUND
 |__ [] `computerec.py` : rewrite the module, performing only the operations when asked and with caching and gpu support
 |__ [] `zernike.py` : Enable caching and gpu support of the fitting functions
 |__ [] `osutils.py` : Enable array on GPU support for I/O operations (save/read fits)
 |
[] DMUTILS
 |__ [] `iff_processing.py` : Enable GPU support for image processing
 |
[] `analyzers.py` : Check for GPU useful operations
[] `alignment.py` : Check for GPU useful operations
```
