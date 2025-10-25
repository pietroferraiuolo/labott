# Changes to be made in the `XuPy Transition` branch

```plain
[] GROUND
 |__ ✅ `computerec.py` : rewrite the module, performing only the operations when asked and with caching and gpu support
 |__ ✅ `zernike.py` : Enable caching and gpu support of the fitting functions
 |__ ✅ `osutils.py` : Enable array on GPU support for I/O operations (save/read fits)
 |                      Example: read fits as xupyMaskedArrays (flag)
 |
[] DMUTILS
 |__ ✅ `iff_processing.py` : Enable GPU support for image processing
 |__ ✅ `flattening.py` : make CmdMat, IntMat e RecMat pubblicly available
 |
[] `analyzers.py` : Check for GPU useful operations
[] `alignment.py` : Check for GPU useful operations
```
