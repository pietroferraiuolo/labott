# Changes to be made in the `XuPy Transition` branch

```plain
✅ GROUND
 |__ ✅ `computerec.py` : rewrite the module, performing only the operations when asked and with caching and gpu support
 |__ ✅ `zernike.py` : Enable caching and gpu support of the fitting functions
 |__ ✅ `osutils.py` : Enable array on GPU support for I/O operations (save/read fits)
 |                      Example: read fits as xupyMaskedArrays (flag)
 |
✅ DMUTILS
 |__ ✅ `iff_processing.py` : Enable GPU support for image processing
 |__ ✅ `flattening.py` : make CmdMat, IntMat e RecMat pubblicly available
 |
[] `analyzers.py` : Check for GPU useful operations
 |__ modify and/or rewrite lot of functions 
 |
[] `alignment.py` : Check for GPU useful operations
 |__ ✅ `_push_pull_redux` : copy that of `iff_processing` -> actually abstracted and put into `analyzers.py`
 |__ `zern_routine` + `global_zern_on_roi` : rewrite into the more general way, using the `ZernikeFitter`

```

## Performance gains considerations

### `IFF_PROCESSING.PY`


| function | execution time before (average) | execution time after (average) | changes | notes |
| -------- | --------------- | -------------- | ------- | ----- |
|`iffRedux` | 41.2 s | 14.9 s | Computation vectorization, GPU, I/O parallelization and prefetching | Test dome with DP data tn=20250911_110614: ~1200 images to be processed. I/O sweet spot performance reached with n_workers = 8, 1 modes prefetch |
| `filterZernikeCube` | 43 s | 22 s | Fitting algorithm optimization with little GPU support | - |
