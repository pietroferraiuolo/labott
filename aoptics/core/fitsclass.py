import numpy as _np
from numpy.typing import ArrayLike
from astropy.io import fits as _fits


class FitsArray(_np.ndarray):
    """
    A custom subclass of numpy.ma.MaskedArray that includes a FITS header.
    This class is used to store data along with its associated FITS header,
    allowing for easier manipulation and storage of FITS data.
    The header can be a dictionary or an astropy.io.fits.Header object.

    Parameters
    ----------
    data : list, numpy.ndarray, or numpy.ma.MaskedArray
        The data to be stored in the array.
    header : dict or astropy.io.fits.Header, optional
        The FITS header associated with the data. If not provided, defaults to None.
    """

    def __new__(cls, data: list | ArrayLike, *, mask : ArrayLike = None, header: dict | _fits.Header = None):
        """
        Create a new instance of the FitsArray class.

        Parameters
        ----------
        data : list, numpy.ndarray
            The data to be stored in the array.
        header : dict or astropy.io.fits.Header, optional
            The FITS header associated with the data. If not provided, defaults to None.
        """
        obj = _np.asarray(data).view(cls)
        obj.header = _fits.Header(header) if isinstance(header, dict) else header
        # The following is a workaround for the issue with numpy 1.24.0
        # Trying to include a masked array. The upper part works fine
        try:
            obj.mask = obj._mask = _np.asarray(data.mask)
            obj.fill_value = obj._fill_value = data.get_fill_value()
        except AttributeError:
            if mask is not None:
                obj.mask = obj._mask = _np.asarray(mask)
                obj.fill_value = obj._fill_value = _np.ma.masked_array(data, mask=mask).get_fill_value()
            else:
                pass
        return obj

    def __array_finalize__(self, obj):
        # This is called when the object is created or viewed as a subclass
        if obj is None:
            return
        self.header = getattr(obj, "header", None)
        if hasattr(obj, "mask"):
            self.mask = self._mask = getattr(obj, "mask", None)
            self.fill_value = self._fill_value = getattr(obj, "fill_value", None)

    def writeto(self, filename: str, overwrite: bool = False, *args):
        """
        Write the array and its header to a FITS file.

        Parameters
        ----------
        filename : str
            The name of the FITS file to write to.
        overwrite : bool, optional
            If True, overwrite the file if it exists. Defaults to False.
        """
        self.header["MASKED"] = (hasattr(self, 'mask'), 'is masked array')
        hdu = _fits.PrimaryHDU(data=self, header=self.header)
        if hasattr(self, 'mask'):
            mask_hdu = _fits.ImageHDU(data=self.mask.astype(_np.uint8))
            hdu_list = _fits.HDUList([hdu, mask_hdu])
        else:
            hdu_list = _fits.HDUList([hdu])
        hdu_list.writeto(filename, overwrite=overwrite, *args)


# class FitsArrayMasked(_np.ma.MaskedArray):
#     """
#     A custom subclass of numpy.ma.MaskedArray that includes a FITS header.
#     This class is used to store data along with its associated FITS header,
#     allowing for easier manipulation and storage of FITS data.
#     The header can be a dictionary or an astropy.io.fits.Header object.

#     Parameters
#     ----------
#     data : list, numpy.ndarray, or numpy.ma.MaskedArray
#         The data to be stored in the array.
#     mask : numpy.ndarray, optional
#         The mask to be applied to the data. If not provided, defaults to None.
#     header : dict or astropy.io.fits.Header, optional
#         The FITS header associated with the data. If not provided, defaults to None.
#     """

#     def __new__(
#         cls,
#         data,
#         *,
#         mask=_np.ma.nomask,
#         header=None,
#         dtype=None,
#         copy=False,
#         fill_value=None,
#         keep_mask=True,
#         hard_mask=False,
#         shrink=True,
#         ndmin=0
#     ):
#         # If data is already a MaskedArray, extract its data and mask
#         if isinstance(data, _np.ma.MaskedArray):
#             base_data = data.data
#             base_mask = data.mask
#         else:
#             base_data = data
#             base_mask = mask

#         # Always use masked_array, then view as subclass
#         obj = _np.ma.masked_array(
#             base_data,
#             mask=base_mask,
#             dtype=dtype,
#             copy=copy,
#             fill_value=fill_value,
#             keep_mask=keep_mask,
#             hard_mask=hard_mask,
#             shrink=shrink,
#             subok=False,
#             ndmin=ndmin,
#         ).view(cls)
#         if not hasattr(obj, '_mask'):
#             obj._mask = _np.ma.getmaskarray(obj)
#         if not hasattr(obj, '_sharedmask'):
#             obj._sharedmask = False
#         if not hasattr(obj, '_fill_value'):
#             obj._fill_value = fill_value
#         obj.header = dict(header) if isinstance(header, _fits.Header) else header
#         return obj

#     def __array_finalize__(self, obj):
#         if obj is None:
#             return
#         self.header = getattr(obj, "header", None)

#     def writeto(self, filename: str, overwrite: bool = False):
#         """
#         Write the array and its header to a FITS file.

#         Parameters
#         ----------
#         filename : str
#             The name of the FITS file to write to.
#         overwrite : bool, optional
#             If True, overwrite the file if it exists. Defaults to False.
#         """
#         # Convert the header to an astropy.io.fits.Header if it's a dictionary
#         fits_header = (
#             _fits.Header(self.header) if isinstance(self.header, dict) else self.header
#         )
#         fits_header["MASKED"] = True
#         hdu = _fits.PrimaryHDU(data=self, header=fits_header)
#         mask_hdu = _fits.ImageHDU(data=self.mask.astype(_np.uint8), name="MASK")
#         hdu_list = _fits.HDUList([hdu, mask_hdu])
#         hdu_list.writeto(filename, overwrite=overwrite)
