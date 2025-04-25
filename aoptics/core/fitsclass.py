import numpy as _np
from numpy.typing import ArrayLike
from astropy.io.fits import Header, PrimaryHDU, HDUList

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
    def __new__(cls, data: list | ArrayLike, header: dict | Header = None):
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
        # Add the header attribute
        obj.header = dict(header) if isinstance(header, Header) else header
        return obj

    def __array_finalize__(self, obj):
        # This is called when the object is created or viewed as a subclass
        if obj is None:
            return
        self.header = getattr(obj, 'header', None)

    def writeto(self, filename: str, overwrite: bool = False):
        """
        Write the array and its header to a FITS file.

        Parameters
        ----------
        filename : str
            The name of the FITS file to write to.
        overwrite : bool, optional
            If True, overwrite the file if it exists. Defaults to False.
        """
        # Convert the header to an astropy.io.fits.Header if it's a dictionary
        fits_header = Header(self.header) if isinstance(self.header, dict) else self.header
        fits_header['MASKED'] = False
        hdu = PrimaryHDU(data=self, header=fits_header)
        hdu.writeto(filename, overwrite=overwrite)


class FitsArrayMasked(_np.ma.MaskedArray):
    """
    A custom subclass of numpy.ma.MaskedArray that includes a FITS header.
    This class is used to store data along with its associated FITS header,
    allowing for easier manipulation and storage of FITS data.
    The header can be a dictionary or an astropy.io.fits.Header object.

    Parameters
    ----------
    data : list, numpy.ndarray, or numpy.ma.MaskedArray
        The data to be stored in the array.
    mask : numpy.ndarray, optional
        The mask to be applied to the data. If not provided, defaults to None.
    header : dict or astropy.io.fits.Header, optional
        The FITS header associated with the data. If not provided, defaults to None.
    """
    def __new__(cls, data: list | ArrayLike, mask: ArrayLike = None, header: dict | Header = None):
        """
        Create a new instance of the FitsArrayMasked class.
        
        Parameters
        ----------
        data : list, numpy.ndarray, numpy.ma.MaskedArray
            The data to be stored in the array. Can be a masked array
        mask : numpy.ndarray, optional
            The mask to be applied to the data. If not provided and `data` is a 
            masked array, then the mask is automatically set. Defaults to None.
        header : dict or astropy.io.fits.Header, optional
            The FITS header associated with the data. If not provided, defaults to None.
        """
        if all(
            [isinstance(data, _np.ma.MaskedArray), 
             hasattr(data, 'mask')]
        ):
            obj = _np.ma.masked_array(data, data.mask) # .view(cls)
        else:
            obj = _np.ma.masked_array(data, mask) # .view(cls)
        obj.header = dict(header) if isinstance(header, Header) else header
        return obj

    # def __repr__(self, obj):
    #     """The representation"""
    #     return obj.__repr__().replace('masked_array', 'FitsArrayMasked')
    
    def __array_finalize__(self, obj):
        # This is called when the object is created or viewed as a subclass
        if obj is None:
            return
        self.header = getattr(obj, 'header', None)

    def writeto(self, filename: str, overwrite: bool = False):
        """
        Write the array and its header to a FITS file.

        Parameters
        ----------
        filename : str
            The name of the FITS file to write to.
        overwrite : bool, optional
            If True, overwrite the file if it exists. Defaults to False.
        """
        # Convert the header to an astropy.io.fits.Header if it's a dictionary
        fits_header = Header(self.header) if isinstance(self.header, dict) else self.header
        fits_header['MASKED'] = True
        hdu      = PrimaryHDU(data=self, header=fits_header)
        mask_hdu = PrimaryHDU(data=self.mask.astype(_np.uint8), name='MASK')
        hdu_list = HDUList([hdu, mask_hdu])
        hdu_list.writeto(filename, overwrite=overwrite)
