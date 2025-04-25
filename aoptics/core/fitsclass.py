import numpy as _np
from numpy.typing import ArrayLike
from astropy.io.fits import Header

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


class FitsArrayMasked(FitsArray):
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
            obj = _np.ma.masked_array(data, data.mask)
        else:
            obj = _np.ma.masked_array(data, mask)
        obj.header = dict(header) if isinstance(header, Header) else header
        return obj

    def __repr__(self, obj):
        """The representation"""
        return obj.__repr__().replace('masked_array', 'FitsArrayMasked')
    
    def __array_finalize__(self, obj):
        # This is called when the object is created or viewed as a subclass
        if obj is None:
            return
        self.header = getattr(obj, 'header', None)
