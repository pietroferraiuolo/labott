import numpy as _np

class FitsArray(_np.ndarray):
    def __new__(cls, data, header=None):
        # Create the ndarray instance
        obj = _np.asarray(data).view(cls)
        # Add the header attribute
        obj.header = header
        return obj

    def __array_finalize__(self, obj):
        # This is called when the object is created or viewed as a subclass
        if obj is None:
            return
        self.header = getattr(obj, 'header', None)