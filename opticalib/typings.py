from typing import (
    Union,
    Optional,
    Any,
    TypeVar,
    TypeAlias,
    Callable,
    TYPE_CHECKING,
)
import numpy as _np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .devices import (
        AlpaoDm,
        SplattDm,
        PhaseCam
    )

ImageData = TypeVar("ImageData", bound=_np.ma.core.MaskedArray)

DeformableMirrorDevice: TypeAlias = Union[
    "AlpaoDm",
    "SplattDm",
]

InterferometerDevice: TypeAlias = Union[
    "PhaseCam",
]