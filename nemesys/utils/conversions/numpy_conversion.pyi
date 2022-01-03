from typing import Any

import numpy
import numpy.typing as npt

class NumPyConversion:
    @staticmethod
    def to_array(content: Any, dtype: npt.DTypeLike): ...
