from typing import Any

import numpy as np
import numpy.typing as npt


class NumPyConversion:
    @staticmethod
    def to_array(content: Any, dtype: npt.DTypeLike = np.int32):
        return np.array(content, dtype=dtype)
