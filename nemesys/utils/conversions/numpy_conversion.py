import numpy as np


class NumPyConversion:
    @staticmethod
    def to_array(content, dtype):
        return np.array(content, dtype=dtype)
