class SizeException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class PaddingException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class ReshapingException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class ShapeMismatch(Exception):
    def __init__(self, message: str):
        super().__init__(message)
