"""Custom Exceptions for simulation purposes"""


class CoordsValueError(ValueError):
    """Error with creation of coordinates"""
    pass

class FilterError(Exception):
    """Error with the processing of a filter"""
    pass
