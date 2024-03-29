"""Custom Exceptions for simulation purposes"""


class CoordsValueError(ValueError):
    """Error with creation of coordinates"""
    pass

class FilterError(Exception):
    """Error with the processing of a filter"""
    pass

class LGNError(Exception):
    '''Error with the definition and setup of an LGN layer'''
    pass

class SimulationError(Exception):
    """Error in the course of simulation that has disrupted its execution"""
    pass
