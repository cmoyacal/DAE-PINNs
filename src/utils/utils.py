import time
import sys
import numpy as np

from functools import wraps 

def timing(f):
    """ 
    Decorator for measuring the execution time of methods. 
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        print("%r took %f s\n" % (f.__name__, te - ts))
        sys.stdout.flush()
        return result
    return wrapper

class dotdict(dict):
    """
    dot.notation access to dictionary attributes.
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def list_to_str(nums, precision=3):
    """
    list to str for displaying errors and metrics.
    """
    if nums is None:
        return ""
    if not isinstance(nums, (list, tuple, np.ndarray)):
        return "{:.{}e}".format(nums, precision)
    return "[{:s}]".format(", ".join(["{:.{}e}".format(x, precision) for x in nums]))