import numpy as np

from scipy.interpolate import splev, splrep

def define_curve(X, Y):
    spl = splrep(X, Y)
    return lambda x : splev(x, spl)

def is_array(x):
    return type(x) != str and hasattr(x, "__iter__")

def imerge(A, B):
    d1 = len(A)
    d2 = len(B)

    if d1 != d2:
        raise ValueError("Both arrays should have the same dimensions")
    if len(A.shape) != len(B.shape) or len(A.shape) > 1:
        raise ValueError("Both arrays should be 1-dimensional")
    if A.dtype != B.dtype:
        raise ValueError("Both arrays should have the same dtype")

    x = np.zeros(2*d1, dtype = A.dtype)
    x[0::2] = A
    x[1::2] = B

    return x