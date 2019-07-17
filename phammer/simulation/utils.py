from scipy.interpolate import splev, splrep

def define_curve(X, Y):
    spl = splrep(X, Y)
    return lambda x : splev(x, spl)