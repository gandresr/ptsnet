from scipy.interpolate import splev, splrep

def define_curve(X, Y):
    spl = splrep(X, Y)
    return lambda x : splev(x, spl)

def is_iterable(x):
    try:
        iter(x)
        return True
    except:
        return False

def set_coefficients(obj_curves, coeff, obj_setting):
    for obj_curve in obj_curves:
            obj_id = obj_curve[0]
            fcn = obj_curve[1]
            coeff[obj_id] = fcn(obj_setting[obj_id])