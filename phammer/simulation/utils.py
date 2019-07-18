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

def set_settings(t, settings, setting_property):
    for obj_id in settings:
        if t < len(settings[obj_id]) - 1:
            setting_property[obj_id] = settings[obj_id][t]

def set_coefficients(obj_curves, coeff, obj_setting):
    for obj_id in obj_curves:
        fcn = obj_curves[obj_id]
        coeff[obj_id] = fcn(obj_setting[obj_id])