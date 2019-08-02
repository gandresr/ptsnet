import numpy as np
from collections import namedtuple

class Table:
    Selector = namedtuple('Selector', ['value', 'context'])

    def __init__(self, properties, size):
        for p, dtype in properties.items():
            self.__dict__[p] = np.zeros(size, dtype=dtype)
        self.__dict__['selectors'] = {}
        self.__dict__['shape'] = (len(properties), size,)

    def __setattr__(self, name, value):
        if name not in self.__dict__:
            raise TypeError("'Table' object does not support attribute assignment")
        else:
            if type(value) != type(self.__dict__[name]):
                raise ValueError("Property '%s' can only be updated not replaced" % name)
            elif value.shape != self.__dict__[name].shape:
                raise ValueError("Property '%s' can only be updated not replaced by new size array" % name)
            elif value.dtype != self.__dict__[name].dtype:
                raise ValueError("Property '%s' can only be updated not replaced by ndarray of different type" % name)
            else:
                object.__setattr__(self, name, value)

    def __repr__(self):
        return "Table <properties: %d, rows: %d>" % self.__dict__['shape']

class Table2D(Table):
    def __init__(self, properties, num_rows, num_cols):
        for property, dtype in properties.items():
            self.__dict__[property] = np.zeros((num_rows, num_cols), dtype = dtype)
        self.__dict__['shape'] = (len(properties), num_rows, num_cols,)

    def __repr__(self):
        return "Table2D <properties: %d, rows: %d, cols: %d>" % self.__dict__['shape']