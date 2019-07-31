import numpy as np
from collections import namedtuple

class Table:
    Selector = namedtuple('Selector', ['value', 'context'])

    def __init__(self, properties, size):
        self.selectors = {}
        for p, dtype in properties.items():
            self.__dict__[p] = np.zeros(size, dtype=dtype)

    def __getitem__(self, selector):
        if type(selector) is tuple:
            return self.selectors[selector[0]].context
        return self.selectors[selector].value

    def __setitem__(self, index, value):
        if type(index) is tuple:
            if not (type(value) is np.int):
                raise Exception('Only numpy arrays with dtype = np.int are valid as context')
            self.selectors[index[0]].context = value
        self.selectors[index] = self.Selector(value, None)

point_properties = {
    'are_start' : np.bool,
    'are_end' : np.bool,
    'B' : np.float,
    'R' : np.float,
    'Cm' : np.float,
    'Cp' : np.float,
    'Bm' : np.float,
    'Bp' : np.float,
    'flowrate' : np.float,
    'head' : np.float
}

points = Table(point_properties, 5)
print(points.are_start)
points['are_ghost'] = [1,5,6]
print(points['are_ghost',])