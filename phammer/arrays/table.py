import numpy as np
from collections import namedtuple

class Table:
    Selector = namedtuple('Selector', ['value', 'context'])

    def __init__(self, properties, size):
        for p, dtype in properties.items():
            self.__dict__[p] = np.zeros(size, dtype=dtype)
        self.__dict__['selectors'] = {}

    def __setattr__(self, name, value):
        if name not in self.__dict__:
            raise TypeError("'Table' object does not support attribute assignment")
        else:
            object.__setattr__(self, name, value)

    def __getitem__(self, selector):
        if type(selector) is tuple:
            try:
                return self.selectors[selector[0]].context
            except:
                raise KeyError("%s is not a selector in the Table" % selector)
        try:
            return self.selectors[selector].value
        except:
            raise KeyError("%s is not a selector in the Table" % selector)

    def __setitem__(self, index, value):
        if type(index) is tuple:
            if type(value) is np.ndarray:
                if not (value.dtype == np.int):
                    raise TypeError('Only numpy arrays with dtype = np.int are valid as context')
                self.selectors[index[0]].context = value
            else:
                raise TypeError('Only numpy arrays with dtype = np.int are valid as context')
        else:
            if type(value) is np.ndarray:
                if not (value.dtype == np.int):
                    raise TypeError('Only numpy arrays with dtype = np.int are valid as context')
                self.selectors[index] = self.Selector(value, None)
            else:
                raise TypeError('Only numpy arrays with dtype = np.int are valid as context')