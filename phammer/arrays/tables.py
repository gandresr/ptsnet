import numpy as np
from collections import namedtuple

class Row:
    def __init__(self, value, index = None):
        self._value = value
        self._index = index

    def __getitem__(self, index):
        if self._index is None:
            if not type(index) in (int, slice):
                raise ValueError("not valid index")
            return self._value[index]
        else:
            if type(index) in (int, tuple, list, slice, np.int):
                print(index)
                return self._value[index]
            else:
                return self._value[self._index[index]]

    def __setitem__(self, index, value):
        if self._index is None:
            if type(index) != int:
                raise ValueError("not valid index")
            return self._value[index]
        else:
            if type(index) == int:
                self._value[index] = value
            else:
                self._value[self._index[index]] = value

    def __repr__(self):
        return self._value.__repr__()

    @property
    def shape(self):
        return self._value.shape

    def iloc(self, index):
        if self._index is None:
            raise ValueError("'index' has not been defined for the table")
        return self._index[index]

class Table:
    Selector = namedtuple('Selector', ['value', 'context'])

    def __init__(self, properties, size, index = None):
        self.__dict__['shape'] = (len(properties), size,)
        self.setindex(index, self.shape[0])

        for p, dtype in properties.items():
            self.__dict__[p] = Row(np.zeros(size, dtype=dtype), self.__dict__['index'])

    def __setattr__(self, name, value):
        if name not in self.__dict__:
            raise TypeError("'Table' object does not support attribute assignment")
        else:
            if type(value) != type(self.__dict__[name].value):
                raise ValueError("Property '%s' can only be updated not replaced" % name)
            elif value.shape != self.__dict__[name].value.shape:
                raise ValueError("Property '%s' can only be updated not replaced by new size array" % name)
            elif value.dtype != self.__dict__[name].value.dtype:
                raise ValueError("Property '%s' can only be updated not replaced by ndarray of different type" % name)
            else:
                self.__dict__[name].value = value

    @property
    def shape(self):
        return self.__dict__['shape']

    def __repr__(self):
        return "Table <properties: %d, size: %d>" % self.shape

    def setindex(self, index, size=None):
        self._setindex(index, size, '_index')

    def _setindex(self, index, size, _index_name):
        if size == None:
            if _index_name == 'index':
                size = self.shape[0]
            else:
                size = self.shape[1]
        if index:
            if len(index) != size:
                raise ValueError("could not assing index of len (%d) to entry of size (%d)" % (len(index), size))
            self.__dict__[_index_name] = {}
            for i in range(size):
                if not index[i] in self.__dict__[_index_name]:
                    self.__dict__[_index_name][index[i]] = i
                else:
                    raise ValueError("index values have to be unique, '%s' is repeated" % str(index[i]))
        else:
            self.__dict__[_index_name] = None

class Table2D(Table):
    def __init__(self, properties, num_rows, num_cols, index = None):
        self.__dict__['shape'] = (num_rows, num_cols, len(properties))
        self.setindex(index, self.shape[0])

        for p, dtype in properties.items():
            self.__dict__[p] = Row(np.zeros((num_rows, num_cols), dtype=dtype), self.__dict__['_index'])

    def __repr__(self):
        return "Table2D <rows: %d, cols: %d, properties: %d>" % self.shape