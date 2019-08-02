import numpy as np
from collections import namedtuple

class Row:
    def __init__(self, value, _super = None):
        self._value = value
        self._super = _super

    def __getitem__(self, index):
        if self._index is None:
            if not type(index) in (int, tuple, list, slice, np.ndarray):
                raise ValueError("not valid index")
            return self._value[index]
        else:
            if type(index) in (int, tuple, list, slice, np.ndarray):
                return self._value[index]
            else:
                return self._value[self._index[index]]

    def __setitem__(self, index, value):
        if self._index is None:
            if not type(index) in (int, tuple, list, slice, np.ndarray):
                raise ValueError("not valid index")
            self._value[index] = value
        else:
            if type(index) in (int, tuple, list, slice, np.ndarray):
                self._value[index] = value
            else:
                self._value[self._index[index]] = value

    @property
    def _index(self):
        return self._super._index

class Table:
    Selector = namedtuple('Selector', ['value', 'context'])

    def __init__(self, properties, size, index = None):
        self.__dict__['shape'] = (len(properties), size,)
        self.setindex(index, self.shape[1])

        for p, dtype in properties.items():
            self.__dict__[p] = Row(np.zeros(size, dtype=dtype), self)

    def __getattribute__(self, name):
        if name == '__dict__':
            return object.__getattribute__(self, name)
        else:
            if name in self.__dict__:
                if type(self.__dict__[name]) == Row:
                    return self.__dict__[name]._value
                return self.__dict__[name]
            return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if name not in self.__dict__:
            raise TypeError("'Table' does not support attribute assignment")
        else:
            old_val = self[name]
            new_val = value

            if type(new_val) == Row:
                new_val = new_val._value

            if type(old_val) != type(new_val):
                raise ValueError("Property '%s' can only be updated not replaced" % name)
            elif old_val.shape != new_val.shape:
                raise ValueError("Property '%s' can only be updated not replaced by new size array" % name)
            elif old_val.dtype != new_val.dtype:
                raise ValueError("Property '%s' can only be updated not replaced by ndarray of different type" % name)
            else:
                old_val[:] = new_val

    @property
    def shape(self):
        return self.__dict__['shape']

    def __repr__(self):
        return "Table <properties: %d, size: %d>" % self.shape

    def setindex(self, index, size=None):
        self._setindex(index, size, '_index')

    def iloc(self, index):
        if self._index is None:
            raise ValueError("'index' has not been defined for the table")
        return self._index[index]

    def ival(self, index):
        if self._index is None:
            raise ValueError("'index' has not been defined for the table")
        if type(index) != int:
            raise ValueError("'%s' has to be a integer" % str(index))
        return self._index_keys[index]

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
            self.__dict__[_index_name + '_keys'] = index
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
            self.__dict__[p] = Row(np.zeros((num_rows, num_cols), dtype=dtype), self)

    def __repr__(self):
        return "Table2D <rows: %d, cols: %d, properties: %d>" % self.shape