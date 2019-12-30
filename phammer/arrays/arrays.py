import numpy as np

from collections import namedtuple
from phammer.simulation.util import is_iterable
from functools import lru_cache

class Row(np.ndarray):
    def __new__(subtype, shape, dtype=float, _super=None):
        obj = super(Row, subtype).__new__(subtype, shape, dtype)
        obj._super = _super
        obj.fill(0)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self._super = getattr(obj, '_super', None)

    def __getitem__(self, index):
        if type(index) == str:
            return super().__getitem__(self.index[index])
        else:
            return super().__getitem__(index)

    def __setitem__(self, index, value):
        if type(index) == str:
            super().__setitem__(self.index[index], value)
        else:
            super().__setitem__(index, value)

    @property
    def index(self):
        return self._super._index

class Table:
    def __init__(self, properties, size, index = None):
        self.__dict__['shape'] = (len(properties), size,)
        self.__dict__['properties'] = properties
        self.setindex(index, self.shape[1])

        for p, dtype in self.properties.items():
            self.__dict__[p] = Row(size, dtype=dtype, _super=self)

    def __setattr__(self, name, value):
        if name not in self.__dict__:
            raise TypeError("'Table' does not support attribute assignment")
        else:
            old_val = self.__dict__[name]
            new_val = value

            if type(old_val) != type(new_val):
                raise ValueError("Property '%s' can only be updated not replaced" % name)
            elif old_val.shape != new_val.shape:
                raise ValueError("Property '%s' can only be updated not replaced by new size array" % name)
            else:
                old_val[:] = new_val

    def __getitem__(self, index):
        if not ((is_iterable(index) and type(index) != str) or type(index) == slice):
            raise ValueError("index is not valid, only numpy valid iterable indices")
        idx = index
        if type(index) == slice:
            idx = range(*index.indices(self.shape[1]))
        new_index = None
        if hasattr(self, '_index_keys'):
            new_index = self._index_keys[idx]
        sliced_table = Table(self.properties, len(idx), new_index)
        for p in self.properties:
            sliced_table.__dict__[p][:] = self.__dict__[p][idx]

        return sliced_table

    @property
    def shape(self):
        return self.__dict__['shape']

    def __repr__(self):
        return "<Table properties: %d, size: %d>" % self.shape

    def setindex(self, index, size=None):
        self._setindex(index, size, '_index')

    # @lru_cache(maxsize=64)
    def iloc(self, index):
        if self._index is None:
            raise ValueError("'index' has not been defined for the table")
        if is_iterable(index):
            return [self._index[i] for i in index]
        return self._index[index]

    # @lru_cache(maxsize=64)
    def ival(self, index):
        if self._index is None:
            raise ValueError("'index' has not been defined for the table")
        return self._index_keys[index]

    def _setindex(self, index, size, _index_name):
        if size == None:
            if _index_name == 'index':
                size = self.shape[0]
            else:
                size = self.shape[1]
        if not index is None:
            if len(index) != size:
                raise ValueError("could not assing index of len (%d) to entry of size (%d)" % (len(index), size))
            self.__dict__[_index_name] = {}
            self.__dict__[_index_name + '_keys'] = np.array(index)
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
        self.__dict__['properties'] = properties
        self.setindex(index, self.shape[0])

        for p, dtype in self.properties.items():
            self.__dict__[p] = Row((num_rows, num_cols), dtype=dtype, _super=self)

    def __repr__(self):
        return "<Table2D rows: %d, cols: %d, properties: %d>" % self.shape

    def __getitem__(self, index):
        if not ((is_iterable(index) and type(index) != str) or type(index) == slice):
            raise ValueError("index is not valid, only numpy valid iterable indices")
        idx = index
        if type(index) == slice:
            idx = range(*index.indices(self.shape[1]))
        new_index = None
        if hasattr(self, '_index_keys'):
            new_index = self._index_keys[idx]
        sliced_table = Table2D(self.properties, len(idx), self.shape[1], new_index)
        for p in self.properties:
            sliced_table.__dict__[p][:] = self.__dict__[p][idx]

            return sliced_table

class ObjArray:
    def __init__(self):
        self.index = {}
        self.keys = []
        self.values = []

    def __setitem__(self, name, value):
        if name in self.index:
            self.values[self.index[name]] = value
        else:
            self.index[name] = len(self)
            self.values.append(value)
            self.keys.append(name)

    def __getitem__(self, index):
        return self.values[self.index[index]]

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        self.i += 1
        if (self.i - 1) < len(self):
            return self.values[self.i - 1]
        else:
            raise StopIteration

    def __repr__(self):
        return "<ObjArray " + str(self.index) + " >"

    def iloc(self, index):
        return self.index[index]
