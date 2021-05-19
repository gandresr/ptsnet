import numpy as np

from collections import namedtuple
from ptsnet.utils.data import is_array
from functools import lru_cache
from h5py import Dataset

class Row(np.ndarray):
    def __new__(subtype, shape, dtype=float, desc = None, _super=None):
        obj = super(Row, subtype).__new__(subtype, shape, dtype)
        obj._super = _super
        obj.fill(0)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self._super = getattr(obj, '_super', None)

    def __getitem__(self, index):
        if type(index) == str: # label
            return super().__getitem__(self._super.indexes[index])
        else:
            return super().__getitem__(index)

    def __setitem__(self, index, value):
        if type(index) == str: # label
            super().__setitem__(self._super.indexes[index], value)
        else:
            super().__setitem__(index, value)

class Table:
    def __init__(self, properties, num_rows, labels = None, allow_replacement = False, persistent = False):
        self.__dict__['shape'] = (len(properties), num_rows,)
        self.__dict__['properties'] = properties
        self.__dict__['allow_replacement'] = allow_replacement
        self.__dict__['labels'] = None
        self.__dict__['indexes'] = None
        self.assign_labels(labels, dim = 1)

        for p, dtype in self.properties.items():
            row = Row(num_rows, dtype=dtype, _super=self) if not persistent else None
            self.__dict__[p] = row

    def __setattr__(self, name, value):
        if not hasattr(self, name):
            raise TypeError("'Table' does not support attribute assignment")
        else:
            if not self.allow_replacement:
                old_val = self.__dict__[name]
                new_val = value

                if type(old_val) != type(new_val):
                    raise ValueError("Property '%s' can only be updated not replaced: types do not coincide" % name)
                elif old_val.shape != new_val.shape:
                    raise ValueError("Property '%s' can only be updated not replaced: shapes do not coincide" % name)
                else:
                    old_val[:] = new_val
            else:
                if type(value) == Dataset:
                    self.__dict__[name] = PersistentArray(value, _super=self)
                else:
                    raise TypeError("'Table' does not support attribute assignment")

    def __getitem__(self, indexes):
        if not (is_array(indexes) or type(indexes) == slice):
            raise ValueError("index is not valid, use iterable or slice")
        slice_indexes = index
        if type(index) == slice:
            slice_indexes = range(*index.indices(self.shape[1]))
        slice_labels = None
        if hasattr(self, 'labels'):
            slice_labels = self.labels[slice_indexes]
        sliced_table = Table(self.properties, len(slice_indexes), slice_labels)
        for p in self.properties:
            sliced_table.__dict__[p][:] = self.__dict__[p][slice_indexes]

        return sliced_table

    def __repr__(self):
        return "<Table properties: %d, size: %d, labeled: %s>" % (self.shape + (not self.indexes is None,))

    # @lru_cache(maxsize=64)
    def lloc(self, label):
        if self.indexes is None:
            raise ValueError("labels have not been defined for the table")
        if is_array(label):
            return [self.indexes[l] for l in label]
        return self.indexes[label]

    # @lru_cache(maxsize=64)
    def ilabel(self, index):
        if self.indexes is None:
            raise ValueError("labels have not been defined for the table")
        if is_array(index):
            return [self.labels[i] for i in index]
        return self.labels[index]

    def assign_labels(self, labels, dim = 1):
        '''
        It is assumed that the labels are ordered therefore, int indexes in the table
        match the order of the labels
        '''
        if dim == 1:
            size = self.shape[1]
        elif dim == 2:
            size = self.shape[0]
        else:
            raise ValueError('dimension not supported')

        if not labels is None:
            if len(labels) != size:
                raise ValueError("could not assing labels of len (%d) to table with (%d) rows" % (len(labels), size))
            self.__dict__['indexes'] = {}
            self.__dict__['labels'] = np.array(labels)
            for i in range(size):
                if not labels[i] in self.indexes:
                    self.indexes[labels[i]] = i
                else:
                    raise ValueError("index values have to be unique, '%s' is repeated" % str(labels[i]))

class Table2D(Table):
    def __init__(self, properties, num_rows, num_cols, labels = None, allow_replacement = False, persistent = False):
        self.__dict__['shape'] = (num_rows, num_cols, len(properties))
        self.__dict__['properties'] = properties
        self.__dict__['allow_replacement'] = allow_replacement
        self.__dict__['labels'] = None
        self.__dict__['indexes'] = None
        self.assign_labels(labels, dim = 2)

        for p, dtype in self.properties.items():
            row = Row((num_rows, num_cols), dtype=dtype, _super=self) if not persistent else None
            self.__dict__[p] = row

    def __repr__(self):
        return "<Table2D rows: %d, cols: %d, properties: %d, labeled: %s>" % (self.shape + (not self.labels is None,))

    def __getitem__(self, index):
        if not (is_array(index) or type(index) == slice):
            raise ValueError("index is not valid, use iterable or slice")
        slice_index = index
        if type(index) == slice:
            slice_index = range(*index.indices(self.shape[1]))
        slice_labels = None
        if hasattr(self, 'labels'):
            slice_labels = self.labels[slice_index]
        sliced_table = Table2D(self.properties, len(slice_index), self.shape[1], slice_labels)
        for p in self.properties:
            sliced_table.__dict__[p][:] = self.__dict__[p][slice_index]
            return sliced_table

class ObjArray:
    def __init__(self):
        self.indexes = {}
        self.labels = []
        self.values = []

    def __setitem__(self, label, value):
        if label in self.indexes:
            self.values[self.indexes[label]] = value
        else:
            self.indexes[label] = len(self)
            self.values.append(value)
            self.labels.append(label)

    def __getitem__(self, index):
        return self.values[self.indexes[index]]

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
        return "<ObjArray " + str(self.indexes) + " >"

    def lloc(self, label):
        return self.indexes[label]

class PersistentArray:
    def __init__(self, array, _super = None):
        self.__dict__['array'] = array
        self.__dict__['_super'] = _super

    def __setattr__(self, name, value):
        if not hasattr(self, name):
            raise TypeError("'Table' does not support attribute assignment")

    def __getitem__(self, index):
        if type(index) == str: # label
            return self.array[self._super.indexes[index]]
        else:
            return self.array[index]

    def __str__(self):
        s = str(self.array)
        return s.replace('<HDF5 dataset', '<PersistentArray')

    @property
    def shape(self):
        return self.array.shape