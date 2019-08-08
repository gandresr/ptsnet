import numpy as np

from phammer.arrays.arrays import Row

class SelectorList:
    def __init__(self):
        self.__dict__['_selectors'] = {}
        self.__dict__['_contexts'] = {}

    def __setattr__(self, name, value):
        raise TypeError("Attribute assignment is not suported for SelectorList")

    def __getitem__(self, index):
        if type(index) == tuple:
            if len(index) != 1:
                raise KeyError("In order to get context use [%s,]" % index[0])
            else:
                if index[0] in self._contexts:
                    return self._contexts[index[0]]
                else:
                    raise KeyError("No context defined for '%s' selector" % index[0])
        else:
            if index in self._selectors:
                return self._selectors[index]
            else:
                raise KeyError("Selector '%s' has not been defined" % index)


    def __setitem__(self, index, value):
        if type(index) == tuple:
            if len(index) > 1:
                raise ValueError("In order to set context use [%s,]" % index[0])
            else:
                self._contexts[index[0]] = value
        else:
            if (type(value) == np.ndarray and value.dtype != np.int) or \
                (not type(value) in (Row, np.ndarray)) or (type(value) == Row and value.dtype != np.int):
                    raise ValueError("'selector' is not valid: Only numpy arrays of dtype = int")
            self._selectors[index] = value

    def __repr__(self):
        rep = "\nSelectors: \n"
        for selector in self.__dict__['_selectors']:
            c = '' if not selector in self.__dict__['_contexts'] else self.__dict__['_contexts'][selector]
            has_context = ('<context : %s>' % c) if selector in self.__dict__['_contexts'] else ''
            rep += "  * '" + selector + ("' %s" % has_context) + '\n'
        return rep

class SelectorSet:
    def __init__(self, categories):
        for category in categories:
            self.__dict__[category] = SelectorList()

    def __setattr__(self, name, value):
        raise TypeError("Attribute assignment is not suported for SelectorSet")

    def __iadd__(self, category):
        self.append(category)
        return self

    def append(self, category):
        self.__dict__[category] = SelectorList()