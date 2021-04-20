import numpy as np

from ptsnet.arrays import Row

class SelectorList:
    '''
    SelectorList Class

    A SelectorList is an array-like object that stores selectors

    A Selector is an array of (int) indexes used to describe the location of
        objects or object properties in a particular array. For instance,
        selectors can be used to extract subsets of objects stored in a
        numpy array like R = [.1, .2, .45, .9]. Let's say that it is necessary
        for some application to use the subset P of numbers greater than 0.5
        in R several times. If you do not want to create a separate float array
        or performing a > operation everytime P is needed, a selector can be
        created as follows:

        P_selector = np.where(R > 0.5)[0]

    With P_selector, one can simply run R[P_selector] everytime the set P
        is equired. This is more efficient than R[R > 0.5]. Also, when the indexing
        rule becomes long and complicated a selector becomes useful to keep
        the code DRY and clean.

    However, having multiple selectors can make your code less organized, that
        is why, SelectorLists are useful. They allow you to group selectors.

    A SelectorList can be defined as follows:

        numbers = SelectorList()

    Now, you can add P_selector to numbers

        numbers['>0.5'] = np.where(R > 0.5)[0]

    Also, selectors may have a context, which is an array containing information
        that complements the usability of the selector, i.e., it is an array that
        makes a particular selector useful.

    Back to our example, let's say that now the set P needs to be stored in another
        array X = [0, 1, 0, 0, 1, 0], but in specific indexes of X, say where X is
        equal to 1. Again, one can simply do

        X[X == 1] = R[R > 0.5]

        but this can be inneficient if such operation is the bottleneck in your code.

    Therefore, a more efficient approach is to store separately X == 1 as the context
        of P_selector in the SelectorList as follows

        numbers['>0.5',] = X[X == 1]

        recall that adding a comma ',' to the index of 'numbers' means you refer to the
        context of the selector labeled as '>0.5'.

    Finally, the whole operation with the selector and its context should look like this

        X[numbers['>0.5',]] = R[numbers['>0.5']]
    '''

    def __init__(self):
        self.__dict__['_selectors'] = {}
        self.__dict__['_contexts'] = {}

    def __len__(self):
        return len(self.__dict__['_selectors'])

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
            if (type(value) == np.ndarray and value.dtype not in (np.int, np.bool)) or \
                (not type(value) in (Row, np.ndarray)) or (type(value) == Row and value.dtype not in (np.int, np.bool)):
                    raise ValueError("'selector' is not valid: Only numpy arrays of dtype = int/bool")
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

    @property
    def categories(self):
        return list(self.__dict__.keys())

    def append(self, category):
        self.__dict__[category] = SelectorList()

    def __repr__(self):
        rep = 'Sets of selectors: \n\n'
        for category in self.categories:
            rep += '[{n}] '.format(n = len(self.__dict__[category])) + category + '\n'
        return rep