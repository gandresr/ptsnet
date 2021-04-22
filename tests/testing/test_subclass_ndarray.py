import numpy as np



index = {
    'b' : 0,
    'a' : 1,
}

x = Row(10, dtype = np.int, index = index)
x['a']