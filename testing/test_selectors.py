import numpy as np
from phammer.arrays.selectors import SelectorSet

where = SelectorSet(['node', 'pipe', 'valve'])
where.nodes['are_ghost'] = np.zeros(10, dtype = np.int)
print(where.nodes['are_ghost',])

where += 'pump'
where.nodes
where.nodes['are_ghost',] = 2