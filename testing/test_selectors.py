import numpy as np
from phammer.arrays.selectors import SelectorSet

where = SelectorSet(['nodes', 'pipes', 'valves'])
where.nodes['are_ghost'] = np.zeros(10, dtype = np.int)
print(where.nodes['are_ghost',])

where += 'pumps'
where.nodes
where.nodes['are_ghost',] = 2