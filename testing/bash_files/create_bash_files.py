from math import ceil

for i in range(1,1001):
    with open('sjob_%d.sh' % i, 'w') as f:
        x = '''#!/bin/bash

#SBATCH -J tphammer{p}     # job name
#SBATCH -o tphammer{p}.o%j # output and error file name (%j expands to j$
#SBATCH -N {nodes}               # number of nodes requested
#SBATCH -n {p}                # total number of mpi tasks requested
#SBATCH -p normal           # queue (partition) -- normal, developmen$
#SBATCH -t 00:20:00         # run time (hh:mm:ss) - 1.5 hours
#SBATCH -A Urban-Stormwater-Mod
#SBATCH --mail-user=griano@utexas.edu

# run the executable named a.out
ibrun python ../test_hammer_sim.py
'''.format(p = i, nodes = ceil(i/64))
        f.write(x)
