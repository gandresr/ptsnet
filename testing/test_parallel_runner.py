from phammer.simulation.util import run_shell
from pkg_resources import resource_filename
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
path = resource_filename(__name__, 'phammer/parallel/run_parallel.py')
run_shell("mpiexec -n 4 python %s" % path)
print("main %d" % rank)
print(comm, comm.size)

