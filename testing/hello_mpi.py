# from mpi4py import MPI
import wntr, phammer
print('hey')
from mpi4py import MPI
print('hey2')

comm = MPI.COMM_WORLD

print( "Hello ! I’m rank %02d from %02d" % (comm.rank, comm.size) )

# print( "Hello ! I’m rank %02d from %02d" % (comm.Get_rank() , comm.Get_size()) )

# print( "Hello ! I’m rank %02d from %02d" % (MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()) )
