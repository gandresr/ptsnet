from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print("real %d" % rank)
print(comm, comm.size)
