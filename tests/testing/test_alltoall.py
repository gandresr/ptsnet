from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

a_size = 1
senddata = np.array([1,2,3,4])
recvdata = np.array([0,0,0,0])
comm.Alltoall(senddata, recvdata)

print("process %s sending %s receiving %s " % (rank,senddata,recvdata))