from mpi4py import MPI

class CommManager:
    def __init__(self):
        self.MPI = MPI
        self.communicators = {}
        self.intra_communicators = {}
        self.add_communicator('main', MPI.COMM_WORLD)

    def __getitem__(self, index):
        if index in self.intra_communicators:
            return self.intra_communicators[index]
        elif index in self.communicators:
            return self.communicators[index]
        else:
            raise IndexError("(intra)communicator does not exist")

    def add_communicator(self, label, comm):
        if type(comm) == MPI.Intracomm:
            self.intra_communicators[label] = comm
        else:
            self.communicators[label] = comm