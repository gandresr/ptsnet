import random
from threading import Thread
from numba import njit
from multiprocessing import Process
import numpy as np

size = 10000000   # Number of random numbers to add to list
threads = 40 # Number of threads to create

def func(init, end, mylist):
    count = 0
    for i in range(init, end+1):
        count += my_list[i]
    print(count)

@njit(parallel = True)
def simple():
    my_list = np.random.random(size*threads)
    count = 0
    for i in range(len(my_list)):
        count += my_list[i]
    print(count)
    


def multiprocessed():
    processes = []
    for i in range(0, threads):
        p = Process(target=func,args=(i*size, size*(i+1)-1, my_list))
        processes.append(p)
    # Start the processes
    for p in processes:
        p.start()
    # Ensure all processes have finished execution
    for p in processes:
        p.join()
if __name__ == "__main__":
    #multithreaded()
    simple()
    #multiprocessed()