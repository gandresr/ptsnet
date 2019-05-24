import ctypes
import multiprocessing as mp
import numpy as np

def dot_slice(a, b, out, i):
    t = np.empty_like(a[i,:])
    for j in range(b.shape[1]):
        # out[i,j] = sum(a[i,:] * b[:,j])
        np.multiply(a[i,:], b[:,j], t).sum(axis=1, out=out[i,j])

def dot(a, b, nprocesses=mp.cpu_count()):
    """Perform matrix multiplication using multiple processes."""
    if (a.shape[1] != b.shape[0]):
        raise ValueError("wrong shape")

    # create shared array
    mp_arr = mp.RawArray(ctypes.c_double, a.shape[0]*b.shape[1])

    # start processes
    np_args = mp_arr, (a.shape[0], b.shape[1]), a.dtype
    pool = mp.Pool(nprocesses, initializer=init, initargs=(a, b)+np_args)

    # perform multiplication
    for i in pool.imap_unordered(mpdot_slice, slices(a.shape[0], nprocesses)):
        print("done %s" % (i,))
    pool.close()
    pool.join()

    # return result
    return tonumpyarray(*np_args)

def mpdot_slice(i):
    dot_slice(ga, gb, gout, i)
    return i

def init(a, b, *np_args):
    """Called on each child process initialization."""
    global ga, gb, gout
    ga, gb = a, b
    gout = tonumpyarray(*np_args)

def tonumpyarray(mp_arr, shape, dtype):
    """Convert shared multiprocessing array to numpy array.

    no data copying
    """
    return np.frombuffer(mp_arr, dtype=dtype).reshape(shape)

def slices(nitems, mslices):
    """Split nitems on mslices pieces.

    >>> list(slices(10, 3))
    [slice(0, 4, None), slice(4, 8, None), slice(8, 10, None)]
    >>> list(slices(1, 3))
    [slice(0, 1, None), slice(1, 1, None), slice(2, 1, None)]
    """
    step = nitems // mslices + 1
    for i in range(mslices):
        yield slice(i*step, min(nitems, (i+1)*step))

def test():
    n = 100000
    a = np.random.rand(50, n)
    b = np.random.rand(n, 60)
    np.dot(a,b)
    # assert np.allclose(np.dot(a,b), dot(a,b, nprocesses=1))

test()