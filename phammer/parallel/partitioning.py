import numpy as np

def even(N, k):
    p = np.ones(N); n = N // k; r = N % k
    for i in range(k):
        start = i*n
        end = start + n
        if i < r:
            start += i; end += i
        elif r > 0:
            start += r; end += r
        p[start:end+1] = i
    return p
