import numba, timeit
import numpy as np
@numba.jit(nopython=True)
def sqeuclidean(x, y):
    distance = 0.
    for n in range(x.shape[0]):
        distance += (x[n] - y[n])**2
    return distance

@numba.jit(nopython=True)
def cdist(x, y):
    D = np.zeros((x.shape[0], y.shape[0]))
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            for d in range(x.shape[1]):
                D[i, j] += (x[i, d] - y[i, d])**2
    return D

@numba.jit(nopython=True)
def cdist_func(x, y):
    D = np.empty((x.shape[0], y.shape[0]))
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            D[i, j] = sqeuclidean(x[i], y[i])
    return D

a = np.random.random_sample((1000, 10))
b = np.random.random_sample((1000, 10))
cdist(a,b)
print(timeit.timeit('cdist(a, b)', number = 100, setup="from __main__ import cdist, a, b"))
print(timeit.timeit('cdist(a, b)', number = 100, setup="from __main__ import sqeuclidean, cdist_fun, a, b"))

# %timeit cdist_func(a, b)
