import numpy as np
from numba import jit, prange

@jit(nopython = True)
def kernel_uniform_2D(x,y):
    return np.ones(x.shape)

@jit(nopython = True)
def kernel_uniform_1D(x):
    return np.ones(x.shape)


@jit(nopython = True)
def kernel_uniform_3D(x,y,z):
    return np.ones(x.shape)

@jit(nopython = True)
# @jit(nopython = True, parallel = True)
def kernel_Gauss_2D(x,y,sigma):
    k = 1/(2*np.pi*sigma**2)*np.exp(-(np.power(x,2)+np.power(y,2))/sigma**2)
    return k

@jit(nopython = True)
def mobilityUFRPY_x_dir_1(rx, ry):
    # rx, ry = rx/a, ry/a.
    # r>=2
    r2 = rx*rx + ry*ry
    r = np.sqrt(r2)
    invr = 1 / r
    invr2 = invr * invr
    c1 = 1+2/(3*r2)
    c2 = (1 - 2 * invr2) * invr2
    Mxx = (c1 + c2*rx*rx) * invr
    return Mxx

@jit(nopython = True)
def mobilityUFRPY_x_dir_2(rx, ry):
    # rx, ry = rx/a, ry/a.
    # r<2
    fourOverThree = 4/3
    r2 = rx*rx + ry*ry
    r = np.sqrt(r2)
    invr = 1 / r
    c1 = fourOverThree * (1 - 0.28125 * r) #9/32 = 0.28125
    c2 = fourOverThree * 0.09375 * invr #3/32 = 0.09375
    Mxx = c1 + c2 * rx*rx
    return Mxx