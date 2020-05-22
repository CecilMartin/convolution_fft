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

# Donev: I would prefer if you use the kernels in mobility_numba.py and just modify that code
# In that code the loop is over pairs of particles. But, you can rewrite this so that:
# 1. The actual kernel code just evaluates the kernel for two points (no loops, simple function that could also be written in C)
#    You don't have to rewrite the kernels, just extract them from inside mobility numba loops and move them to a routine
#    Later we can try to move this to C or Fortran
# 2. The loops over pairs or particles, or over grid points, are in numba, parallel, and they call the kernel from point #1
# 3. One can also use CUDA via numba for example instead of openMP parallel loops. The kernel functon from #1 should not need to change
@jit(nopython = True)
def mobilityUFRPY_x_dir_1(rx, ry): # Donev: Where is this function actually used? Can't find it in the code.
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
