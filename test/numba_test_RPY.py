from __future__ import division, print_function
import sys
sys.path.append('/home/cecil/Cecil/Projects/convolution_fft/src')
import convolution_fft as conv
import kernel.kernel_func as kernel_mod
import numpy as np
import time
import timeit
from numba import jit
nj = 16
Lx, Ly = [10, 10]
eps = 1e-8
nx = ny = int(np.ceil(np.sqrt(nj)*5))
x = (np.random.rand(nj))*Lx
y = (np.random.rand(nj))*Ly
scalar = np.array([x,y])
source_strenth = np.ones(nj)
L = np.array([Lx, Ly])
periodic = np.array([1, 0])
eta = 1
a = 1
@jit(nopython = True)
def kernel_handle(x,y):
    x_inva = x/a
    y_inva = y/a
    M = kernel_mod.mobilityUFRPY_x_dir_2(x_inva, y_inva)
    return M


# @jit(nopython = True, parallel = True, cache = True)
# def test_func(f,x,y): return f(x,y)
# test_object = 'test_func(kernel_handle, x, y)' 
test_object = 'conv.kernel_evaluate(np.array([x,y]), kernel_handle, periodic, L)'
# test_object = 'conv.vel_direct_convolution(scalar,source_strenth,kernel_handle, L, periodic)'
number = 100
start = time.time()
eval(test_object)
end = time.time()
print("Elapsed (before compilation) = %s" % ((end - start)))

start = time.time()
for i in range(number):
    eval(test_object)
end = time.time()
print("Elapsed (after compilation) = %s" % ((end - start)/number))
# print(timeit.timeit('v_direct = conv.vel_direct_convolution(scalar,source_strenth,kernel_handle, L, periodic)', globals=globals()),number = 100)

