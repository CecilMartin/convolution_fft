import sys
sys.path.append('/home/cecil/Cecil/Projects/convolution_fft/src')
import convolution_fft as conv
import kernel.mobility_numba as kernel_mob
import kernel.mobility_times_force_numba as mob_force
import numpy as np
import time
import timeit
from numba import njit, prange, jit
import scipy.sparse




eta = 1
a = 1
L = 10
N=100
r_vectors = np.random.rand(3*N)*L
force = np.random.rand((3*N))






def my_test(test_object, number =100):
    print("Testing: "+test_object)
    start = time.time()
    eval(test_object)
    end = time.time()
    print("Elapsed (before compilation) = %s" % ((end - start)))

    start = time.time()
    for i in range(number):
        eval(test_object)
    end = time.time()
    print("Elapsed (after compilation) = %s" % ((end - start)/number))
    return 

@njit
def mobility_times_force_wrapper(func, r_vectors, force, eta, a):
    N = r_vectors.size // 3
    r_vectors = r_vectors.reshape(N, 3)
    force = force.reshape(N, 3)
    u = np.zeros((N, 3))
    for i in range(N):
        for j in range(N):
            M = func(r_vectors[i,:]-r_vectors[j,:], eta, a)
            u[i,:] += M.dot(force[j,:])
    return u.flatten()
            
            


# test_object = 'conv.vel_direct_convolution(scalar,source_strenth,kernel_handle, L, periodic)'
number = 100

# Function wrapper
test_object_1 = 'mobility_times_force_wrapper(kernel_mob.no_wall_mobility_trans_trans_numba, r_vectors, force, eta, a)'
my_test(test_object_1,number=number)

# Floren's code
test_object_2 = 'mob_force.no_wall_mobility_trans_times_force_numba(r_vectors, force, eta, a, np.zeros(3))'
my_test(test_object_2,number=number)

# No function wrapper
test_object_3 = 'kernel_mob.no_wall_mobility_trans_trans_numba_many(r_vectors, eta, a)'
my_test(test_object_3,number=number)

