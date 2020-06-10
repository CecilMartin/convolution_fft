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
N=1000
r_vectors = np.random.rand(3*N)*L
force = np.random.rand(3*N)
M = np.random.rand(3,3)
u = np.zeros((N, 3))

def my_test(test_object, number =100):
    print("\nTesting: "+test_object)
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


@njit(parallel = True)
# @njit
def local_vector(force, M):
    N = force.size // 3
    force = force.reshape(N, 3)
    # u = np.zeros((N, 3))
    for i in prange(N):
        for j in range(N):
            u_local = M.dot(force[j,:])
    return

@njit(parallel = True)
# @njit
def local_vector_my_matrix_product(force, M):
    N = force.size // 3
    force = force.reshape(N, 3)
    # u = np.zeros((N, 3))
    for i in prange(N):
        for j in range(N):
            u_local = np.zeros(3)
            for p in range(3):
                for q in range(3):
                    u_local[p] += M[p,q]*force[j,q]
    return

(Mxx, Mxy, Mxz), (Myx, Myy, Myz), (Mzx, Mzy, Mzz) = M
@njit(parallel = True)
# @njit
def local_vector_my_matrix_product_no_index(force, Mxx, Mxy, Mxz, Myx, Myy, Myz, Mzx, Mzy, Mzz):
    N = force.size // 3
    force = force.reshape(N, 3)
    # u = np.zeros((N, 3))
    for i in prange(N):
        ux = uy = uz = 0
        for j in range(N):
            ux += (Mxx * force[j,0] + Mxy * force[j,1] + Mxz * force[j,2])
            uy += (Myx * force[j,0] + Myy * force[j,1] + Myz * force[j,2])
            uz += (Mzx * force[j,0] + Mzy * force[j,1] + Mzz * force[j,2])
    return

@njit(parallel = True)
# @njit
def global_vector(force, M, u):
    N = force.size // 3
    force = force.reshape(N, 3)
    for i in prange(N):
        for j in range(N):
            u[i,:] = M.dot(force[j,:])
    return
            
            



number = 4

# Test 1
test_object_1 = 'local_vector(force, M)'
my_test(test_object_1,number=number)

# Test 2
test_object_2 = 'global_vector(force, M, u)'
my_test(test_object_2,number=number)

# Test 3
test_object_3 = 'local_vector_my_matrix_product(force, M)'
my_test(test_object_3,number=number)

# Test 4
test_object_4 = 'local_vector_my_matrix_product_no_index(force, Mxx, Mxy, Mxz, Myx, Myy, Myz, Mzx, Mzy, Mzz)'
my_test(test_object_4,number=number)