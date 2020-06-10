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
eps = np.finfo(float).eps # TODO: Double and single precision







def my_test(test_object, number =100):
    print("\nTesting: "+test_object)
    start = time.time()
    eval(test_object)
    end = time.time()
    print("Elapsed (before compilation) = %.4e" % ((end - start)))

    start = time.time()
    for i in range(number):
        eval(test_object)
    end = time.time()
    print("Elapsed (after compilation) = %.4e" % ((end - start)/number))
    return 

@njit(parallel = True)
# @njit
# def mobility_times_force_wrapper(func, r_vectors, force, eta, a):
def mobility_times_force_wrapper(r_vectors, force, eta, a):
    N = r_vectors.size // 3
    r_vectors = r_vectors.reshape(N, 3)
    force = force.reshape(N, 3)
    u = np.zeros((N, 3))
    for i in prange(N):
        # u_local = np.zeros(3)
        for j in range(N):
            # pass
            # M = func(r_vectors[i,:]-r_vectors[j,:], eta, a)
            # r = r_vectors[i,:]-r_vectors[j,:]
            # M = kernel_mob.no_wall_mobility_trans_trans_numba(r, eta, a)
            M = kernel_mob.no_wall_mobility_trans_trans_numba(r_vectors[i,:]-r_vectors[j,:], eta, a)
            # u[i,:] += M.dot(force[j,:])
            # M = np.ones((3,3))
            # u[i,:] += (M[0,0] * force[j,0] + M[0,1] * force[j,1] + M[0,2] * force[j,2])
            # u[i,:] += (M[1,0] * force[j,0] + M[1,1] * force[j,1] + M[1,2] * force[j,2])
            # u[i,:] += (M[2,0] * force[j,0] + M[2,1] * force[j,1] + M[2,2] * force[j,2])
            for p in range(3):
                for q in range(3):
                    u[i,p] += M[p,q]*force[j,q]
                    # u_local[p] += M[p,q]*force[j,q]

    return u.flatten()

            
if __name__ == "__main__":       
    eta = 1
    a = 1
    L = 10
    N=1000
    r_vectors = np.random.rand(3*N)*L
    force = np.random.rand((3*N))
    number = 10

    print('N={}, repeat={}'.format(N, number))

    # Function wrapper, call Zhe's code to compute mobility matrix at one single place in for loops.
    # test_object_1 = 'mobility_times_force_wrapper(kernel_mob.no_wall_mobility_trans_trans_numba, r_vectors, force, eta, a)'
    test_object_1 = 'mobility_times_force_wrapper(r_vectors, force, eta, a)'
    my_test(test_object_1,number=number)

    # Floren's code, compute velocity of multi-particles
    test_object_2 = 'mob_force.no_wall_mobility_trans_times_force_numba(r_vectors, force, eta, a, np.zeros(3))'
    my_test(test_object_2,number=number)

    # No function wrapper, directly inline code to compute mobility matrix
    test_object_3 = 'kernel_mob.no_wall_mobility_trans_trans_numba_many(r_vectors, eta, a)'
    my_test(test_object_3,number=number)

    # No function wrapper, directly inline code, but do not use huge N*N*3*3 matrix.
    test_object_4 = 'kernel_mob.no_wall_mobility_trans_trans_numba_many_no_matrix(r_vectors, force, eta, a)'
    my_test(test_object_4,number=number)