import sys
sys.path.append('/home/cecil/Cecil/Projects/convolution_fft/src')
import convolution_fft as conv
import kernel.mobility_numba as kernel_mod
import numpy as np
import time
import timeit
from numba import jit




eta = 1
a = 1
L = 10
r_vectors = np.random.rand(3)*L






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



# test_object = 'conv.vel_direct_convolution(scalar,source_strenth,kernel_handle, L, periodic)'
number = 100

test_object = 'kernel_mod.no_wall_mobility_trans_trans_numba(r_vectors, eta, a)'
my_test(test_object,number=number)

