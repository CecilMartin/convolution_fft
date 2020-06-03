import sys
sys.path.append('/home/cecil/Cecil/Projects/convolution_fft/src')
import convolution_fft as conv
import kernel.kernel_func as kernel_mod
import numpy as np
import time
import timeit
from numba import jit




nx, ny, nz = 16, 16, 16
Lx, Ly, Lz = 10,10,10
dx, dy, dz = Lx/nx, Ly/ny, Lz/nz
x = (np.linspace(0, nx-1, nx)+0.5)*dx
y = (np.linspace(0, ny-1, ny)+0.5)*dy
z = (np.linspace(0, nz-1, nz)+0.5)*dz
grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing = 'xy')
periodic = np.array([1, 0, 0])
sigma = 1

@jit(nopython = True)
def kernel_Gauss_3D(x,y,z):
    sigma = 1
    k = 1/(2*np.pi*sigma**2)**(3/2)*np.exp(-(np.power(x,2)+np.power(y,2)+np.power(z,2))/sigma**2)
    return k

kernel = kernel_Gauss_3D


@jit(nopython = True)
def kernel_evaluate_3D_wrapped(Lx, Ly, Lz, grid_x, grid_y, grid_z, kernel, periodic):
    periodic_flag = ~(periodic == 0)
    scalar_d = kernel(grid_x, grid_y, grid_z)
    if periodic_flag[0]:
        for i in range(periodic[0]):
            if periodic_flag[1]:
                for j in range(periodic[1]):
                    if periodic_flag[2]:
                        for k in range(periodic[2]):
                            scalar_d += kernel(grid_x+(i+1)*Lx, grid_y+(j+1)*Ly, grid_z+(k+1)*Lz) + kernel(
                                grid_x+(i+1)*Lx, grid_y+(j+1)*Ly, grid_z-(k+1)*Lz)
                            scalar_d += kernel(grid_x+(i+1)*Lx, grid_y-(j+1)*Ly, grid_z+(k+1)*Lz) + kernel(
                                grid_x+(i+1)*Lx, grid_y-(j+1)*Ly, grid_z-(k+1)*Lz)
                            scalar_d += kernel(grid_x-(i+1)*Lx, grid_y+(j+1)*Ly, grid_z+(k+1)*Lz) + kernel(
                                grid_x-(i+1)*Lx, grid_y+(j+1)*Ly, grid_z-(k+1)*Lz)
                            scalar_d += kernel(grid_x-(i+1)*Lx, grid_y-(j+1)*Ly, grid_z+(k+1)*Lz) + kernel(
                                grid_x-(i+1)*Lx, grid_y-(j+1)*Ly, grid_z-(k+1)*Lz)
                    else:
                        scalar_d += kernel(grid_x+(i+1)
                                            * Lx, grid_y+(j+1)*Ly, grid_z)
                        scalar_d += kernel(grid_x+(i+1)
                                            * Lx, grid_y-(j+1)*Ly, grid_z)
                        scalar_d += kernel(grid_x-(i+1)
                                            * Lx, grid_y+(j+1)*Ly, grid_z)
                        scalar_d += kernel(grid_x-(i+1)
                                            * Lx, grid_y-(j+1)*Ly, grid_z)
            else:
                if periodic_flag[2]:
                    for k in range(periodic[2]):
                        scalar_d += kernel(grid_x+(i+1)*Lx, grid_y, grid_z+(k+1)*Lz) + kernel(
                            grid_x+(i+1)*Lx, grid_y, grid_z-(k+1)*Lz)
                        scalar_d += kernel(grid_x-(i+1)*Lx, grid_y, grid_z+(k+1)*Lz) + kernel(
                            grid_x-(i+1)*Lx, grid_y, grid_z-(k+1)*Lz)
                else:
                    scalar_d += kernel(grid_x+(i+1)*Lx, grid_y, grid_z)
                    scalar_d += kernel(grid_x-(i+1)*Lx, grid_y, grid_z)
    else:
        if periodic_flag[1]:
            for j in range(periodic[1]):
                if periodic_flag[2]:
                    for k in range(periodic[2]):
                        scalar_d += kernel(grid_x, grid_y+(j+1)*Ly, grid_z+(k+1)*Lz) + kernel(
                            grid_x, grid_y+(j+1)*Ly, grid_z-(k+1)*Lz)
                        scalar_d += kernel(grid_x, grid_y-(j+1)*Ly, grid_z+(k+1)*Lz) + kernel(
                            grid_x, grid_y-(j+1)*Ly, grid_z-(k+1)*Lz)
                else:
                    scalar_d += kernel(grid_x, grid_y+(j+1)*Ly, grid_z) + \
                        kernel(grid_x, grid_y-(j+1)*Ly, grid_z)
        else:
            if periodic_flag[2]:
                for k in range(periodic[2]):
                    scalar_d += kernel(grid_x, grid_y, grid_z+(k+1)*Lz) + \
                        kernel(grid_x, grid_y, grid_z-(k+1)*Lz)
    return scalar_d

@jit(nopython = True)
def kernel_evaluate_3D(Lx, Ly, Lz, grid_x, grid_y, grid_z, periodic):
    # @jit(nopython = True)
    def kernel(x,y,z):
        sigma = 1
        k = 1/(2*np.pi*sigma**2)**(3/2)*np.exp(-(np.power(x,2)+np.power(y,2)+np.power(z,2))/sigma**2)
        return k
    periodic_flag = ~(periodic == 0)
    scalar_d = kernel(grid_x, grid_y, grid_z)
    if periodic_flag[0]:
        for i in range(periodic[0]):
            if periodic_flag[1]:
                for j in range(periodic[1]):
                    if periodic_flag[2]:
                        for k in range(periodic[2]):
                            scalar_d += kernel(grid_x+(i+1)*Lx, grid_y+(j+1)*Ly, grid_z+(k+1)*Lz) + kernel(
                                grid_x+(i+1)*Lx, grid_y+(j+1)*Ly, grid_z-(k+1)*Lz)
                            scalar_d += kernel(grid_x+(i+1)*Lx, grid_y-(j+1)*Ly, grid_z+(k+1)*Lz) + kernel(
                                grid_x+(i+1)*Lx, grid_y-(j+1)*Ly, grid_z-(k+1)*Lz)
                            scalar_d += kernel(grid_x-(i+1)*Lx, grid_y+(j+1)*Ly, grid_z+(k+1)*Lz) + kernel(
                                grid_x-(i+1)*Lx, grid_y+(j+1)*Ly, grid_z-(k+1)*Lz)
                            scalar_d += kernel(grid_x-(i+1)*Lx, grid_y-(j+1)*Ly, grid_z+(k+1)*Lz) + kernel(
                                grid_x-(i+1)*Lx, grid_y-(j+1)*Ly, grid_z-(k+1)*Lz)
                    else:
                        scalar_d += kernel(grid_x+(i+1)
                                            * Lx, grid_y+(j+1)*Ly, grid_z)
                        scalar_d += kernel(grid_x+(i+1)
                                            * Lx, grid_y-(j+1)*Ly, grid_z)
                        scalar_d += kernel(grid_x-(i+1)
                                            * Lx, grid_y+(j+1)*Ly, grid_z)
                        scalar_d += kernel(grid_x-(i+1)
                                            * Lx, grid_y-(j+1)*Ly, grid_z)
            else:
                if periodic_flag[2]:
                    for k in range(periodic[2]):
                        scalar_d += kernel(grid_x+(i+1)*Lx, grid_y, grid_z+(k+1)*Lz) + kernel(
                            grid_x+(i+1)*Lx, grid_y, grid_z-(k+1)*Lz)
                        scalar_d += kernel(grid_x-(i+1)*Lx, grid_y, grid_z+(k+1)*Lz) + kernel(
                            grid_x-(i+1)*Lx, grid_y, grid_z-(k+1)*Lz)
                else:
                    scalar_d += kernel(grid_x+(i+1)*Lx, grid_y, grid_z)
                    scalar_d += kernel(grid_x-(i+1)*Lx, grid_y, grid_z)
    else:
        if periodic_flag[1]:
            for j in range(periodic[1]):
                if periodic_flag[2]:
                    for k in range(periodic[2]):
                        scalar_d += kernel(grid_x, grid_y+(j+1)*Ly, grid_z+(k+1)*Lz) + kernel(
                            grid_x, grid_y+(j+1)*Ly, grid_z-(k+1)*Lz)
                        scalar_d += kernel(grid_x, grid_y-(j+1)*Ly, grid_z+(k+1)*Lz) + kernel(
                            grid_x, grid_y-(j+1)*Ly, grid_z-(k+1)*Lz)
                else:
                    scalar_d += kernel(grid_x, grid_y+(j+1)*Ly, grid_z) + \
                        kernel(grid_x, grid_y-(j+1)*Ly, grid_z)
        else:
            if periodic_flag[2]:
                for k in range(periodic[2]):
                    scalar_d += kernel(grid_x, grid_y, grid_z+(k+1)*Lz) + \
                        kernel(grid_x, grid_y, grid_z-(k+1)*Lz)
    return scalar_d

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

test_object = 'kernel_evaluate_3D_wrapped(Lx, Ly, Lz, grid_x, grid_y, grid_z, kernel, periodic)'
my_test(test_object,number=number)

test_object = 'kernel_evaluate_3D(Lx, Ly, Lz, grid_x, grid_y, grid_z, periodic)'
my_test(test_object,number=number)