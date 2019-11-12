from __future__ import division, print_function
import numpy as np
import copy
import multiprocessing
import pyfftw
# Configure PyFFTW to use all cores (the default is single-threaded)
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
# pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'

# These lines set the precision of the cuda code
# to single or double. Set the precision
# in the following lines and edit the lines
# after   'mod = SourceModule("""'    accordingly
# precision = 'single'
precision = 'double'


def real(x):
    if precision == 'single':
        return np.float32(x)
    else:
        return np.float64(x)


def vel_convolution_fft(scalar, L, x, kernel, periodic, method=1, fft_object=None, ifft_object=None, *args, **kwargs):
    Nshape = scalar.shape
    dim = len(Nshape)
    if dim == 2:
        [ny, nx] = Nshape[:]
        [Lx, Ly] = L[:]
        [x, y] = x[:]
        if method == 1:
            scalar_d = np.zeros([2*ny, 2*nx])
            scalar_d[:ny, :nx] = scalar
            y_k_d = kernel_evaluate_2D(x, y, kernel, periodic, Lx, Ly)
            # fftw objects, either a global object or created at every call
            if (fft_object == None) & (ifft_object == None):
                a = pyfftw.empty_aligned((2*ny, 2*nx), dtype='float64')
                # Save efforts by knowing that a is real
                b = pyfftw.empty_aligned((2*ny, nx+1), dtype='complex128')
                # Real to complex FFT Over the both axes
                fft_object = pyfftw.FFTW(a, b, axes=(
                    0, 1), flags=('FFTW_MEASURE', ))
                ifft_object = pyfftw.FFTW(b, a, axes=(
                    0, 1), direction='FFTW_BACKWARD', flags=('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'))
            else:
                a = fft_object.input_array
                b = fft_object.output_array
            a[:][:] = scalar_d
            scalar_d_hat = copy.deepcopy(fft_object())
            a[:][:] = y_k_d
            y_k_d_hat = copy.deepcopy(fft_object())
            b[:][:] = scalar_d_hat * y_k_d_hat
            v = copy.deepcopy(ifft_object())
            return v[:ny, :nx]

    elif dim == 3:
        raise Exception('Haven\'t implemented convolution for %dD!' % dim)
    else:
        raise Exception('Haven\'t implemented convolution for %dD!' % dim)
    return 0


def kernel_evaluate_2D(x, y, kernel, periodic, Lx, Ly):
    nx = x.size
    ny = y.size
    x_d = np.concatenate((x, x-Lx), axis=None)
    y_d = np.concatenate((y, y-Ly), axis=None)
    grid_x, grid_y = np.meshgrid(x_d, y_d)
    periodic_flag = ~(periodic == 0)
    scalar_d = kernel(grid_x, grid_y)
    if periodic_flag[0]:
        for i in range(periodic[0]):
            if periodic_flag[1]:
                for j in range(periodic[1]):
                    scalar_d = scalar_d + \
                        kernel(grid_x+(i+1)*Lx, grid_y+(j+1)*Ly)
                    scalar_d = scalar_d + \
                        kernel(grid_x+(i+1)*Lx, grid_y-(j+1)*Ly)
                    scalar_d = scalar_d + \
                        kernel(grid_x-(i+1)*Lx, grid_y+(j+1)*Ly)
                    scalar_d = scalar_d + \
                        kernel(grid_x-(i+1)*Lx, grid_y-(j+1)*Ly)
            else:
                scalar_d = scalar_d + kernel(grid_x+(i+1)*Lx, grid_y)
                scalar_d = scalar_d + kernel(grid_x-(i+1)*Lx, grid_y)
    else:
        if periodic_flag[1]:
            for j in range(periodic[1]):
                scalar_d = scalar_d + kernel(grid_x, grid_y+(j+1)*Ly)
                scalar_d = scalar_d + kernel(grid_x, grid_y-(j+1)*Ly)
    return scalar_d
