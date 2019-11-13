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


def vel_convolution_fft(scalar, method=1, *args, **kwargs):
    if 'kernel' in kwargs:
        kernel = kwargs['kernel']
        kernel_flag = 0  # doubled kernel is provided
        assert isinstance(kernel, np.ndarray)
    elif ('kernel_handle' in kwargs) & ('L' in kwargs) & ('x' in kwargs) & ('periodic' in kwargs):
        kernel_handle = kwargs['kernel_handle']
        L = kwargs['L']
        x = kwargs['x']
        periodic = kwargs['periodic']
        kernel_flag = 1  # given kernel handle
        assert callable(kernel_handle)
    elif ('kernel_hat' in kwargs):
        kernel_hat = kwargs['kernel_hat']
        kernel_flag = 2  # given fft of kernel
        assert isinstance(kernel_hat, np.ndarray)

    Nshape = scalar.shape
    dim = len(Nshape)
    if dim == 2:
        [ny, nx] = Nshape[:]
        if method == 1:
            scalar_d = np.zeros([2*ny, 2*nx])
            scalar_d[:ny, :nx] = scalar
            # fftw objects, either a global object or created at every call
            if ('fft_object' in kwargs) & ('ifft_object' in kwargs):
                fft_object = kwargs['fft_object']
                ifft_object = kwargs['ifft_object']
                a = fft_object.input_array
                b = fft_object.output_array
            else:
                a = pyfftw.empty_aligned((2*ny, 2*nx), dtype='float64')
                # Save efforts by knowing that a is real
                b = pyfftw.empty_aligned((2*ny, nx+1), dtype='complex128')
                # Real to complex FFT Over the both axes
                fft_object = pyfftw.FFTW(a, b, axes=(
                    0, 1), flags=('FFTW_MEASURE', ))
                ifft_object = pyfftw.FFTW(b, a, axes=(
                    0, 1), direction='FFTW_BACKWARD', flags=('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'))
            if kernel_flag == 0:
                a[:][:] = kernel
                kernel_hat = copy.deepcopy(fft_object())
            elif kernel_flag == 1:
                kernel = kernel_evaluate(x, kernel_handle, periodic, L)
                a[:][:] = kernel
                kernel_hat = copy.deepcopy(fft_object())
            elif kernel_flag == 2:
                pass

            a[:][:] = scalar_d
            scalar_d_hat = copy.deepcopy(fft_object())

            b[:][:] = scalar_d_hat * kernel_hat
            v = copy.deepcopy(ifft_object())
            return v[:ny, :nx]

    elif dim == 3:
        raise Exception('Haven\'t implemented convolution for %dD!' % dim)
    else:
        raise Exception('Haven\'t implemented convolution for %dD!' % dim)
    return 0


def kernel_evaluate(x, kernel, periodic, L):
    dim = len(L)
    if dim == 2:
        Lx, Ly = L[:]
        x, y = x[:]
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
    else:
        raise Exception('%iD is not implemented yet!' % dim)


if __name__ == "__main__":
    nx, ny = [4, 4]
    Lx, Ly = [10, 10]
    dx, dy = [Lx/nx, Ly/ny]
    x = (np.linspace(0, nx-1, nx)+0.5)*dx
    y = (np.linspace(0, ny-1, ny)+0.5)*dy
    scalar = np.ones([ny, nx])
    periodic = np.array([1, 0])
    kernel = np.ones([ny*2, nx*2])*(1+2*periodic[0])*(1+2*periodic[1])
    v = vel_convolution_fft(scalar, kernel=kernel)
    print(np.allclose(v, np.ones([ny, nx])*ny *
                      nx*(1+2*periodic[0])*(1+2*periodic[1])))
