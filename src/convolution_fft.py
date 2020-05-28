from __future__ import division, print_function
import numpy as np
import finufftpy
from numba import jit, prange
# import copy
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
# Only double precision is implemented for now, TODO

def real(x): # TODO, this is the function to choose single or double
    if precision == 'single':
        return np.float32(x)
    else:
        return np.float64(x)

# Donev: You should write these routines to be more specific and less general
# So instead of having a general **kwargs you should always require the user to pass in the kernel_hat, and always require the user to pass in fft_object etc.
# It is nice to have flexible code but it reads to code duplication and for scientific applications you want to make the code efficient and force
# the user to call the code in an efficient manner, rather than make it most convenient
# You can also make it simple by always taking in 3d arrays and making 2d mean nz=1 (one cell in z) and 1d mean nz=ny=1.
def vel_convolution_fft(scalar, method=1, *args, **kwargs):
    """ Compute convolution of given 'scalar' with kernel, 
    dimension of 1, 2, 3 is accepted. 
    """
    if ('kernel_hat' in kwargs):
        # User provides with Fourier transform of the kernel
        kernel_hat = kwargs['kernel_hat']
        assert isinstance(kernel_hat, np.ndarray)

    Nshape = scalar.shape
    dim = len(Nshape)
    if dim == 1:
        # One dimension 
        nx = Nshape[0]
        if method == 1:
            scalar_d = np.zeros(2*nx) # Double zero-pad to do method 1
            scalar_d[:nx] = scalar
            # fftw objects, either a global object or created at every call
            if ('fft_object' in kwargs) & ('ifft_object' in kwargs):
                # fftw objects is passed as input, it's used globally
                fft_object = kwargs['fft_object']
                ifft_object = kwargs['ifft_object']
            else:
                # fftw object is created every time this function is called, 
                # suitable for situation where Foutier modes change at every time step
                fft_object, ifft_object = create_fftw_plan([nx])

            a = fft_object.input_array
            b = fft_object.output_array
            

            # Fourier transform of the scalar
            a[:] = scalar_d
            scalar_d_hat = (fft_object()).copy()
            b[:] = scalar_d_hat * kernel_hat
            v = (ifft_object()).copy()

            return v[:nx]
    elif dim == 2:
        # two dimension
        [ny, nx] = Nshape[:]
        if method == 1:
            scalar_d = np.zeros([2*ny, 2*nx]) # Double zero-pad to do method 1
            scalar_d[:ny, :nx] = scalar
            # fftw objects, either a global object or created at every call
            if ('fft_object' in kwargs) & ('ifft_object' in kwargs):
                fft_object = kwargs['fft_object']
                ifft_object = kwargs['ifft_object']                
            else:
                fft_object, ifft_object = create_fftw_plan([nx,ny])

            a = fft_object.input_array
            b = fft_object.output_array

            # Fourier transform of the scalar
            a[:][:] = scalar_d
            scalar_d_hat = (fft_object()).copy()

            b[:][:] = scalar_d_hat * kernel_hat
            v = (ifft_object()).copy()
            return v[:ny, :nx]

    elif dim == 3:
        # three dimension
        [ny, nx, nz] = Nshape[:]
        if method == 1:
            scalar_d = np.zeros([2*ny, 2*nx, 2*nz]) # Double zero-pad to do method 1
            scalar_d[:ny, :nx, :nz] = scalar
            # fftw objects, either a global object or created at every call
            if ('fft_object' in kwargs) & ('ifft_object' in kwargs):
                fft_object = kwargs['fft_object']
                ifft_object = kwargs['ifft_object']
            else:
                fft_object, ifft_object = create_fftw_plan([nx,ny,nz])

            a = fft_object.input_array
            b = fft_object.output_array
            
            # Fourier transform of the scalar
            a[:][:][:] = scalar_d
            scalar_d_hat = (fft_object()).copy()

            b[:][:][:] = scalar_d_hat * kernel_hat
            v = (ifft_object()).copy()
            return v[:ny, :nx, :nz]
    else:
        raise Exception('Haven\'t implemented convolution for %dD!' % dim)
    return 0

# Donev: This works for a scalar kernel but in fact we work with vector forces and tensor kernels
# The easiest way to do this is to treat the kernel as being a matrix (mobility matrix) of size (D,D) and the force (strength) a vector of of size (D)
# Then a scalar kernel is D=1 and plane kernel is D=2, and full 3D is D=3 etc.
# Note that compilers can optimize code better if they now D=2 at compile time (instead of being an unknown value)
# I am not sure what the best way to do that is but perhaps when the Just In Time compiler compiles the code it can see the value of D from the caller? 
# Donev: Why is this limited only to d=2? It seems it also works in 3d equally well (it is an alternative method to periodizing).
# I am not sure if Method 1 will work for a kernel that decays like 1/r instead of 1/r^3 (probably not) but in principle I think it will work, we can discuss on zoom
def vel_convolution_nufft(source_location, source_strenth, num_modes, L,  eps=1e-4, method=1, shift_flag = 1, *args, **kwargs):
    """ Compute convolution of given 'source_location' with kernel,
    only dimension of two is implemented yet. TODO
    'source_location', the location of each point, not uniform 
    'soruce_strenth', strenth of each point  
    'num_modes', number of fourier modes in each direction
    'L', length of the box
    'eps=1e-4', error of the nufft
    'method=1', only method 1 is implemented TODO
    'shift_flag = 1'. whether to shift particle to the periodic box or not. 
    """
    
    # fourier transform of the kernel is better to be evaluated only once.
    if('kernel_hat' in kwargs):
        kernel_hat = kwargs['kernel_hat']
        assert isinstance(kernel_hat, np.ndarray)
    else:
        raise Exception(
            'kernel_hat, fourier transform of kernel should be given')

    Nshape = source_location.shape
    dim = Nshape[0]
    assert dim == 2, "Dim should be 2!"
    nx, ny = num_modes[:]
    if method == 1:
        xj, yj = source_location
        nj = len(xj)
        if shift_flag:
            # Dicide the box
            xmin = xj.min()
            ymin = yj.min()
            Lx, Ly = L
            xj = (((xj-xmin)%Lx)/Lx-0.5)*np.pi
            yj = (((yj-ymin)%Ly)/Ly-0.5)*np.pi
        else:
            # Dicide the box
            xmin = xj.min()
            xmax = xj.max()
            ymin = yj.min()
            ymax = yj.max()
            Lx, Ly = L
            assert ((xmax-xmin) <= Lx) & ((ymax-ymin) <= Ly), "All particles should be limieted to given [Lx,Ly] box!"
            # double zero-padding the delta function and scale it into [-pi,pi)
            xj = (xj-(xmax+xmin)/2)/Lx*np.pi
            yj = (yj-(ymax+ymin)/2)/Ly*np.pi

        fk = np.zeros(num_modes, dtype=np.complex128, order='F')     
        # TODO: Reuse finufft object.
        ret = finufftpy.nufft2d1(xj, yj, source_strenth, -1, eps, nx, ny, fk, modeord=1)
        assert ret == 0, "NUFFT not successful!"
        v_hat = np.zeros((nx, ny, 2), order='F', dtype=np.complex128)
        v_hat[:, :, 0] = fk * kernel_hat[0]
        v_hat[:, :, 1] = fk * kernel_hat[1]
        v = np.zeros((nj, 2), order='F', dtype=np.complex128) # Donev: The return result should be double64 not complex128
        ret = finufftpy.nufft2d2many(xj, yj, v, 1, eps, v_hat, modeord=1)
        assert ret == 0, "NUFFT not successful!"
        v = np.real(v)/(2*Lx*2*Ly)  # Real? # Donev: Make sure this is a real number not complex. Return only the real part
        return v

# Donev: You should learn how to use NUMBA and write this in numba. There is a CUDA version in ConvolutionAdvection already as well
@jit(nopython = True)
def vel_direct_convolution(scalar, source_strenth, kernel_handle, L, periodic):
    """ Direct convolution, which is for comparison with vel_convolution_nufft
    """
    Np = len(source_strenth)
    dim = len(L)
    v = np.zeros(Np)
    if dim == 2:
        x = scalar[0]
        y = scalar[1]
        # x, y = scalar[:] # iterating over multidimensional array is a bug in numba
        for i in range(Np):
            for j in range(Np):
                kernel = 0
                for kernel_index_x in range(-periodic[0],periodic[0]+1):
                    for kernel_index_y in range(-periodic[1],periodic[1]):
                        xd = x[i]-x[j]-L[0]*kernel_index_x
                        yd = y[i]-y[j]-L[1]*kernel_index_y
                        kernel += kernel_handle(xd,yd)
                # kernel_index_x, kernel_index_y = np.meshgrid(np.linspace(-periodic[0],periodic[0],2*periodic[0]+1),np.linspace(-periodic[1],periodic[1],2*periodic[1]+1), indexing = 'ij')
                # xd = x[i]-x[j]-L[0]*kernel_index_x
                # yd = y[i]-y[j]-L[1]*kernel_index_y
                # kernel = kernel_handle(xd,yd).sum()
                v[i] += kernel * source_strenth[j]
    else:
        raise Exception("Not implemented yet!")
    return v
        
# Donev: Numba will accelerate this greatly. Even though we only need to do this once, it is nice to make it more efficient and it should be easy
# Remember that we need to sum over many periodic images sometimes so this should be somewhat efficient not plain python loops    
# @jit(nopython = True) # A function wrapper is needed here since some functions in numpy are not well supported
# Donev: I think this can be handled best by always making the arrays be 3d
def kernel_evaluate(grids, kernel, periodic, L, indexing = 'xy'):
    """ Evaluate kernel on given grid
    1, 2, 3D are accepted.
    """
    dim = len(L)
    if dim == 1:
        Lx = L[0]
        x = grids[0]
        return __kernel_evaluate_1D(Lx, np.concatenate((x, x-Lx), axis=None), kernel, periodic)
        # periodic_flag = ~(periodic == 0)
        # scalar_d = kernel(x_d)
        # if periodic_flag[0]:
        #     for i in range(periodic[0]):
        #         scalar_d = scalar_d + kernel(x_d+(i+1)*Lx)
        #         scalar_d = scalar_d + kernel(x_d-(i+1)*Lx)
        # return scalar_d
    if dim == 2:
        Lx, Ly = L[:]
        x, y = grids[:]
        x_d = np.concatenate((x, x-Lx), axis=None)  # Numba has problem with (x,x-Lx)
        y_d = np.concatenate((y, y-Ly), axis=None)
        grid_x, grid_y = np.meshgrid(x_d, y_d, indexing = indexing)
        return __kernel_evaluate_2D(Lx, Ly, grid_x, grid_y, kernel, periodic)
        # periodic_flag = ~(periodic == 0)
        # scalar_d = kernel(grid_x, grid_y)
        # if periodic_flag[0]:
        #     for i in range(periodic[0]):
        #         if periodic_flag[1]:
        #             for j in range(periodic[1]):
        #                 scalar_d = scalar_d + \
        #                     kernel(grid_x+(i+1)*Lx, grid_y+(j+1)*Ly)
        #                 scalar_d = scalar_d + \
        #                     kernel(grid_x+(i+1)*Lx, grid_y-(j+1)*Ly)
        #                 scalar_d = scalar_d + \
        #                     kernel(grid_x-(i+1)*Lx, grid_y+(j+1)*Ly)
        #                 scalar_d = scalar_d + \
        #                     kernel(grid_x-(i+1)*Lx, grid_y-(j+1)*Ly)
        #         else:
        #             scalar_d = scalar_d + kernel(grid_x+(i+1)*Lx, grid_y)
        #             scalar_d = scalar_d + kernel(grid_x-(i+1)*Lx, grid_y)
        # else:
        #     if periodic_flag[1]:
        #         for j in range(periodic[1]):
        #             scalar_d = scalar_d + kernel(grid_x, grid_y+(j+1)*Ly)
        #             scalar_d = scalar_d + kernel(grid_x, grid_y-(j+1)*Ly)
        # return scalar_d
    elif dim == 3:
        Lx, Ly, Lz = L[:]
        x, y, z = grids[:]
        x_d = np.concatenate((x, x-Lx), axis=None) # Numba
        y_d = np.concatenate((y, y-Ly), axis=None)
        z_d = np.concatenate((z, z-Lz), axis=None)
        grid_x, grid_y, grid_z = np.meshgrid(x_d, y_d, z_d, indexing = indexing)
        return __kernel_evaluate_3D(Lx, Ly, Lz, grid_x, grid_y, grid_z, kernel, periodic)
        # periodic_flag = ~(periodic == 0)
        # scalar_d = kernel(grid_x, grid_y, grid_z)
        # if periodic_flag[0]:
        #     for i in range(periodic[0]):
        #         if periodic_flag[1]:
        #             for j in range(periodic[1]):
        #                 if periodic_flag[2]:
        #                     for k in range(periodic[2]):
        #                         scalar_d += kernel(grid_x+(i+1)*Lx, grid_y+(j+1)*Ly, grid_z+(k+1)*Lz) + kernel(
        #                             grid_x+(i+1)*Lx, grid_y+(j+1)*Ly, grid_z-(k+1)*Lz)
        #                         scalar_d += kernel(grid_x+(i+1)*Lx, grid_y-(j+1)*Ly, grid_z+(k+1)*Lz) + kernel(
        #                             grid_x+(i+1)*Lx, grid_y-(j+1)*Ly, grid_z-(k+1)*Lz)
        #                         scalar_d += kernel(grid_x-(i+1)*Lx, grid_y+(j+1)*Ly, grid_z+(k+1)*Lz) + kernel(
        #                             grid_x-(i+1)*Lx, grid_y+(j+1)*Ly, grid_z-(k+1)*Lz)
        #                         scalar_d += kernel(grid_x-(i+1)*Lx, grid_y-(j+1)*Ly, grid_z+(k+1)*Lz) + kernel(
        #                             grid_x-(i+1)*Lx, grid_y-(j+1)*Ly, grid_z-(k+1)*Lz)
        #                 else:
        #                     scalar_d += kernel(grid_x+(i+1)
        #                                        * Lx, grid_y+(j+1)*Ly, grid_z)
        #                     scalar_d += kernel(grid_x+(i+1)
        #                                        * Lx, grid_y-(j+1)*Ly, grid_z)
        #                     scalar_d += kernel(grid_x-(i+1)
        #                                        * Lx, grid_y+(j+1)*Ly, grid_z)
        #                     scalar_d += kernel(grid_x-(i+1)
        #                                        * Lx, grid_y-(j+1)*Ly, grid_z)
        #         else:
        #             if periodic_flag[2]:
        #                 for k in range(periodic[2]):
        #                     scalar_d += kernel(grid_x+(i+1)*Lx, grid_y, grid_z+(k+1)*Lz) + kernel(
        #                         grid_x+(i+1)*Lx, grid_y, grid_z-(k+1)*Lz)
        #                     scalar_d += kernel(grid_x-(i+1)*Lx, grid_y, grid_z+(k+1)*Lz) + kernel(
        #                         grid_x-(i+1)*Lx, grid_y, grid_z-(k+1)*Lz)
        #             else:
        #                 scalar_d += kernel(grid_x+(i+1)*Lx, grid_y, grid_z)
        #                 scalar_d += kernel(grid_x-(i+1)*Lx, grid_y, grid_z)
        # else:
        #     if periodic_flag[1]:
        #         for j in range(periodic[1]):
        #             if periodic_flag[2]:
        #                 for k in range(periodic[2]):
        #                     scalar_d += kernel(grid_x, grid_y+(j+1)*Ly, grid_z+(k+1)*Lz) + kernel(
        #                         grid_x, grid_y+(j+1)*Ly, grid_z-(k+1)*Lz)
        #                     scalar_d += kernel(grid_x, grid_y-(j+1)*Ly, grid_z+(k+1)*Lz) + kernel(
        #                         grid_x, grid_y-(j+1)*Ly, grid_z-(k+1)*Lz)
        #             else:
        #                 scalar_d += kernel(grid_x, grid_y+(j+1)*Ly, grid_z) + \
        #                     kernel(grid_x, grid_y-(j+1)*Ly, grid_z)
        #     else:
        #         if periodic_flag[2]:
        #             for k in range(periodic[2]):
        #                 scalar_d += kernel(grid_x, grid_y, grid_z+(k+1)*Lz) + \
        #                     kernel(grid_x, grid_y, grid_z-(k+1)*Lz)
        # return scalar_d
    else:
        raise Exception('Only 1,2,3 dimension(s) are supported!')


# Numba requires same shape of Numba array of function arguments. So We need to use seperate jitted function for different dimensions.
# Donev: I am a little worried about using lambda functions here. Will they be inlined? What is the overhead of using a lambda function?
# You need to do testing to check this
# Instead, the right way to do this in python is to make a class Kernel that has an object evaluate. But this may not work with numba...
@jit(nopython = True)
def __kernel_evaluate_1D(Lx, x_d, kernel, periodic):
    periodic_flag = ~(periodic == 0)
    scalar_d = kernel(x_d)
    if periodic_flag[0]:
        for i in range(periodic[0]):
            scalar_d = scalar_d + kernel(x_d+(i+1)*Lx) # Donev: Will numba inline kernel ?
            scalar_d = scalar_d + kernel(x_d-(i+1)*Lx)
    return scalar_d

@jit(nopython = True)
def __kernel_evaluate_2D(Lx, Ly, grid_x, grid_y, kernel, periodic):
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

@jit(nopython = True)
def __kernel_evaluate_3D(Lx, Ly, Lz, grid_x, grid_y, grid_z, kernel, periodic):
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

def kernel_fft_evaluate(method = 1, indexing = 'xy', *arg, **kwargs):
    """ Compute Fourier transform of kenrel. user can provide with kernel evaluated at grids,
    or kernel function handle and grid. It's recommended that user provides with FFTW objects
    so that it could be used among iterations.
    """
    if 'kernel' in kwargs:
        # User provides with evaluated kernel 
        kernel = kwargs['kernel']
        kernel_flag = 0  # doubled kernel is provided
        assert isinstance(kernel, np.ndarray)
        Nshape = [i//2 for i in kernel.shape]
    elif ('kernel_handle' in kwargs) & ('L' in kwargs) & ('x' in kwargs) & ('periodic' in kwargs):
        # User provides with hernel function handle.
        kernel_handle = kwargs['kernel_handle']
        L = kwargs['L']
        x = kwargs['x']
        periodic = kwargs['periodic']
        kernel_flag = 1  # given kernel handle
        assert callable(kernel_handle)
        Nshape = [len(i) for i in x]
    else:
        raise Exception('Please provide kernel or kernel function')
    dim = len(Nshape)
    if dim == 1:
        # One dimension 
        nx = Nshape[0]
        if method == 1:
            # fftw objects, either a global object or created at every call
            if ('fft_object' in kwargs):
                # fftw objects is passed as input, it's used globally
                fft_object = kwargs['fft_object']
            else:
                # fftw object is created every time this function is called, 
                # suitable for situation where Foutier modes change at every time step
                fft_object = create_fftw_plan([nx],ifft_flag=False)

            a = fft_object.input_array
            b = fft_object.output_array
            
            # To get the Fourier transform of the kernel
            if kernel_flag == 0:
                a[:] = kernel
                kernel_hat = (fft_object()).copy()
            elif kernel_flag == 1:
                # kernel needs to  be evaluated
                kernel = kernel_evaluate(x, kernel_handle, periodic, L)
                a[:] = kernel
                kernel_hat = (fft_object()).copy()
    elif dim == 2:
        # two dimension
        [ny, nx] = Nshape[:] # TODO: indexing of arrays
        if method == 1:
            # fftw objects, either a global object or created at every call
            if ('fft_object' in kwargs):
                fft_object = kwargs['fft_object']
            else:
                fft_object = create_fftw_plan([nx,ny],ifft_flag=False)

            a = fft_object.input_array
            b = fft_object.output_array
            
            # To get the Fourier transform of the kernel
            if kernel_flag == 0:
                a[:][:] = kernel
                kernel_hat = (fft_object()).copy()
            elif kernel_flag == 1:
                kernel = kernel_evaluate(x, kernel_handle, periodic, L)
                a[:][:] = kernel
                kernel_hat = (fft_object()).copy()
    elif dim == 3:
        # three dimension
        [ny, nx, nz] = Nshape[:]
        if method == 1:
            # fftw objects, either a global object or created at every call
            if ('fft_object' in kwargs):
                fft_object = kwargs['fft_object']
            else:
                fft_object = create_fftw_plan([nx,ny,nz],ifft_flag=False)

            a = fft_object.input_array
            b = fft_object.output_array
            # To get the Fourier transform of the kernel
            if kernel_flag == 0:
                a[:][:][:] = kernel
                kernel_hat = (fft_object()).copy()
            elif kernel_flag == 1:
                kernel = kernel_evaluate(x, kernel_handle, periodic, L)
                a[:][:][:] = kernel
                kernel_hat = (fft_object()).copy()
    else:
        raise Exception('Haven\'t implemented convolution for %dD!' % dim)
    return kernel_hat

def create_fftw_plan(n, ifft_flag = True):
    """Create Pyfftw plan
    n, array of [nx[,ny[,nz]]], 1,2,3D are accepted
    ifft_flag = True, whether to create ifft object at the same time. Notice that, to save memory, this use
    input array a of fft object as output array, it uses output array b of fft objects as input array.
    Users should be cautious about that this sharing array issue.
    """
    dim = len(n)
    if dim == 1:
        nx = n[0]
        a = pyfftw.empty_aligned((2*nx), dtype='float64')
        # Save efforts by knowing that a is real
        b = pyfftw.empty_aligned((nx+1), dtype='complex128')
        # Real to complex FFT Over the both axes
        fft_object = pyfftw.FFTW(a, b, axes=(
            -1,), flags=('FFTW_MEASURE', ))
        if ifft_flag:
            ifft_object = pyfftw.FFTW(b, a, axes=(
                -1,), direction='FFTW_BACKWARD', flags=('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'))
    elif dim == 2:
        nx, ny = n[:]
        a = pyfftw.empty_aligned((2*ny, 2*nx), dtype='float64')
        # Save efforts by knowing that a is real
        b = pyfftw.empty_aligned((2*ny, nx+1), dtype='complex128')
        # Real to complex FFT Over the both axes
        fft_object = pyfftw.FFTW(a, b, axes=(
            0, 1), flags=('FFTW_MEASURE', ))
        if ifft_flag:
            ifft_object = pyfftw.FFTW(b, a, axes=(
                0, 1), direction='FFTW_BACKWARD', flags=('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'))
    elif dim == 3:
        nx, ny, nz = n[:]
        a = pyfftw.empty_aligned((2*ny, 2*nx, 2*nz), dtype='float64')
        # Save efforts by knowing that a is real
        b = pyfftw.empty_aligned(
            (2*ny, 2*nx, nz+1), dtype='complex128')
        # Real to complex FFT Over the both axes
        fft_object = pyfftw.FFTW(a, b, axes=(
            0, 1, 2), flags=('FFTW_MEASURE', ))
        if ifft_flag:
            ifft_object = pyfftw.FFTW(b, a, axes=(
                0, 1, 2), direction='FFTW_BACKWARD', flags=('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'))
    else:
        raise Exception('It work only in 1,2,3D!')
    if ifft_flag:
        return fft_object, ifft_object
    else:
        return fft_object

    