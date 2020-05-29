from __future__ import division, print_function
import numpy as np
import finufftpy
from numba import jit, prange
# import copy
import multiprocessing
from kernel.kernel import create_fftw_plan

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


def vel_convolution_fft(source, kernel_hat, method=1, *args, **kwargs):
    """ Compute convolution of given 'source' with kernel, 
    dimension of 1, 2, 3 is accepted. 'source' ndarrays of nx*ny*nz, 
    each component of 'source' is ndarrays in corresponding dimension
    """
    # User provides with Fourier transform of the kernel
    assert isinstance(kernel_hat, np.ndarray)

    Nshape = source.shape
    dim = len(Nshape)
    if dim == 1:
        # One dimension 
        nx = Nshape[0]
        if method == 1:
            source_d = np.zeros(2*nx) # Double zero-pad to do method 1
            source_d[:nx] = source
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
            

            # Fourier transform of the source
            a[:] = source_d
            source_d_hat = (fft_object()).copy()
            b[:] = source_d_hat * kernel_hat
            v = (ifft_object()).copy()

            return v[:nx]
    elif dim == 2:
        # two dimension
        [ny, nx] = Nshape[:]
        if method == 1:
            source_d = np.zeros([2*ny, 2*nx]) # Double zero-pad to do method 1
            source_d[:ny, :nx] = source
            # fftw objects, either a global object or created at every call
            if ('fft_object' in kwargs) & ('ifft_object' in kwargs):
                fft_object = kwargs['fft_object']
                ifft_object = kwargs['ifft_object']                
            else:
                fft_object, ifft_object = create_fftw_plan([nx,ny])

            a = fft_object.input_array
            b = fft_object.output_array

            # Fourier transform of the source
            a[:][:] = source_d
            source_d_hat = (fft_object()).copy()

            b[:][:] = source_d_hat * kernel_hat
            v = (ifft_object()).copy()
            return v[:ny, :nx]

    elif dim == 3:
        # three dimension
        [ny, nx, nz] = Nshape[:]
        if method == 1:
            source_d = np.zeros([2*ny, 2*nx, 2*nz]) # Double zero-pad to do method 1
            source_d[:ny, :nx, :nz] = source
            # fftw objects, either a global object or created at every call
            if ('fft_object' in kwargs) & ('ifft_object' in kwargs):
                fft_object = kwargs['fft_object']
                ifft_object = kwargs['ifft_object']
            else:
                fft_object, ifft_object = create_fftw_plan([nx,ny,nz])

            a = fft_object.input_array
            b = fft_object.output_array
            
            # Fourier transform of the source
            a[:][:][:] = source_d
            source_d_hat = (fft_object()).copy()

            b[:][:][:] = source_d_hat * kernel_hat
            v = (ifft_object()).copy()
            return v[:ny, :nx, :nz]
    else:
        raise Exception('Haven\'t implemented convolution for %dD!' % dim)
    return 0


def vel_convolution_fft_vector(source, kernel_hat, method=1, *args, **kwargs):
    """ Compute convolution of given 'source' with kernel, 
    dimension of 1, 2, 3 is accepted. 'source' ndarrays of ny*nx*nz*D, 
    At each point, source is a D*D mobility matrix
    kernel_hat is the fourier transform of the kernel 2ny*2nx*(nz+1)*D*D
    """
    # This wrapper function serves for vector source convolved with vector kernel, I want to keep the generality of 
    # vel_convolution_fft function, which only deal with nx*ny*nz tensor such that users can still call that function to compute scalar kernel
    assert len(source.shape)+1 == len(kernel_hat.shape)
    assert source.shape[-1] == kernel_hat.shape[-1] == kernel_hat.shape[-2]

    D = source.shape[-1]
    Nshape = source.shape[:-1]
    dim = len(Nshape)
    v = np.zeros_like(source)

    if dim == 1:
        # One dimension
        nx = Nshape[0]
        # fftw objects, either a global object or created at every call
        if ('fft_object' in kwargs) & ('ifft_object' in kwargs):
            fft_object = kwargs['fft_object']
            ifft_object = kwargs['ifft_object']
        else:
            fft_object, ifft_object = create_fftw_plan([nx])
        for i in range(D):
            for j in range(D):
                v[:,i] += vel_convolution_fft(source[:,j], kernel_hat[:,i,j], method = method, fft_object = fft_object, ifft_object = ifft_object)

    elif dim == 2:
        # two dimension
        [ny, nx] = Nshape[:]
        # fftw objects, either a global object or created at every call
        if ('fft_object' in kwargs) & ('ifft_object' in kwargs):
            fft_object = kwargs['fft_object']
            ifft_object = kwargs['ifft_object']
        else:
            fft_object, ifft_object = create_fftw_plan([nx,ny])
        for i in range(D):
            for j in range(D):
                v[:,:,i] += vel_convolution_fft(source[:,:,j], kernel_hat[:,:,i,j], method = method, fft_object = fft_object, ifft_object = ifft_object)

    elif dim == 3:
        # three dimension
        [ny, nx, nz] = Nshape[:]
        # fftw objects, either a global object or created at every call
        if ('fft_object' in kwargs) & ('ifft_object' in kwargs):
            fft_object = kwargs['fft_object']
            ifft_object = kwargs['ifft_object']
        else:
            fft_object, ifft_object = create_fftw_plan([nx,ny,nz])
        for i in range(D):
            for j in range(D):
                v[:,:,:,i] += vel_convolution_fft(source[:,:,:,j], kernel_hat[:,:,:,i,j], method = method, fft_object = fft_object, ifft_object = ifft_object)
    else:
        raise Exception('Haven\'t implemented convolution for %dD!' % dim)
    return v

# Donev: This works for a scalar kernel but in fact we work with vector forces and tensor kernels
# The easiest way to do this is to treat the kernel as being a matrix (mobility matrix) of size (D,D) and the force (strength) a vector of of size (D)
# Then a scalar kernel is D=1 and plane kernel is D=2, and full 3D is D=3 etc.
# Note that compilers can optimize code better if they now D=2 at compile time (instead of being an unknown value)
# I am not sure what the best way to do that is but perhaps when the Just In Time compiler compiles the code it can see the value of D from the caller? 
# Donev: Why is this limited only to d=2? It seems it also works in 3d equally well (it is an alternative method to periodizing).
# I am not sure if Method 1 will work for a kernel that decays like 1/r instead of 1/r^3 (probably not) but in principle I think it will work, we can discuss on zoom
def vel_convolution_nufft(source_location, source_strenth, kernel_hat, num_modes, L,  eps=1e-4, method=1, shift_flag = 1, *args, **kwargs):
    """ Compute convolution of given 'source_location' with kernel,
    only dimension of two is implemented yet. TODO
    'source_location', the location of each point, not uniform, dim*nj
    'source_strenth', strenth of each point, nj*D
    'num_modes', number of fourier modes in each direction, nx*ny
    'L', length of the box
    'eps=1e-4', error of the nufft
    'method=1', only method 1 is implemented TODO
    'shift_flag = 1'. whether to shift particle to the periodic box or not. 
    """
    
    # fourier transform of the kernel is better to be evaluated only once.
    assert isinstance(kernel_hat, np.ndarray)
    assert kernel_hat.shape[-1]==kernel_hat.shape[-2]==source_strenth.shape[-1], "Matrix shape error!"
    D = source_strenth.shape[-1]
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

        fk = np.zeros((nx,ny,D), dtype=np.complex128, order='F')     
        # TODO: Reuse finufft object.
        ret = finufftpy.nufft2d1many(xj, yj, source_strenth, -1, eps, nx, ny, fk, modeord=1)
        assert ret == 0, "NUFFT not successful!"
        v_hat = np.zeros((nx, ny, D), order='F', dtype=np.complex128)
        for i in range(D):
            for j in range(D):
                v_hat[:,:,i] += fk[:,:,j] * kernel_hat[:,:,i,j]
        v = np.zeros((nj, D), order='F', dtype=np.complex128) 
        # Donev: The return result should be double64 not complex128 
        # Zhe: This is not implemented by finufft(At least for python interface), it output complex128 and I take the real part of it.
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


