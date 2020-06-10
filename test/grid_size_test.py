# Test grid size
# %%
from __future__ import division, print_function
import sys
sys.path.append('/home/cecil/Cecil/Projects/convolution_fft/src')
import pyfftw
import multiprocessing
# import copy
import numpy as np
import kernel.kernel_func as kernel_mod
import convolution_fft as conv
import kernel.mobility_numba as mob_numba
import scipy.io as sio
# import os
# os.system('export NUMBA_DISABLE_JIT=1')


#%%
nj = 2
Lx, Ly, Lz = [8, 8, 8]

periodic = np.array([0, 0, 0])
sigma = 1
L = [Lx, Ly]
a = 1.0
eta  = 1
h = 2*a

r = np.pi*a


kernel_handle = kernel_mod.kernel_grid_wrapper(mob_numba.no_wall_mobility_trans_trans_numba, eta, a)



repeat = 4
num = 7
err = np.zeros((num, repeat))

for i in range(num):
    nx = 2**(i+2)
    ny = 2**(i+2)
    for j in range(repeat):
        print('nx = {}, run {}'.format(nx,j))
        source_location = np.zeros((2,nj))
        source_location[0,0] = np.random.rand(1)*(Lx-r)
        source_location[1,0] = np.random.rand(1)*(Ly-r)
        # source_location[2,:] = h
        theta = np.random.rand(1)*np.pi/2
        # theta = 0
        source_location[0,1] = source_location[0,0] + r*np.cos(theta)
        source_location[1,1] = source_location[1,0] + r*np.sin(theta)

        # source_strength = np.array([-np.cos(theta),-np.sin(theta),np.cos(theta),np.sin(theta)])
        source_strength = np.array([np.cos(theta),np.sin(theta),0,0])
        source_strength = source_strength.reshape((2,2))
        # source_strength = np.random.rand(2,2)
        # source_strength = np.zeros((2,1))
        # source_strength[0,0] = 1
        

        x = np.linspace(0, nx-1, nx)*Lx/nx
        y = np.linspace(0, ny-1, ny)*Ly/ny
        x_d = np.concatenate((x, x-Lx), axis=None) # Numba
        y_d = np.concatenate((y, y-Ly), axis=None)
        z_d = np.array([0])

        grid_x, grid_y, grid_z = np.meshgrid(x_d, y_d, z_d, indexing = 'ij')

        kernel = kernel_mod.__kernel_evaluate_3D(Lx, Ly, Lz, grid_x, grid_y, grid_z, kernel_handle, periodic)

        a_fft = pyfftw.empty_aligned((2*nx, 2*ny), dtype='complex128')
        # Save efforts by knowing that a is real
        b_fft = pyfftw.empty_aligned((2*nx, 2*ny), dtype='complex128')
        # Real to complex FFT Over the both axes
        fft_object = pyfftw.FFTW(a_fft, b_fft, axes=(
            0, 1), flags=('FFTW_MEASURE', ))

        kernel_hat  = np.zeros((2*nx, 2*ny, 2, 2), dtype = np.complex128)
        for k in range(2):
            for p in range(2):
                a_fft[:][:] = kernel[:,:,0,k,p]
                kernel_hat[:,:,k,p] = (fft_object()).copy()*Lx/nx*Ly/ny
                
        
        num_modes = [2*nx, 2*ny]
        # fft_object, ifft_object = kernel_mod.create_fftw_plan(num_modes)
        kernel_hat = kernel_hat[:,:,0,0].reshape((2*nx,2*ny,1,1))
        v = conv.vel_convolution_nufft(source_location, source_strength, kernel_hat, num_modes, L, eps = 1e-12)
        # v = np.sqrt(np.sum(np.power(v[1,:],2)))
        # v = v[1,0]*np.cos(theta)+v[1,1]*np.sin(theta)
        v = v[1,0]
        
        
        # v_direct = mob_numba.no_wall_mobility_trans_trans_numba(np.array([0,0,0]),eta,a).dot(np.append(source_strength[1,:],0)) + \
        #      mob_numba.no_wall_mobility_trans_trans_numba(np.array([r,0,0]),eta,a).dot(np.append(source_strength[0,:],0))
        v_direct = mob_numba.no_wall_mobility_trans_trans_numba(np.array([r,0,0]),eta,a).dot(np.concatenate((source_strength[0],np.zeros(2))))
        # v_direct = np.sqrt(np.sum(np.power(v_direct[:2],2)))
        v_direct = v_direct[0]
        # v_direct = np.array([v_direct[0]*np.cos(theta),v_direct[0]*np.sin(theta)])
        print(v,v_direct)
        err[i, j] = np.sqrt(np.sum(np.power(v-v_direct, 2))) / \
            np.sqrt(np.sum(v_direct**2))
        # err[i,j] = abs(v-v_direct)/abs(v_direct)
        # print(np.isclose(v[:,0],v_direct))
        # err[i, j] = np.sqrt(np.sum(np.power(v-v_direct, 2))) / \
        #     np.sqrt(np.sum(v_direct**2))


# %%

save_fn = 'err.mat'

save_array = err

sio.savemat(save_fn,{'err':save_array})

# %%
