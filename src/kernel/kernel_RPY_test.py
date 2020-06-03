# Test consistence between mobility_numba(by Zhe) with mobility_times_force_numba(by Floren)
import numpy as np
import mobility_numba as mob_1
import mobility_times_force_numba as mob_2
a = 1
eta = 1
h = 1
number = 100
L = 100

def my_get_mobility(func, r_vectors, *arg):
    M = np.zeros((3,3))
    f = np.array([0,0,0,1,0,0])
    u =  func(r_vectors, f, *arg)
    M[:,0] = u[0:3]
    f = np.array([0,0,0,0,1,0])
    u = func(r_vectors, f, *arg)
    M[:,1] = u[0:3]
    f = np.array([0,0,0,0,0,1])
    u =  func(r_vectors, f, *arg)
    M[:,2] = u[0:3]
    return M
    
def test(func1, func2, func_with_wall_1, func_with_wall_2, number, a, eta, h):
    rtol = 1e-8
    atol = 1e-12
    r_vectors = np.random.rand(number,3)*L
    c_flag = 1
    for i in range(number):
        r = np.zeros(6)
        r[0:3] = r_vectors[i,:]
        r[2] += h
        r[5] = h
        M_1 = func1(r_vectors[i,:],eta,a)
        M_2 = my_get_mobility(func2, r, eta, a, np.array([0,0,0]))
        if not np.allclose(M_1, M_2, rtol=rtol, atol=atol):
            print('Not consistent!')
            c_flag = 0
            break

        M_1 = func_with_wall_1(r_vectors[i,:],h,eta,a)
        M_2 = my_get_mobility(func_with_wall_2, r, eta, a, np.array([0,0,0]))
        if not np.allclose(M_1, M_2, rtol=rtol, atol=atol):
            print('Not consistent!')
            c_flag = 0
            break
    if c_flag:
        print('Success! It\'s consistent!')
        

print('Testing Trans-Trans kernel!')
func1 = mob_1.no_wall_mobility_trans_trans_numba
func2 = mob_2.no_wall_mobility_trans_times_force_numba
func_with_wall_1 = mob_1.single_wall_mobility_trans_trans_numba
func_with_wall_2 = mob_2.single_wall_mobility_trans_times_force_numba
test(func1, func2, func_with_wall_1, func_with_wall_2, number, a, eta, h)
print('\n\n\n\n\n\n')


print('Testing Trans-Rot kernel!')
func1 = mob_1.no_wall_mobility_trans_rot_numba
func2 = mob_2.no_wall_mobility_trans_times_torque_numba
func_with_wall_1 = mob_1.single_wall_mobility_trans_rot_numba
func_with_wall_2 = mob_2.single_wall_mobility_trans_times_torque_numba
test(func1, func2, func_with_wall_1, func_with_wall_2, number, a, eta, h)
print('\n\n\n\n\n\n')


print('Testing Rot-Rot kernel!')
func1 = mob_1.no_wall_mobility_rot_rot_numba
func2 = mob_2.no_wall_mobility_rot_times_torque_numba
func_with_wall_1 = mob_1.single_wall_mobility_rot_rot_numba
func_with_wall_2 = mob_2.single_wall_mobility_rot_times_torque_numba
test(func1, func2, func_with_wall_1, func_with_wall_2, number, a, eta, h)
print('\n\n\n\n\n\n')