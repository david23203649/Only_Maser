
import numpy as np
import matplotlib.pyplot as plt
from qutip import *


import sympy as sym



from datetime import datetime
import scipy.sparse as sp


from Pulse_and_time_dependent import Pulses
import h5py
import os
PI = np.pi
two_pi= 2 * np.pi
GHz = pow(10,3)
MHz = pow(10,0)
KHz = pow(10,-3)
def find_linewidth(real_eigenenergies, dim):
    for i in range(dim):
        if real_eigenenergies[i] < -1e-10:

            return real_eigenenergies[i]
        else:
            i = i + 1


# number of zero eigenvalue
def number_of_zero_eigenvalue(real_eigenenergies_list):
    number_of_zero_eigenvalue = 0
    for i in range(len(real_eigenenergies_list)):
        if abs(real_eigenenergies_list[-i-1]) < 1e-10:
            number_of_zero_eigenvalue = number_of_zero_eigenvalue + 1
            i = i + 1
    return number_of_zero_eigenvalue


def find_rho_ss2(eigenstate_list, num_to_check):
    temp = []
    temp1 = []
    temp2 = []
    temp3 = []
    for j in range(num_to_check):

        # check trace
        L_steadystate = []
        for i in range(dim ** 2):
            L_steadystate.append(np.real(eigenstate_list[1][j][i]))

        L_steadystate_reshape = np.reshape(L_steadystate, (dim, dim))
        L_steadystate_reshape_qobj = Qobj(L_steadystate_reshape, dims=rho0.dims)
        L_steadystate_reshape_qobj_n = L_steadystate_reshape_qobj.unit()
        temp.append(abs(np.real(expect(a_cav.dag() * a_cav, L_steadystate_reshape_qobj))))
        temp2.append(L_steadystate_reshape_qobj.tr())
        temp3.append(abs(np.real(expect(a_cav.dag() * a_cav, L_steadystate_reshape_qobj_n))))

    max_value = max(temp3)
    max_index_intemp3 = temp3.index(max_value)

    # print(j,max_value)
    return temp2[max_index_intemp3], max_value, temp[max_index_intemp3]


# find steady state from time evolution array,
# if the array is not steady enough (difference btw nearby data is lower than threshold)
# will return none.


def find_steady_value(arr, threshold=0.001, n=10):
    if len(arr) < 2 or len(arr) < n:
        return None  # Not enough data for comparison

    recent_data = arr[-n:]  # Consider only the last 'n' elements of the array
    steady_val = recent_data[0]  # Initialize steady value with the first element

    for i in range(1, len(recent_data)):
        if abs(recent_data[i] - steady_val) <= threshold:
            return arr[-1]  # Return the steady value within the threshold
        else:
            steady_val = recent_data[i]  # Update the steady value if the difference is larger

    return None  # Return None if the threshold was not met within the recent data


# def save_with_custom_info(array, prefix, args, custom_info, directory):
#     amp = args['amp']
#     alpha = args['alpha']
#     g1 = args['g1']
#
#     g3 = args['g3']
#     g4 = args['g4']
#     g5 = args['g5']
#     gamma_cav = args['gamma_cav']
#     gamma_down = args['gamma_down']
#     gamma_SNAIL = args['gamma_SNAIL']
#
#     current_date = datetime.now().strftime('%Y-%m-%d')  # Get current date in YYYY-MM-DD format
#     file_directory = directory
#     filename = directory + f"{prefix}_{current_date}" + "_amp = " + f"{amp}" + "_g1 = " + f"{np.round(g1 / two_pi, 2)}" + "_g3 = " + f"{np.round(g3 / two_pi, 2)}" + "_g4 = " + f"{np.round(g4 / two_pi, 2)}" + "_g5 = " + f"{np.round(g5 / two_pi, 2)}" + "_r_q = " + f"{np.round(gamma_down / two_pi, 3)}" + "_r_s = " + f"{np.round(gamma_SNAIL / two_pi, 2)}" + "_rc = " + f"{np.round(gamma_cav / two_pi, 2)}_{custom_info}" + ".csv"
#     np.savetxt(filename, array, delimiter=',')

## loaded_data = np.loadtxt(filename, delimiter=',')
def save_with_custom_info2(array, prefix, args, custom_info, directory):
    amp = args['amp']
    alpha = args['alpha']
    g1 = args['g1']

    g3 = args['g3']
    g4 = args['g4']
    g5 = args['g5']
    gamma_cav = args['gamma_cav']
    gamma_down = args['gamma_down']
    gamma_SNAIL = args['gamma_SNAIL']

    current_date = datetime.now().strftime('%Y-%m-%d')  # Get current date in YYYY-MM-DD format
    filename = f"{directory}{prefix}_{current_date}_amp_{amp}_g1_{np.round(g1 / two_pi, 2)}_g3_{np.round(g3 / two_pi, 2)}_g4_{np.round(g4 / two_pi, 2)}_g5_{np.round(g5 / two_pi, 2)}_r_q_{np.round(gamma_down / two_pi, 3)}_r_s_{np.round(gamma_SNAIL / two_pi, 2)}_rc_{np.round(gamma_cav / two_pi, 2)}_{custom_info}.npy"

    # Save the array
    np.save(filename, array)

def save_3d_array(array_3d, prefix, args, custom_info, directory):
    """Save a 3D NumPy array to a file."""
    amp = args['amp']
    alpha = args['alpha']
    g1 = args['g1']

    g3 = args['g3']
    g4 = args['g4']
    g5 = args['g5']
    gamma_cav = args['gamma_cav']
    gamma_down = args['gamma_down']
    gamma_SNAIL = args['gamma_SNAIL']

    file_directory = directory
    current_date = datetime.now().strftime('%Y-%m-%d')
    filename = directory + f"{prefix}_{current_date}" + "_amp_" + f"{amp}" + "_g1_" + f"{np.round(g1 / two_pi, 2)}" + "_g3_" + f"{np.round(g3 / two_pi, 2)}" + "_g4_" + f"{np.round(g4 / two_pi, 2)}" + "_g5_" + f"{np.round(g5 / two_pi, 2)}" + "_r_q_" + f"{np.round(gamma_down / two_pi, 3)}" + "_r_s_" + f"{np.round(gamma_SNAIL / two_pi, 2)}" + "_rc_" + f"{np.round(gamma_cav / two_pi, 2)}_{custom_info}"

    np.save(filename, array_3d)

def load_custom_data(file_path):
    try:
        loaded_data = np.load(file_path)
        return loaded_data
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None


def read_3d_array(directory, filename ):
    """Read a 3D NumPy array from a file."""
    array_3d = np.load(directory + filename)
    return array_3d





def custom_formatter(value, pos):
    return f'{value:.2f}'  # Adjust the format as needed

def save_data_and_args(filename, data_dict, **kwargs):
    with h5py.File(filename, 'w') as f:
        # Save data
        for name, data in data_dict.items():
            f.create_dataset(name, data=data)

        # Save kwargs
        for key, value in kwargs.items():
            f.attrs[key] = value


# Example usage

def read_data_and_args(filename):
        data_dict = {}
        kwargs = {}
        with h5py.File(filename, 'r') as f:
                # Read data
                for name, dataset in f.items():
                        data_dict[name] = dataset[:]

                # Read kwargs
                for key, value in f.attrs.items():
                        kwargs[key] = value

        return data_dict, kwargs



def savedata(filename, overwrite=0, **kwargs):

    if not overwrite:
        import os.path
        if os.path.exists(filename + f""):
            raise OSError("the file already exists!")
    hf = h5py.File(filename, 'w')
    for key, value in kwargs.items():
        hf.create_dataset(key, data=value)
    hf.close()

def loadData(filePath, fileName):
    f = h5py.File(filePath + fileName, 'r')
    # nbar = f['nbar'][()]
    # qbar = f['qbar'][()]
    # sbar = f['sbar'][()]
    # timeevo_m = f['timeevo_m'][()]
    # timeevo_q = f['timeevo_q'][()]
    # timeevo_s = f['timeevo_s'][()]
    # tlist = f['tlist'][()]
    # Omega_p = f['Omega_p'][()]
    # Omega_q = f['Omega_q'][()]


    return f #nbar, qbar, sbar, timeevo_m, timeevo_q, timeevo_s, tlist, Omega_p, Omega_q