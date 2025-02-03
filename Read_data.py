import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # You can try 'Qt5Agg' or 'Agg' as well
import matplotlib.pyplot as plt

#AAAAAAA
from Maser_function import read_3d_array, load_custom_data, read_data_and_args

def custom_formatter(value, pos):
    return f'{value:.2f}'  # Adjust the format as needed


file_directory = r"X:\Data\Maser\Simulation\pythonProject1\Data\20240923\\"
common_directory = r''
total_directory = file_directory + common_directory
file_name = r"2024-09-24Masing_DAC_100000_g4_12.566370614359172_g_over_delta_True_6timeonqqmm.ddh5"


# nbar = load_custom_data(total_directory + 'maser.npy')
# sbar = load_custom_data(total_directory + 'SNAIL.npy')
# qbar = load_custom_data(total_directory + 'qubit.npy')
# Omega_p = load_custom_data(total_directory + 'Omega_p.npy')
# Omega_q = load_custom_data(total_directory + 'Omega_q.npy')
# tlist = load_custom_data(total_directory + 'tlist.npy')
#
#
# timeevo_m = read_3d_array(file_directory, common_directory + 'maser_timeevo.npy')
# timeevo_s = read_3d_array(file_directory, common_directory + 'SNAIL_timeevo.npy')
# timeevo_q = read_3d_array(file_directory, common_directory + 'qubit_timeevo.npy')

data, args = read_data_and_args(file_directory+file_name)
nbar = data['nbar']
sbar = data['sbar']
qbar = data['qbar']
Omega_p = data['Omega_p']
Omega_q = data['Omega_q']
tlist = data['tlist']
timeevo_q = data['timeevo_q']
timeevo_s = data['timeevo_s']
timeevo_m = data['timeevo_m']



distant_btw_ticks = 4


o_q = 7.08
o_p = 12.4
o_q_idx = np.argmin(abs(Omega_q - o_q))
o_p_idx = np.argmin(abs(Omega_p - o_p))
q = o_q_idx
P = o_p_idx
two_pi = np.pi * 2
x_mark, y_mark = q, P
y, x = nbar.shape
omega_p_ticks = Omega_p
omega_q_ticks = Omega_q

# Create a 2D color plot




# Plot the first image and scatter plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1]})


# new
cax = ax1.pcolormesh(Omega_q, Omega_p, nbar, shading='auto')
cbar = fig.colorbar(cax, ax=ax1, label='average photon number', fraction=0.046, pad=0.04)
ax1.scatter(o_q, o_p, color='red', marker='*', s=100, label='Marked Point')

ax1.set_xlabel('$\omega_q$(GHz)')
ax1.set_ylabel('$\omega_{p}$ (GHz)')
ax1.legend()
ax1.set_xlabel('$\omega_q$(GHz)')
ax1.set_ylabel('$\omega_{p}$ (GHz)')
ax1.legend()

# Set font for the entire figure
# font = {'family': 'monospace',
#         'weight': 'bold',
#         'size': 10}
# plt.rc('font', **font)

# Plot the second graph
ax2.plot(tlist, timeevo_m[P][q], color='red', label='Maser')
ax2.plot(tlist, timeevo_q[P][q], color='green', label='qubit')
ax2.plot(tlist, timeevo_s[P][q], color='blue', label='SNAIL')

# Labels and legend for the second graph
ax2.set_xlabel('Time (us)')
ax2.set_ylabel('expectation value')
ax2.legend()

plt.tight_layout()  # Adjust layout for better appearance
plt.show()



