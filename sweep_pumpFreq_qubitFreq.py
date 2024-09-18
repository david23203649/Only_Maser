import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from tqdm import tqdm
import time
import sympy as sym
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib import pyplot
import csv
from datetime import datetime
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import time

from Maser_function import *
from Pulse_and_time_dependent import Pulses

n=5
q=3
s =3

two_pi= 2 * np.pi
GHz = pow(10,3)
MHz = pow(10,0)
KHz = pow(10,-3)



g = basis(q,0)
e = basis(q,1)
f = basis(q,2)

id_qubit = qeye(q)
id_qubit2 = qeye(2)
id_maser = qeye(n)
id_SNAIL = qeye(s)

#ket


#operator
a_q_1 = Qobj([[0,1,0],[0,0,0],[0,0,0]],dims = [[3], [3]])
a_q_2 = Qobj([[0,0,0],[0,0,np.sqrt(2)],[0,0,0]],dims = [[3], [3]])


# globa operator
a_q_1_ext = tensor(a_q_1, id_maser,id_SNAIL)
a_dagger_q_1_ext = a_q_1_ext.dag()
a_q_2_ext = tensor(a_q_2, id_maser,id_SNAIL)
a_dagger_q_2_ext = a_q_2_ext.dag()

a_q = tensor(destroy(q), id_maser,id_SNAIL)
a_dagger_q = a_q.dag()

a_m = tensor(id_qubit, destroy(n),id_SNAIL)
a_dagger_m = a_m.dag()

e_op_m = a_m.dag()*a_m
e_op_q  = a_q.dag()*a_q

a_s = tensor(id_qubit, id_maser, destroy(s))
a_dagger_s = a_s.dag()



a_q2 = tensor(destroy(2), id_maser,id_SNAIL)
a_m2 = tensor(id_qubit2, destroy(n),id_SNAIL)
a_dagger_m2 = a_m2.dag()
e_op_m2 = a_m2.dag()*a_m2

state = ['g', 'e', 'f']
for i in range(0, n):
    for j in range(len(state)):
        for k in range(0, s):
            globals()[state[j] + '_' + str(i) + '_' + str(k)] = tensor(basis(3, j), fock(n, i), fock(s, k))
            # print(i,j,k)
# initial state and parameter

rho0 = g_0_0 * g_0_0.dag()

PI = np.pi

opts = Options(nsteps=1e6, atol=1e-8, rtol=1e-6)
p_bar = None  # qt.ui.TextProgressBar()
tlist = np.linspace(0, 20, 21)



p = Pulses()


def steadysol(Omega_p, Omega_q, tlist, threshold=0.001):
    start = time.time()
    steady_s = []
    steady_q = []
    steady_m = []
    timeevo_s = []
    timeevo_q = []
    timeevo_m = []
    timeevo_s_temp = []
    timeevo_q_temp = []
    timeevo_m_temp = []

    x = len(Omega_q)
    y = len(Omega_p)
    t = len(tlist)
    #     initial_array_s = np.zeros((x, y))
    #     initial_array_q = np.zeros((x, y))
    #     initial_array_m = np.zeros((x, y))
    for omega_p in tqdm(Omega_p, desc="Loading…", ascii=False, ncols=75):
        for omega_q in Omega_q:
            args = {'func': p.smoothbox,
                    'sigma': 0.02,
                    'amp': 10000,
                    'omega': 2 * PI * omega_p * GHz,
                    't0': 0,
                    'width': 200,
                    'omega_q': 2 * PI * omega_q * GHz,
                    'omega_m': 2 * PI * 7 * GHz,
                    'omega_s': 2 * PI * 5 * GHz,
                    'alpha': 2 * PI * 160 * MHz,
                    'g1': 2 * PI * 0.5 * MHz,
                    'g3': 2 * PI * 60 * MHz,
                    'g4': 2 * PI * 0 * MHz,
                    'g5': 2 * PI * 0 * MHz,
                    'gamma_cav': 100 * KHz * two_pi,
                    'gamma_down': 16 * KHz * two_pi,
                    'gamma_SNAIL': 10 * MHz * two_pi
                    }

            alpha = args['alpha']
            g1 = args['g1']

            g3 = args['g3']
            g4 = args['g4']
            g5 = args['g5']
            gamma_cav = args['gamma_cav']
            gamma_down = args['gamma_down']
            gamma_SNAIL = args['gamma_SNAIL']
            Delta = args['omega_q'] - args['omega_m']

            H_0_qm = Delta * (a_dagger_q * a_q) - \
                     alpha / 2 * (a_q.dag() * a_q.dag() * a_q * a_q) + \
                     g1 * (a_q.dag() * a_m + a_q * a_m.dag())

            H_0_qm = [H_0_qm,
                      [g3 * (a_s * a_q), p.SNAIL3], [g3 * a_s.dag() * a_q.dag(), p.SNAIL3_dag],
                      [g4 * (a_s.dag() * a_q), p.SNAIL4], [g4 * a_s * a_q.dag(), p.SNAIL4_dag],
                      [g5 * (a_s.dag() * a_q), p.SNAIL5], [g5 * a_s * a_q.dag(), p.SNAIL5_dag]]

            e_op_s = a_s.dag() * a_s
            e_op_q = a_q.dag() * a_q
            e_op_m = a_m.dag() * a_m
            e_ops = [e_op_s, e_op_q, e_op_m]
            C_maser = lindblad_dissipator(np.sqrt(gamma_cav) * a_m)

            C_qubit_down = lindblad_dissipator(np.sqrt(gamma_down) * (a_q))
            C_SNAIL_down = lindblad_dissipator(np.sqrt(gamma_SNAIL) * (a_s))

            c_ops_notime = [C_maser, C_qubit_down, C_SNAIL_down]

            result = mesolve(H_0_qm, rho0, tlist, c_ops_notime, e_ops, args=args, options=opts, progress_bar=p_bar)

            steady_s.append(find_steady_value(result.expect[0]))
            steady_q.append(find_steady_value(result.expect[1]))
            steady_m.append(find_steady_value(result.expect[2]))

            timeevo_s_temp.append(result.expect[0])
            timeevo_q_temp.append(result.expect[1])
            timeevo_m_temp.append(result.expect[2])

    #         initial_array_s = np.dstack(initial_array_s,timeevo_s_temp)
    #         initial_array_q = np.dstack(initial_array_q,timeevo_q_temp)
    #         initial_array_m = np.dstack(initial_array_m,timeevo_m_temp)

    steady_m = np.reshape(steady_m, (y, x))
    steady_q = np.reshape(steady_q, (y, x))
    steady_s = np.reshape(steady_s, (y, x))

    timeevo_s = np.reshape(timeevo_s_temp, (y, x, t))
    timeevo_q = np.reshape(timeevo_q_temp, (y, x, t))
    timeevo_m = np.reshape(timeevo_m_temp, (y, x, t))
    end = time.time()
    elapsed_time = end - start
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60

    # Print the elapsed time in minutes and seconds
    print(f"Time taken to run the code: {minutes} minutes {seconds:.2f} seconds")

    return steady_m, steady_q, steady_s, timeevo_s, timeevo_q, timeevo_m, args



#Omega_p = np.linspace(11.95,12.05,11)
Omega_p = np.linspace(11.96,12.1,2)
Omega_q = np.linspace(6.96,7.2,2)
tlist = np.linspace(0,40,41)
nbar = []
qbar = []
sbar = []

nbar, qbar, sbar, timeevo_s, timeevo_q, timeevo_m,args = steadysol(Omega_p,Omega_q,tlist)
maskq = qbar == None
maskn = nbar == None
masks = sbar == None
qbar[maskq] = 0
nbar[maskn] = 0
sbar[masks] = 0


#save_as_csv(nbar,filename_nbar)
file_directory = r"C:\Users\Chun-Che\PycharmProjects\pythonProject1\Data\\"
from Maser_function import save_with_custom_info2
save_with_custom_info2(nbar, 'array_data',args, 'maser',file_directory)
save_with_custom_info2(sbar, 'array_data',args, 'SNAIL',file_directory)
save_with_custom_info2(qbar, 'array_data',args, 'qubit',file_directory)

save_3d_array(timeevo_m, 'array_data',args, 'maser_timeevo',file_directory)
save_3d_array(timeevo_s, 'array_data',args, 'SNAIL_timeevo',file_directory)
save_3d_array(timeevo_q, 'array_data',args, 'qubit_timeevo',file_directory)




x = len(Omega_q)
y = len(Omega_p)
pyplot.figure(figsize=((y, x)))

im = plt.imshow(nbar)

font = {'family': 'monospace',

        'weight': 'bold',
        'size': 20}

plt.xlabel('$\Delta$(GHz)')
# plt.ylabel('g1/g2')
plt.ylabel('$\omega_{p}$ (GHz)')
x_ticks = [0, x / 2, x - 1]  # Custom tick positions for the X-axis
x_labels = ['6.96', '7.08', '7.2']  # Custom tick labels for the X-axis

y_ticks = [0, y / 2, y - 1]  # Custom tick positions for the Y-axis
y_labels = ['11.96', '12.03', '12.1']  # Custom tick labels for the Y-axis
plt.colorbar(im, label='average photon number', fraction=0.046, pad=0.04)
plt.xticks(x_ticks, x_labels)
plt.yticks(y_ticks, y_labels)

# Add a color bar

plt.rc('font', **font)
plt.show()

## increase SNAIL gamma