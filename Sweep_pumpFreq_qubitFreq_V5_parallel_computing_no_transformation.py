import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # You can try 'Qt5Agg' or 'Agg' as well
import matplotlib.pyplot as plt
from functools import partial
from qutip import *
from qutip.parallel import parallel_map
from tqdm import tqdm
import time
import sympy as sym
import pandas as pd
from scipy.optimize import curve_fit

import csv
from datetime import datetime
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import time

from Maser_function import *
from Pulse_and_time_dependent import Pulses
# testtttttt
n=5
q=3
s =3

PI = np.pi
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



opts = Options(nsteps=1e6, atol=1e-8, rtol=1e-6)
p_bar = None  # qt.ui.TextProgressBar()

p = Pulses()



def steadysol_sweep_omega_q(Omega_q, tlist, args ):

    steady_s = []
    steady_q = []
    steady_m = []
    timeevo_s_temp = []
    timeevo_q_temp = []
    timeevo_m_temp = []

    args['omega_q'] = Omega_q
    omega_m = args['omega_m']
    omega_S = args['omega_s']
    alpha = args['alpha']
    g1 = args['g1']
    phi = args.get('phi', 0)
    g3 = args['g3']
    g4 = args['g4']
    g_sm = args['g_sm']



    gamma_cav = args['gamma_cav']
    gamma_down = args['gamma_down']
    gamma_SNAIL = args['gamma_SNAIL']



    e_op_s = a_s.dag() * a_s
    e_op_q = a_q.dag() * a_q
    e_op_m = a_m.dag() * a_m
    e_ops = [e_op_s, e_op_q, e_op_m]
    C_maser = lindblad_dissipator(np.sqrt(gamma_cav) * a_m)

    C_qubit_down = lindblad_dissipator(np.sqrt(gamma_down) * (a_q))
    C_SNAIL_down = lindblad_dissipator(np.sqrt(gamma_SNAIL) * (a_s))

    c_ops_notime = [C_maser, C_qubit_down, C_SNAIL_down]





    H_0_qm = Omega_q * (a_dagger_q * a_q) + omega_m * (a_dagger_m * a_m) + omega_S * (a_dagger_s * a_s)- \
             alpha / 12 * (a_q.dag()+ a_q)**4 + \
             g1 * (a_q.dag()+ a_q) * (a_m  + a_m.dag()) + g_sm * (a_s.dag()+ a_s) * (a_m  + a_m.dag()) +\
             g3 * (a_s + a_dagger_s)**3 + \
             g4 * (a_s + a_dagger_s)**4

    H_0_qm = [H_0_qm,
              [ (a_s + a_dagger_s), p.cos_drive]]

    result = mesolve(H_0_qm, rho0, tlist, c_ops_notime, e_ops, args=args, options=opts)

    steady_s.append(find_steady_value(result.expect[0]))

    steady_q.append(find_steady_value(result.expect[1]))
    steady_m.append(find_steady_value(result.expect[2]))
    timeevo_s_temp.append(result.expect[0])
    timeevo_q_temp.append(result.expect[1])
    timeevo_m_temp.append(result.expect[2])

    #         initial_array_s = np.dstack(initial_array_s,timeevo_s_temp)
    #         initial_array_q = np.dstack(initial_array_q,timeevo_q_temp)
    #         initial_array_m = np.dstack(initial_array_m,timeevo_m_temp)




    return steady_m, steady_q, steady_s, timeevo_s_temp, timeevo_q_temp, timeevo_m_temp

def steadysol_sweep_omega_p(Omega_p, Omega_q, tlist, args):
    start = time.time()
    if __name__ == '__main__':

        x = len(Omega_q)
        y = len(Omega_p)
        t = len(tlist)
        steady_s = []
        steady_q = []
        steady_m = []
        timeevo_s = []
        timeevo_q = []
        timeevo_m = []

        for omega_p in tqdm(Omega_p, desc="Loading…", ascii=False, ncols=75):
            args['omega'] = omega_p


            func = partial(steadysol_sweep_omega_q,tlist = tlist, args = args)
            result = parallel_map(func, Omega_q, progress_bar=True)
            #print(result)

            for tuple_data in result:
                steady_m.append(tuple_data[0])
                steady_q.append(tuple_data[1])
                steady_s.append(tuple_data[2])

                timeevo_m.append(tuple_data[3])
                timeevo_q.append(tuple_data[4])
                timeevo_s.append(tuple_data[5])

    steady_m = np.reshape(steady_m, (y, x))
    steady_q = np.reshape(steady_q, (y, x))
    steady_s = np.reshape(steady_s, (y, x))


    timeevo_m = np.reshape(timeevo_m, (y, x, t))
    timeevo_q = np.reshape(timeevo_q, (y, x, t))
    timeevo_s = np.reshape(timeevo_s, (y, x, t))


    steady_m = np.array(steady_m.tolist(), dtype=float)
    steady_q = np.array(steady_q.tolist(), dtype=float)
    steady_s = np.array(steady_s.tolist(), dtype=float)
    end = time.time()
    elapsed_time = end - start
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60

    # Print the elapsed time in minutes and seconds
    print(f"Time taken to run the code: {minutes} minutes {seconds:.2f} seconds")
    return steady_m,steady_q, steady_s, timeevo_m, timeevo_q, timeevo_s, Omega_p/two_pi/1e3, Omega_q/two_pi/1e3, tlist


#Omega_p = np.linspace(11.95,12.05,11)
Omega_p = np.linspace(12.3 * GHz * two_pi,12.5 * GHz * two_pi,21)
Omega_q = np.linspace(6.96 * GHz * two_pi,7.2 * GHz * two_pi,25)
tlist = np.linspace(0,20,61)

g_over_delta = True
amps = [30000]

for amp in amps:
    if __name__ == '__main__':
        args = {'func': p.smoothbox,
                'sigma': 0.02,
                'amp': amp,
                'omega': 2 * PI * 12 * GHz,
                't0': 0,
                'width': 200,
                'omega_q': 2 * PI * 7 * GHz,

                'omega_m': 2 * PI * 7 * GHz,
                'omega_s': 2 * PI * 5.4 * GHz,
                'alpha': 2 * PI * 160 * MHz,
                'g1': 2 * PI * 0.9 * MHz,
                'g3': 2 * PI * -20 * MHz,
                'g4': 2 * PI * -5 * MHz,
                'g5': 2 * PI * 0 * MHz,
                'gamma_cav': 100 * KHz * two_pi,
                'gamma_down': 16 * KHz * two_pi,
                'gamma_SNAIL': 10 * MHz * two_pi,
                'g_sq': 300 * MHz * two_pi,
                'g_sm': 300 * KHz * two_pi,

                }


        nbar, qbar, sbar, timeevo_s, timeevo_q, timeevo_m, Omega_p, Omega_q, tlist = steadysol_sweep_omega_p(Omega_p, Omega_q, tlist, args)

        args['func'] = str(args['func'])

        # print('time out 5 min')
        # time.sleep(300)


        #Save
        filepath = r"Z:\Data\Maser\Simulation\pythonProject1\Data\woRWA\\"
        current_date = datetime.now().strftime('%Y-%m-%d')
        filename =  current_date + f"_Masing_DAC_{amp}_woRWA.ddh5"
        #savedata(filepath + filename, Pump_freq=Omega_p, qubit_freq=Omega_q, steadystate = nbar, time_data = tlist,amp=args['amp'], omega_s = args['omega_s']/two_pi, g_qm = args['g1']/two_pi, g3 = args['g3']/two_pi, g4 = args['g4']/two_pi, gamma_cav = args['gamma_cav']/two_pi, gamma_down = args['gamma_down']/two_pi, gamma_SNAIL = args['gamma_SNAIL']/two_pi )
        # save_with_custom_info2(nbar, 'array_data_V4',args, 'maser',filepath)
        # save_with_custom_info2(sbar, 'array_data_V4',args, 'SNAIL',filepath)
        # save_with_custom_info2(qbar, 'array_data_V4',args, 'qubit',filepath)
        # save_with_custom_info2(Omega_p, 'array_data_V4', args, 'Omega_p', filepath)
        # save_with_custom_info2(Omega_q, 'array_data_V4', args, 'Omega_q', filepath)
        # save_with_custom_info2(tlist, 'array_data_V4', args, 'tlist', filepath)
        # save_3d_array(timeevo_m, 'array_data_V4',args, 'maser_timeevo',filepath)
        # save_3d_array(timeevo_s, 'array_data_V4',args, 'SNAIL_timeevo',filepath)
        # save_3d_array(timeevo_q, 'array_data_V4',args, 'qubit_timeevo',filepath)

        data_dict = {'nbar' : nbar,
                    'sbar' : sbar,
                    'qbar' : qbar,
                    'timeevo_m' : timeevo_m,
                    'timeevo_s' : timeevo_s,
                    'timeevo_q' : timeevo_q,
                    'tlist' : tlist,
                    'Omega_p' : Omega_p,
                    'Omega_q' : Omega_q
                    }

        save_data_and_args(filepath + filename, data_dict, **args)

    ## plot

        x = len(Omega_q)
        y = len(Omega_p)
        #pyplot.figure(figsize=(y, x))
        distant_btw_ticks = 6
        x_ticks = np.arange(nbar.shape[1])[::distant_btw_ticks]
        y_ticks = np.arange(nbar.shape[0])[::distant_btw_ticks]

        # Create a 2D color plot
        plt.imshow(nbar, origin='lower')
                #plt.scatter(x_mark, y_mark, color='red', marker='*', s=1000, label='Marked Point')
        plt.xlabel('$\omega_q$(GHz)')
        plt.ylabel('$\omega_{p}$ (GHz)')

        plt.colorbar(label='average photon number', fraction=0.046, pad=0.04)
        plt.xticks(x_ticks, [custom_formatter(value, None) for value in Omega_q[x_ticks]])
        plt.yticks(y_ticks, [custom_formatter(value, None) for value in Omega_p[y_ticks]])


    ## increase SNAIL gamma

