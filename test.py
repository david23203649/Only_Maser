import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # You can try 'Qt5Agg' or 'Agg' as well
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from System_setup_class import *

from Maser_function import *
# Create subplot

# Initial value for g_qb
initial_time = 100
initial_alpha = 160 * MHz
initial_pump_freq = 2 * PI * 12 * GHz
# Create axes for g_qb slider

# Create a slider for g_qb

p = Pulses()

args = {'func': p.smoothbox,
                'sigma': 0.02,
                'amp': 10000,

                't0': 0,
                'width': 200,

                'omega_m': 2 * PI * 7 * GHz,
                'omega_s': 2 * PI * 5.4 * GHz,

                'g1': 2 * PI * 1.4 * MHz,
                'g3': 2 * PI * 20 * MHz,
                'g4': 2 * PI * 2 * MHz,

                'gamma_cav': 100 * KHz * two_pi,
                'gamma_down': 16 * KHz * two_pi,
                'gamma_SNAIL': 10 * MHz * two_pi,
                'g_sq': 300 * MHz * two_pi,
                'g_sm': 300 * KHz * two_pi,
                'g_qm': 1.2 * MHz * two_pi

                }

    # Your quantum system setup and plotting code here
    # You need to replace this section with your own code
q = 3

m = 3
number_of_eigenstate = 9

def find_state_index(state_density,eigenvector,number_of_eigenstate = number_of_eigenstate):
    # non_zero_eigenvalues = sorted_eigenvalues[np.nonzero(sorted_eigenvalues)]
    temp = []
    for i in range(number_of_eigenstate):
        temp.append(expect(state_density,eigenvector[i]))
    closest_to_one_index = min(range(len(temp)), key=lambda i: abs(temp[i] - 1))
    return closest_to_one_index

dimensions = ( q , m )
operators = QuantumOperators(*dimensions)

a_q = operators.annihilation(0)
a_m = operators.annihilation(1)
a_q_dagger = a_q.dag()

a_m_dagger = a_m.dag()



g1 = args['g1']
phi = args.get('phi', 0)

omega_s = args['omega_s']
omega_m = args['omega_m']
phase1 = two_pi * np.linspace(0.375, 0.4, 20001)
Omega_q = np.sqrt ( 64000000 * abs(np.cos(phase1))) *two_pi
g3 = args['g3']
g4 = args['g4']


g_qm = args['g_qm']
g_sq = args['g_sq']
g_sm = args['g_sm']

gamma_cav = args['gamma_cav']
gamma_down = args['gamma_down']
gamma_SNAIL = args['gamma_SNAIL']
alpha = two_pi * 160 * MHz
eigenvalues_matrix = []
eigenvalues_matrix_transformed = []
omega_q = two_pi * 7 * GHz

Delta = omega_q - omega_m
if Delta == 0:
    Delta = pow(10, -6)
Delta_qs = omega_q - omega_s
Delta_sm = omega_s - omega_m
lambda_qm = g_qm / Delta
lambda_qs = g_sq / Delta_qs
lambda_sm = g_sm / Delta_sm

omega_q_prime = omega_q + lambda_qm * g_qm + lambda_qs * g_sq
omega_s_prime = omega_s - lambda_qs * g_sq + lambda_sm * g_sm
omega_m_prime = omega_m - lambda_qm * g_qm - lambda_sm * g_sm
Delta_prime =  omega_q_prime - omega_m_prime
args['omega_q_prime'] = omega_q_prime
args['omega_s_prime'] = omega_s_prime
args['omega_m_prime'] = omega_m_prime

q_0 = basis(q,0)
q_1 = basis(q,1)
q_2 = basis(q,2)
q_0_global = tensor(q_0 * q_0.dag(), qeye(3))
q_1_global = tensor(q_1 * q_1.dag(), qeye(3))
q_2_global = tensor(q_1 * q_1.dag(), qeye(3))
m_0 = basis(m,0)
m_1 = basis(m,1)
m_2 = basis(m,2)
m_0_global = tensor(m_0, qeye(3))
m_1_global = tensor(m_1, qeye(3))
m_2_global = tensor(m_2, qeye(3))
expt_q_0 = q_0_global.dag() * q_0_global
expt_m_0 = m_0_global.dag() * m_0_global
q_0_m_2 = tensor(q_0 * q_0.dag(), m_2 * m_2.dag())
q_2_m_0 = tensor(q_2 * q_2.dag(), m_0 * m_0.dag())

H = omega_q * (a_q_dagger * a_q) + omega_m * (a_m_dagger * a_m) \
    - (alpha / 12) * (a_q_dagger + a_q) ** 4 \
    + g_qm * (a_q_dagger + a_q) * (a_m_dagger + a_m)


#H = omega_q * (a_q_dagger * a_q) - (alpha / 12) * (a_q_dagger +  a_q)**4

eigenvalues, eigenvector = H.eigenstates()
plt.figure()
plt.plot(eigenvalues[find_state_index(q_0_m_2, eigenvector)])


# sorted_indices = np.argsort(np.abs(eigenvalues))
# sorted_eigenvalues = eigenvalues[sorted_indices]


# H_transformed = Delta_prime * (a_q_dagger * a_q) + \
#          g1 * (a_q.dag() * a_m + a_q * a_m.dag()) + \
#          g3 *(lambda_qs)* (a_q.dag() * a_s.dag() + a_q * a_s) * p.SNAIL3(t=time, args= args) + \
#          g4 * (a_s.dag() * a_s.dag() * a_s * a_s) \
#          - (alpha / 2) * (a_q.dag() * a_q.dag() * a_q * a_q) \
#          - (alpha / 2) * (lambda_qm) ** 2 * (a_q.dag() * a_q * a_m.dag() * a_m) \
#          - (alpha / 2) * (lambda_qm) ** 2 * (a_q.dag() * a_q.dag() * a_m * a_m + a_q * a_q * a_m.dag() * a_m.dag()) \
#          - (alpha / 2) * (lambda_qs) ** 2 * (a_q.dag() * a_q * a_s.dag() * a_s)
#
# eigenvalues_transformed, _ = np.linalg.eig(H_transformed)
# sorted_indices_transformed = np.argsort(np.abs(eigenvalues_transformed))
# sorted_eigenvalues_transformed = eigenvalues_transformed[sorted_indices_transformed]
# non_zero_eigenvalues_transformed = sorted_eigenvalues_transformed[np.nonzero(sorted_eigenvalues_transformed)]
# eigenvalues_matrix_transformed.append(non_zero_eigenvalues_transformed[:number_of_eigenstate])
    #print(np.shape(non_zero_eigenvalues_transformed[:number_of_eigenstate]))
    #print(np.shape(eigenvalues_matrix_transformed))
#print(np.shape(eigenvalues_matrix))


