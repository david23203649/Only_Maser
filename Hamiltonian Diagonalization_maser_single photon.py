import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # You can try 'Qt5Agg' or 'Agg' as well
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from System_setup_class import *
from tqdm import tqdm
from Maser_function import *
def find_state_index(state_density,eigenvector,number_of_eigenstate):
    # non_zero_eigenvalues = sorted_eigenvalues[np.nonzero(sorted_eigenvalues)]
    temp = []
    for i in range(number_of_eigenstate):
        temp.append(expect(state_density,eigenvector[i]))
    closest_to_one_index = min(range(len(temp)), key=lambda i: abs(temp[i] - 1))
    return closest_to_one_index
def seperate_lower_upper(a, b ):
    diff = []
    for i in range(len(a)):

        if a[i] > b[i]:
            b[i],a[i] = a[i], b[i]

    return a, b
def find_diff(a,b):
    diff = []
    for i in range(len(a)):
        diff.append(abs(a[i]-b[i]))
    return diff
# Create subplot


# Initial value for g_qb

# Create axes for g_qb slider

# Create a slider for g_qb

p = Pulses()

args = {'func': p.smoothbox,
                'sigma': 0.02,
                'amp': 10000,

                't0': 0,
                'width': 200,
                'alpha': 2 * PI * 160 * MHz,
                'omega_m': 2 * PI * 7 * GHz,
                'omega_s': 2 * PI * 5.4 * GHz,


                'g3': 2 * PI * 20 * MHz,
                'g4': 2 * PI * 2 * MHz,

                'gamma_cav': 100 * KHz * two_pi,
                'gamma_down': 16 * KHz * two_pi,
                'gamma_SNAIL': 10 * MHz * two_pi,
                'g_sq': 300 * MHz * two_pi,
                'g_sm': 300 * KHz * two_pi,
                'g_qm': 1.2 * MHz * two_pi

                }
# Function to update the plot based on g_qb value



# Your quantum system setup and plotting code here
# You need to replace this section with your own code
q = 3

m = 3
number_of_eigenstate = 9

dimensions = ( q , m )
operators = QuantumOperators(*dimensions)
q_0 = operators.basis(0,0)
m_0 = operators.basis(1,0)
q_1 = operators.basis(0, 1)
m_1 = operators.basis(1, 1)
q_2 = operators.basis(0, 2)
m_2 = operators.basis(1, 2)



q_0_m_1 = tensor(q_0 * q_0.dag(), m_1 * m_1.dag())
q_1_m_0 = tensor(q_1 * q_1.dag(), m_0 * m_0.dag())


a_q = operators.annihilation(0)
a_m = operators.annihilation(1)
a_q_dagger = a_q.dag()

a_m_dagger = a_m.dag()




phi = args.get('phi', 0)

omega_s = args['omega_s']
omega_m = args['omega_m']
phase1 = two_pi * np.linspace(0.35, 0.45, 10001)
Omega_q = np.sqrt ( 64000000 * abs(np.cos(phase1))) *two_pi
g3 = args['g3']
g4 = args['g4']

alpha = args['alpha']
g_qm = args['g_qm']
g_sq = args['g_sq']
g_sm = args['g_sm']

gamma_cav = args['gamma_cav']
gamma_down = args['gamma_down']
gamma_SNAIL = args['gamma_SNAIL']

G_qm = np.linspace(20,320, 31)
min_diff_array = []
min_diff_array_single = []
min_diff_array_transformed = []
### parallel
# def find_diff_sweeping_alpha(alpha):
#     eigenvalues_q0m2 = []
#     eigenvalues_q2m0 = []
#
#     for omega_q in Omega_q:
#         Delta = omega_q - omega_m
#         if Delta == 0:
#             Delta = pow(10, -6)
#
#         H = omega_q * (a_q_dagger * a_q) + omega_m * (a_m_dagger * a_m) \
#             - (alpha / 12) * (a_q_dagger + a_q) ** 4 \
#             + g_qm * (a_q_dagger + a_q) * (a_m_dagger + a_m)
#
#         # H = omega_q * (a_q_dagger * a_q) - (alpha / 12) * (a_q_dagger +  a_q)**4
#
#         eigenvalues, eigenvector = H.eigenstates()
#
#         eigenvalues_q0m2.append(eigenvalues[find_state_index(q_0_m_2, eigenvector, number_of_eigenstate - 1)] / two_pi)
#         eigenvalues_q2m0.append(eigenvalues[find_state_index(q_2_m_0, eigenvector, number_of_eigenstate - 1)] / two_pi)
#
#     a, b = seperate_lower_upper(eigenvalues_q2m0, eigenvalues_q0m2)
#     diff = find_diff(a, b)
#     min_diff_array.append(min(diff))
#     return min_diff_array, eigenvalues_q2m0, eigenvalues_q0m2
#
# result = parallel_map(find_diff_sweeping_alpha, Alpha, progress_bar=True)
eigenvalues_q0m1_matrix = []
eigenvalues_q1m0_matrix = []


for g_qm in tqdm(G_qm):
    print(f"Processing {g_qm}")
    g_qm = ((g_qm*two_pi * MHz).round(4))
    eigenvalues_q0m1 = []
    eigenvalues_q1m0 = []

    for omega_q in Omega_q:
        Delta = omega_q - omega_m
        if Delta == 0:
            Delta = pow(10, -6)

        H = omega_q * (a_q_dagger * a_q) + omega_m * (a_m_dagger * a_m) \
            - (alpha / 12) * (a_q_dagger + a_q) ** 4 \
            + g_qm * (a_q_dagger + a_q) * (a_m_dagger + a_m)

        lambda_qm = g_qm / Delta


        omega_q_prime = omega_q + lambda_qm * g_qm

        omega_m_prime = omega_m - lambda_qm * g_qm
        Delta_prime = omega_q_prime - omega_m_prime
        H_transformed = Delta_prime * (a_q_dagger * a_q) \
                        - (alpha / 2) * (a_q.dag() * a_q.dag() * a_q * a_q) \
                        - (alpha / 2) * (lambda_qm) ** 2 * (a_q.dag() * a_q * a_m.dag() * a_m) \
                        - (alpha / 2) * (lambda_qm) ** 2 * (
                                    a_q.dag() * a_q.dag() * a_m * a_m + a_q * a_q * a_m.dag() * a_m.dag()) \

            #H = omega_q * (a_q_dagger * a_q) - (alpha / 12) * (a_q_dagger +  a_q)**4

        eigenvalues, eigenvector = H.eigenstates()
        eigenvalues_q0m1.append(eigenvalues[find_state_index(q_0_m_1, eigenvector, number_of_eigenstate - 1)] / two_pi)
        eigenvalues_q1m0.append(eigenvalues[find_state_index(q_1_m_0, eigenvector, number_of_eigenstate - 1)] / two_pi)



    eigenvalues_q0m1_matrix.append(eigenvalues_q0m1)  # Stack the q0m2 eigenvalues
    eigenvalues_q1m0_matrix.append(eigenvalues_q1m0)  # Stack the q2m0 eigenvalues

    e, f = seperate_lower_upper(eigenvalues_q1m0, eigenvalues_q0m1)
    diff_single = find_diff(e, f)
    min_diff_array_single.append(min(diff_single))

eigenvalues_q0m1_matrix = np.array(eigenvalues_q0m1_matrix).reshape(len(G_qm), len(Omega_q))
eigenvalues_q1m0_matrix = np.array(eigenvalues_q1m0_matrix).reshape(len(G_qm), len(Omega_q))


# plt.figure()
# plt.plot(Alpha, min_diff_array)
# plt.xlabel('alpha (MHz)')
# plt.ylabel('g_qqmm (MHz)')

non_zero_indices = [i for i, x in enumerate(min_diff_array_single) if x != 0]

min_diff_array_single_non_zero = [x for x in min_diff_array_single if x != 0]
qgm_adjusted1 = G_qm[non_zero_indices]





# Plotting the updated figure
plt.figure()
plt.title('single photon')
#plt.plot(Alpha_adjusted, min_diff_array_non_zero, label = 'w/o transformation')
plt.plot(qgm_adjusted1, min_diff_array_single_non_zero, label = 'single photon transformation')
#plt.plot(Alpha_adjusted2, min_diff_array_non_zero_transformed, label = 'w/ transformation')
plt.xlabel('g_qm (MHz)')
plt.ylabel('avoiding crossing (MHz)')
plt.grid(True)
plt.show()


#plot individual
alpha_wanted = 230
alpha_idx = np.argmin(np.abs(G_qm - alpha_wanted))
plt.figure()
plt.title(f'alpha = {alpha_wanted} (HMz)')

plt.plot(phase1/two_pi, eigenvalues_q1m0_matrix[alpha_idx,:])
plt.plot(phase1/two_pi, eigenvalues_q0m1_matrix[alpha_idx,:])



