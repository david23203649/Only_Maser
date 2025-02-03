import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # You can try 'Qt5Agg' or 'Agg' as well
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from System_setup_class import *

from Maser_function import *
# Create subplot
fig, (ax, ax_transformed) = plt.subplots(2, 1, figsize=(12, 10))
plt.subplots_adjust(bottom=0.3)

# Initial value for g_qb
initial_time = 100
initial_alpha = 160 * MHz
initial_pump_freq = 2 * PI * 12 * GHz
# Create axes for g_qb slider
ax_time = plt.axes([0.25, 0.1, 0.65, 0.03])
ax_alpha= plt.axes([0.25, 0.15, 0.65, 0.03])
ax_pump_freq = plt.axes([0.25, 0.2, 0.65, 0.03])
# Create a slider for g_qb
time_slider = Slider(ax_time, 'Time (us)', 0, 200,  valinit=initial_time, valstep=1.0)
alpha_slider = Slider(ax_alpha, 'alpha (MHz)', 0,  2000* MHz,  valinit=initial_alpha, valstep=1)
pump_freq_slider = Slider(ax_pump_freq, 'pump freq (MHz)',  11.5 * GHz,  12.5 * GHz,  valinit=initial_pump_freq, valstep= MHz)
p = Pulses()

args = {'func': p.smoothbox,
                'sigma': 0.02,
                'amp': 10000,

                't0': 0,
                'width': 200,

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
def update_plot(val):
    time = time_slider.val
    alpha = two_pi * alpha_slider.val
    args['omega'] = two_pi * pump_freq_slider.val
    # Your quantum system setup and plotting code here
    # You need to replace this section with your own code
    q = 5
    s = 2
    m = 5
    number_of_eigenstate = 18
    dimensions = (s, q , m )
    operators = QuantumOperators(*dimensions)
    a_s = operators.annihilation(0)
    a_q = operators.annihilation(1)
    a_m = operators.annihilation(2)
    a_q_dagger = a_q.dag()
    a_s_dagger = a_s.dag()
    a_m_dagger = a_m.dag()




    phi = args.get('phi', 0)
    omega = args['omega']
    omega_s = args['omega_s']
    omega_m = args['omega_m']
    phase1 = two_pi * np.linspace(0.35, 0.45, 10001)
    Omega_q = np.sqrt ( 64000000 * abs(np.cos(phase1))) *two_pi
    g3 = args['g3']
    g4 = args['g4']


    g_qm = args['g_qm']
    g_sq = args['g_sq']
    g_sm = args['g_sm']

    gamma_cav = args['gamma_cav']
    gamma_down = args['gamma_down']
    gamma_SNAIL = args['gamma_SNAIL']
    eigenvalues_matrix = []
    eigenvalues_matrix_transformed = []
    for omega_q in Omega_q:
        Delta = omega_q - omega_m
        if Delta == 0:
            Delta = pow(10, -6)

        lambda_qm = g_qm / Delta

        omega_q_prime = omega_q + lambda_qm * g_qm
        omega_s_prime = omega_s
        omega_m_prime = omega_m - lambda_qm * g_qm
        Delta_prime =  omega_q_prime - omega_m_prime
        args['omega_q_prime'] = omega_q_prime
        args['omega_s_prime'] = omega_s_prime
        args['omega_m_prime'] = omega_m_prime

        H = omega_s * (a_s_dagger * a_s) + omega_q * (a_q_dagger * a_q) + omega_m * (a_m_dagger * a_m)  \
                        - (alpha / 12) * (a_q_dagger + a_q)**4 \
                        + g_sq * (a_s_dagger + a_s) * (a_q_dagger + a_q) \
                        + g_qm * (a_q_dagger + a_q) * (a_m_dagger + a_m) \
                        + g_sq * (a_q_dagger + a_q) * (a_s_dagger + a_s) \
                        + g3 * (a_s + a_s_dagger)**3 \
                        + g4 * (a_s + a_s_dagger)**4 \
                        +  (args['func'](time, args)*np.exp(-1j *omega * time) + args['func'](time, args)*np.exp(1j *omega * time) ) * (a_s_dagger + a_s)
        H = omega_q * (a_q_dagger * a_q) + omega_m * (a_m_dagger * a_m) \
            - (alpha / 12) * (a_q_dagger + a_q) ** 4 \
            + g_qm * (a_q_dagger + a_q) * (a_m_dagger + a_m)


        #H = omega_q * (a_q_dagger * a_q) - (alpha / 12) * (a_q_dagger +  a_q)**4
        H_transformed = omega_q_prime * (a_q_dagger * a_q) + omega_m_prime * (a_m_dagger * a_m) \
                        - (alpha / 12) * (a_q + lambda_qm * a_m + a_q.dag() + lambda_qm * a_m.dag()) ** 4


        eigenvalues, _ = np.linalg.eig(H)
        sorted_indices = np.argsort(np.abs(eigenvalues))
        sorted_eigenvalues = eigenvalues[sorted_indices]
        non_zero_eigenvalues = sorted_eigenvalues[np.nonzero(sorted_eigenvalues)]
        eigenvalues_matrix.append(non_zero_eigenvalues[:number_of_eigenstate])

        eigenvalues_transformed, _ = np.linalg.eig(H_transformed)
        sorted_indices_transformed = np.argsort(np.abs(eigenvalues_transformed))
        sorted_eigenvalues_transformed = eigenvalues_transformed[sorted_indices_transformed]
        non_zero_eigenvalues_transformed = sorted_eigenvalues_transformed[np.nonzero(sorted_eigenvalues_transformed)]
        eigenvalues_matrix_transformed.append(non_zero_eigenvalues_transformed[:number_of_eigenstate])
        #print(np.shape(non_zero_eigenvalues_transformed[:number_of_eigenstate]))
        #print(np.shape(eigenvalues_matrix_transformed))
    #print(np.shape(eigenvalues_matrix))
    eigenvalues_matrix = np.array(eigenvalues_matrix)
    eigenvalues_matrix_transformed = np.array(eigenvalues_matrix_transformed)

    ax.clear()
    for i in range(number_of_eigenstate):
        ax.plot(phase1/two_pi, eigenvalues_matrix[:, i]/two_pi, label=f'Eigenvalue {i + 1}')
    ax.set_ylim(11000, 18000)  # Set x-axis range from 2 to 8
    ax.set_xlabel('Phase')
    ax.set_ylabel('Eigenvalues')
    ax.set_title('Eigenvalues vs. Phase')
   # ax.legend()

    plt.draw()
    ax_transformed.clear()
    for i in range(number_of_eigenstate):

        ax_transformed.plot(phase1/two_pi, eigenvalues_matrix_transformed[:, i]/two_pi, label=f'Transformed Eigenvalue {i + 1}')
    #ax_transformed.set_ylim(-5000, 3000)
    ax.set_ylim(11000, 18000)  # Set x-axis range from 2 to 8
    ax_transformed.set_xlabel('Phase')
    ax_transformed.set_ylabel('Eigenvalues')
    ax_transformed.set_title('Transformed Eigenvalues vs. Phase')
   # ax_transformed.legend()

    plt.draw()
# Call update_plot function when g_qb slider value is changed
time_slider.on_changed(update_plot)
alpha_slider.on_changed(update_plot)
pump_freq_slider.on_changed(update_plot)
# Display the plot
plt.show()
