import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # You can try 'Qt5Agg' or 'Agg' as well
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from System_setup_class import *
from tqdm import tqdm
from Maser_function import *

number_of_eigenstate = 5
q = np.linspace(number_of_eigenstate,50,41)

eigenvalues = []

for i in q:
    i = int(i)


    dimensions = (i,)
    operators = QuantumOperators(*dimensions)

    a_q = operators.annihilation(0)
    a_q_dagger = a_q.dag()

    omega_q = 7 * two_pi * GHz
    alpha = 160 * two_pi * MHz
    H = omega_q * a_q_dagger * a_q - (alpha/12) * (a_q + a_q_dagger)**4



    eigen_energies, eigen_states = H.eigenstates()
    eigenvalues.append(eigen_energies[:number_of_eigenstate]/two_pi)
eigenvalues = np.array(eigenvalues)
plt.figure()
for i in range(number_of_eigenstate):

    plt.plot(q,eigenvalues[:,i],label = f'eigenstate{i} ')
    plt.xlabel('number of trucated state')
    plt.ylabel('frequency (MHz)')
    #plt.legend()