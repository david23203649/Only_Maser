import h5py
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # You can try 'Qt5Agg' or 'Agg' as well
import matplotlib.pyplot as plt

def read_and_plot_ddh5(filename):
    # Open the file
    with h5py.File(filename, 'r') as f:
        print(f['data'].keys())
        curr_data  = f['data']['current'][()]   # shape: (M,)
        freq_data  = f['data']['frequency'][()]   # shape: (N,)
        mag_data   = f['data']['power'][()]    # shape: (M, N)
        phase_data = f['data']['phase'][()]  # shape: (M, N)

    # Create 2D coordinate meshes from the 1D frequency and current axes
    # (We assume freq_data is horizontal axis, curr_data is vertical axis)

    CURR = curr_data.reshape(201, 1601)
    FREQ = freq_data.reshape(201, 1601)
    mag_data = mag_data.reshape(201, 1601)
    phase_data = phase_data.reshape(201,1601)
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(FREQ, CURR, mag_data, shading='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')

    # Set axis labels
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Current (mA)')
    plt.title('2D Color Plot: Magnitude')
    plt.gca().set_aspect('equal', adjustable='box')  # 'box' ensures it fits the figure size

    # Create a figure with two subplots side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # --- First subplot: Magnitude as a 2D color map ---
    c1 = axs[0].pcolormesh(CURR,FREQ, mag_data, shading='auto', cmap='viridis')
    axs[0].set_title('Magnitude')
    axs[0].set_xlabel('Current')
    axs[0].set_ylabel('Frequency')
    fig.colorbar(c1, ax=axs[0], label='Magnitude')

    # --- Second subplot: Phase as a 2D color map ---
    c2 = axs[1].pcolormesh(CURR,FREQ, phase_data, shading='auto', cmap='cividis')
    axs[1].set_title('Phase')
    axs[1].set_xlabel('Current')
    axs[1].set_ylabel('Frequency')
    fig.colorbar(c2, ax=axs[1], label='Phase')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    filename = r"Z:\Data\Maser\To NIck\qubit_msmt\20240708_cooldown\\maser_1__freq_1300_1800_bias_-0.065_-0.05.ddh5"
    read_and_plot_ddh5(filename)

