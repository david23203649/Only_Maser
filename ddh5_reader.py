import h5py
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # You can try 'Qt5Agg' or 'Agg' as well
import matplotlib.pyplot as plt

def read_and_plot_ddh5(filename):
    # Open the file
    with h5py.File(filename, 'r') as f:
        print(f)
        curr_data  = f['data']['current'][()]   # shape: (M,)
        freq_data  = f['data']['frequency'][()]   # shape: (N,)
        mag_data   = f['data']['power'][()]    # shape: (M, N)
        phase_data = f['data']['phase'][()]  # shape: (M, N)

    # Create 2D coordinate meshes from the 1D frequency and current axes
    # (We assume freq_data is horizontal axis, curr_data is vertical axis)
    FREQ, CURR = np.meshgrid(freq_data, curr_data)

    # Create a figure with two subplots side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # --- First subplot: Magnitude as a 2D color map ---
    c1 = axs[0].pcolormesh(CURR,FREQ, mag_data, shading='auto', cmap='viridis')
    axs[0].set_title('Magnitude')
    axs[0].set_xlabel('Frequency')
    axs[0].set_ylabel('Current')
    fig.colorbar(c1, ax=axs[0], label='Magnitude')

    # --- Second subplot: Phase as a 2D color map ---
    c2 = axs[1].pcolormesh(CURR,FREQ, phase_data, shading='auto', cmap='cividis')
    axs[1].set_title('Phase')
    axs[1].set_xlabel('Frequency')
    axs[1].set_ylabel('Current')
    fig.colorbar(c2, ax=axs[1], label='Phase')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    filename = r"Z:\Data\Maser\20240210_cooldown_CR\VNA_Msmt\SNAIL\flux_sweep\\2024-02-13_0001_sweepSNAILmagnet_SNAIL_-5mA_5mA_-45dBmVNA_15avg_3GHzspan_3kHzIFBW_Qmagnet0mA_TWPA7.95GHz_5.0dBm.ddh5"
    read_and_plot_ddh5(filename)

