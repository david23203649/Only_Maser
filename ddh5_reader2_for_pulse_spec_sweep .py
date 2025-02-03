import h5py
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def extract_real_imaginary(complex_array):
    """Extracts the real and imaginary parts from a complex array."""
    real_part = np.real(complex_array)
    imaginary_part = np.imag(complex_array)
    return real_part, imaginary_part


def read_and_plot_ddh5(filename):
    with h5py.File(filename, 'r') as f:
        print("Available datasets:", f['data'].keys())

        # Sweeping parameters
        freq_data = f['data']['freq'][()]  # shape: (N,)
        avg_iq = f['data']['avg_iq_ro_0'][()]  # shape: (M, N)
        msmt_data = f['data']['msmts'][()]  # shape: (M, N)
        reps = f['data']['reps'][()]
        bias_data = f['data']['bias'][()]

    real_data, imaginary_data = extract_real_imaginary(avg_iq)
    bias_unique_values, bias_counts = np.unique(bias_data, return_counts=True)
    freq_data_bias_unique_values, freq_data_counts = np.unique(freq_data, return_counts=True)
    freq_data_reshape = np.reshape(freq_data, [ freq_data_counts[0], bias_counts[0]])
    bias_data_reshape = np.reshape(bias_data, [freq_data_counts[0], bias_counts[0]])
    real_data_reshape = np.reshape(real_data, [freq_data_counts[0], bias_counts[0]])
    imag_data_reshape = np.reshape(imaginary_data, [freq_data_counts[0], bias_counts[0]])


    # Plotting the results
    plt.figure(figsize=(8, 6))
    plt.contourf(freq_data_reshape, bias_data_reshape, real_data_reshape, levels=50, cmap='RdBu_r')
    plt.colorbar(label="Real Part of IQ")
    plt.xlabel("Frequency")
    plt.ylabel("Bias")
    plt.title("2D Sweep - Real Part")
    plt.grid(True)

    plt.figure(figsize=(8, 6))
    plt.contourf(freq_data_reshape, bias_data_reshape, imag_data_reshape, levels=50, cmap='RdBu_r')
    plt.colorbar(label="Imaginary Part of IQ")
    plt.xlabel("Frequency")
    plt.ylabel("Bias")
    plt.title("2D Sweep - Imaginary Part")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    filename = r"Z:\Data\Maser\To NIck\qubit_msmt\20240708_cooldown\\maser_1__freq_1300_1800_bias_-0.065_-0.05.ddh5"
    read_and_plot_ddh5(filename)
