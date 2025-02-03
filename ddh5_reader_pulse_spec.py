import h5py
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # You can try 'Qt5Agg' or 'Agg' as well
import matplotlib.pyplot as plt
import pandas as pd
def extract_real_imaginary(complex_array):
    """
    Extracts the real and imaginary parts from a complex array.

    Parameters:
    complex_array (numpy.ndarray): Input array of complex numbers.

    Returns:
    tuple: Two numpy arrays, the first containing the real parts and the second containing the imaginary parts.
    """
    real_part = np.real(complex_array)
    imaginary_part = np.imag(complex_array)
    return real_part, imaginary_part

def read_and_plot_ddh5(filename):
    # Open the file
    with h5py.File(filename, 'r') as f:

        print(f['data'].keys())
        freq_data  = f['data']['sweepFreq'][()]   # shape: (N,)
        avg_iq   = f['data']['avg_iq_1ro_0'][()]    # shape: (M, N)
        msmt_data = f['data']['msmts'][()]  # shape: (M, N)
        reps = f['data']['reps'][()]
    # Create 2D coordinate meshes from the 1D frequency and current axes
    # (We assume freq_data is horizontal axis, curr_data is vertical axis)

    real_data, imaginary_data = extract_real_imaginary(avg_iq)




    unique_reps, counts = np.unique(reps, return_counts=True)
    max_count = max(counts)  # Find max occurrences per rep
    freq_data_org = freq_data[0:max_count]
    # Create a matrix to store values by occurrence index
    real_organized_data = np.full((len(unique_reps), max_count), np.nan)  # Fill with NaNs initially
    image_organized_data = np.full((len(unique_reps), max_count), np.nan)  # Fill with NaNs initially

    # Fill the matrix
    for i, rep in enumerate(unique_reps):
        real_rep_data = real_data[reps == rep]  # Get data for this rep
        image_rep_data = imaginary_data[reps == rep]
        real_organized_data[i, :len(real_rep_data)] = real_rep_data  # Assign values
        image_organized_data[i, :len(image_rep_data)] = image_rep_data
    # Compute mean, ignoring NaNs
    real_average_per_position = np.nanmean(real_organized_data, axis=0)
    image_average_per_position = np.nanmean(image_organized_data, axis=0)

    plt.figure()
    plt.plot(freq_data_org,real_average_per_position, label = 'real')
    plt.plot(freq_data_org,image_average_per_position, label = 'image')
    plt.xlabel('freq (MHz) with offset')
    plt.ylabel('A.U.')
    plt.show()
if __name__ == "__main__":
    filename = r"Z:\Data\Maser\20240112_cooldown_CR\20240125_Calibration_Throwdown\2024-01-25\\maser_1__QMagnet_-7.0_mA_SMagnet_0.2919_mA_pulseSpec_1000.ddh5"
    read_and_plot_ddh5(filename)

