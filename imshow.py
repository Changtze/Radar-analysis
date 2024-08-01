import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
from scipy.fft import fft, fftfreq
from scipy import signal
from scipy.signal.windows import gaussian
from scipy.signal import stft, get_window
import shutil
import cv2


root_path = os.getcwd()

# Replace as appropriate
mirror_data_name = 'S100_ID_SEATED_FIXED_DISTANCE'  # dataset name with mirror equipped
control_data_name = 'BED_NO_MIRROR'  # control dataset with no mirror equipped

# dataset file paths
mirror_path = root_path + os.path.join(f'\\{mirror_data_name}.mat')
control_path = root_path + os.path.join(f'\\{control_data_name}.mat')




# .MAT files are loaded in dict form
def load_data(filepath):
    """
    Parameters
    ----------
    filepath : location of data.

    Returns :
    -------
    A tuple of the data from both radars.
    
    """
    radar = scipy.io.loadmat(filepath)
    radar_1 = radar['Data_Matrix_1']
    radar_2 = radar['Data_Matrix_2']

    return radar_1, radar_2


# Loading data
data_m_1, data_m_2 = load_data(mirror_path)
data_no_m_1, data_no_m_2 = load_data(control_path)


# High pass filter function
def high_pass_filter(sig, bin_number, order=4, freq_c=5, fs=500):
    """
    Parameters
    ----------
    sig : numpy array to be filtered
    bin_number : range bin indicator
    order : Butterworth filter order
    freq_c : critical frequency for filter
    fs : signal sampling frequency
    

    Returns :
    -------
    An array passed through a high-pass filter
    
    """

    hpf = signal.butter(order, freq_c, btype='highpass', output='sos', fs=fs)
    data = sig[:, bin_number]  # x-axis represents range, y-axis is time
    data_hpf = signal.sosfilt(hpf, data)

    return data_hpf


# filter parameters
cutoff_frequency = 5
filter_order = 4
sampling_frequency = 500


# iterating through both datasets with a high-pass filter
for bin_no in range(0, data_m_1.shape[1]):
    data_m_1[:, bin_no] = high_pass_filter(data_m_1, bin_no, filter_order,
                                           cutoff_frequency)

for bin_no in range(0, data_no_m_1.shape[1]):
    data_no_m_1[:, bin_no] = high_pass_filter(data_no_m_1, bin_no, filter_order,
                                              cutoff_frequency)

# video destination
os.makedirs('heatmap', exist_ok=True)


def create_animation(data, filename):
    """

    Parameters
    ----------
    data : 2x2 matrix for heatmaps to be created from

    Returns
    -------
    None.

    """

    # number of frames, y-axis is number of range bins
    num_frames = data.shape[0] // data.shape[1]

    for frame in range(num_frames):
        frame_data = data[frame*data.shape[1]:(frame+1)*data.shape[1], :]

        # Heatmap creation
        plt.imshow(np.abs(frame_data), aspect='auto', cmap='hsv', vmin=np.abs(data).min(), vmax=np.abs(data).max())
        plt.colorbar()
        plt.xlabel('Timestamp')
        plt.ylabel('Range bin number')
        plt.title(f'Frame {frame}')
 
        # Save plot
        plt.savefig(f'heatmap/frame_{frame}.png')
        plt.close()

    # Video creation
    frame_rate = 4
    
    frame = cv2.imread('heatmap/frame_0.png')
    height, width, layers = frame.shape
    size = (width, height)

    out = cv2.VideoWriter(
        f'{filename}.avi', cv2.VideoWriter_fourcc(*'DIVX'), frame_rate, size)
    
    for i in range(num_frames):
        filename = f'heatmap/frame_{i}.png'
        img = cv2.imread(filename)
        out.write(img)
        
    out.release()
    
    return None

# radar 1
create_animation(data_m_1, f'{mirror_data_name}_radar_1_vid')
#create_animation(data_no_m_1, f'{control_data_name}_radar_1_vid')

# radar 2
create_animation(data_m_2, f'{mirror_data_name}_radar_2_vid')
#create_animation(data_no_m_2, f'{control_data_name}_radar_2_vid')

shutil.rmtree('heatmap')