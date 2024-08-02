import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.io
import os
import seaborn as sns
from scipy.fft import fft, fftfreq
from scipy import signal
from scipy.signal.windows import gaussian
from scipy.signal import stft, get_window
import shutil
import cv2


root_path = os.getcwd()

# Replace as appropriate
# dataset name with mirror equipped
mirror_data_name = 'S80_ID_MM_NT_S'
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


# video destination
os.makedirs('heatmap', exist_ok=True)


def create_video(data, filename):
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
        plt.imshow(np.abs(frame_data), aspect='auto', cmap='hsv',
                   vmin=np.abs(data).min(), vmax=np.abs(data).max())
        plt.colorbar()
        plt.xlabel('Range bin')
        plt.ylabel('Time index')
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


def create_animation():
    return None


def extract_frame(data, start_idx, end_idx=-1, start_bin=0):

    frame = data[start_idx:end_idx, start_bin:]

    return frame


# moving average background subtraction
def update_moving_average(moving_avg, frame, alpha):
    return (1 - alpha) * moving_avg + alpha * frame


def moving_average_subtraction(frame, alpha):

    moving_avg = frame.astype(np.float32)

    moving_avg = update_moving_average(moving_avg, frame, alpha)

    foreground = cv2.absdiff(frame, moving_avg)
    return foreground


def heatmap(data, cmap='viridis', aspect='auto', origin='lower',
            xlabel='', ylabel='', title=''):

    plt.imshow(data, aspect=aspect, cmap=cmap, origin=origin,
               vmin=data.min(), vmax=data.max())
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

    return None


# filter parameters
cutoff_frequency = 1
filter_order = 1
sampling_frequency = 500


# heatmap(np.abs(data_m_2)[2000:2600, 10:], title='1')


# iterating through both datasets with a high-pass filter
# for bin_no in range(0, data_m_1.shape[1]):
#   data_m_1[:, bin_no] = high_pass_filter(data_m_1, bin_no, filter_order,
#                                           cutoff_frequency)
#   data_m_2[:, bin_no] = high_pass_filter(data_m_2, bin_no, filter_order,
#                                           cutoff_frequency)

# for bin_no in range(0, data_no_m_1.shape[1]):
#   data_no_m_1[:, bin_no] = high_pass_filter(data_no_m_1, bin_no, filter_order,
#                                             cutoff_frequency)
#   data_no_m_2[:, bin_no] = high_pass_filter(data_no_m_2, bin_no, filter_order,
#                                             cutoff_frequency)


power = np.abs(data_m_2)
print(power.shape)

current_frame = extract_frame(power, 5600, 5900, start_bin=15)
# current_frame_2 = extract_frame(power, 27, 50, 0)

xlabel = 'Range bins'
ylabel = 'Time index'
title = 'Waterfall plot'

# without butterworth high pass filter
heatmap(current_frame, xlabel=xlabel,
        ylabel=ylabel, title=f'{title} 1')
# heatmap(current_frame_2, xlabel=xlabel,
#         ylabel=ylabel, title=f'{title} 2')


# with moving average background subtraction
alpha = 0.7
foreground = moving_average_subtraction(current_frame, alpha)
# foreground_2 = moving_average_subtraction(current_frame_2, alpha)

heatmap(foreground, xlabel='Range bins',
         ylabel='Time index', title='Waterfall plot 1 (background subtraction)')
# heatmap(foreground_2, xlabel='Range bins',
#          ylabel='Time index', title='Waterfall plot 2 (background subtraction)')
# shutil.rmtree('heatmap')  # clean up frames
