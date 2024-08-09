import numpy as np
import scipy.io as sio
import tqdm
import os
from utils import *
from constants import *


def detect_frequency(num_files: int):
    """
    Given noisy data, filter out unwanted noise based on equation 7
    @param num_files: Number of files to filter

    """
    stampsAll = probe_init(EXPERIMENT, PROBE_FILE, DATA_NAME)

    time_total = 0.1
    time_smaller = 1

    amps = []
    phs = []
    stamps = extract_probing_stamps(stampsAll, time_total)
    stamps_smaller = extract_probing_stamps(stampsAll, time_smaller)
    print(f'Using {len(stamps)} photons.')

    probed_lims = np.arange(2, NUM_FILES_TOTAL, 1) * 400e6
    
    maxFreq = 3e6

    #Load data from first file of 0.1 second probe file.
    filename = f'torch_{time_total}sec_{FREQ_LOW}MHz_{LOW_STEPSIZE}.mat'
    params = sio.loadmat(os.path.join(PROBE_FOLDER, EXPERIMENT, filename))
    frequencies = np.array(params['freqs'].flatten())
    amps = np.array(params['amps']).flatten()
    phs = np.array(params['phs'].flatten())

    #Load data from only file of 1 second probe file.
    filename_smaller = f'torch_{time_smaller}sec_{FREQ_HIGH}MHz_{HIGH_STEPSIZE}.mat'
    params_smaller = sio.loadmat(os.path.join(PROBE_FOLDER, EXPERIMENT, filename_smaller))
    frequencies_smaller = np.array(params_smaller['freqs']).flatten()
    amps_smaller = np.array(params_smaller['amps']).flatten()
    phs_smaller = np.array(params_smaller['phs'].flatten())

    #Compute bounds to cut off noisy frequencies.
    bound_val = compute_bound_np(frequencies, stamps, num_files, time_total)
    bound_val_smaller = compute_bound_np(frequencies_smaller, stamps_smaller, 1, time_smaller)

    #Init arrays to append data to.
    freqs_ab = []
    amps_ab = []
    phs_ab = []

    #Add low counts:
    freqs_ab.append(frequencies_smaller[(amps_smaller>bound_val_smaller) * (frequencies_smaller <= maxFreq)])
    amps_ab.append(amps_smaller[(amps_smaller>bound_val_smaller) * (frequencies_smaller <= maxFreq)])
    phs_ab.append(phs_smaller[(amps_smaller>bound_val_smaller) * (frequencies_smaller <= maxFreq)])
    
    #Add high counts:
    freqs_ab.append(frequencies[(amps>bound_val) * (frequencies > maxFreq)])
    amps_ab.append(amps[(amps>bound_val) * (frequencies > maxFreq)])
    phs_ab.append(phs[(amps>bound_val) * (frequencies > maxFreq)])

    #Add the rest of the counts
    for fNo in tqdm.tqdm(range(len(probed_lims))):
        f_prob = probed_lims[fNo]

        filename = f'torch_{time_total}sec_{f_prob/1e6}MHz_{LOW_STEPSIZE}.mat'
        params = sio.loadmat(os.path.join(PROBE_FOLDER, EXPERIMENT, filename))
        amps = np.array(params['amps'].flatten())
        phs = np.array(params['phs'].flatten())
        frequencies = np.array(params['freqs'][0])
        freqs_ab.append(frequencies[amps>bound_val])
        amps_ab.append(amps[amps>bound_val])
        phs_ab.append(phs[amps > bound_val])

    #Join all the data into one numpy array
    freqs_np = np.hstack(freqs_ab)
    amps_np = np.hstack(amps_ab)
    phs_np = np.hstack(phs_ab)

    mdic = {'freqs' : freqs_np, 'amps': amps_np, 'phs' : phs_np}
    sio.savemat(os.path.join(RECONSTRUCTION_FOLDER, 'swept_freqs_ab.mat'), mdic)

    ## We will also store the frequencies for the entire 0.1s probe only
    new_probed_lims = np.arange(1, NUM_FILES_TOTAL, 1) * 400e6
    #Init arrays to append data to.
    freqs_ab_new = []
    amps_ab_new = []
    phs_ab_new = []
    for fNo in tqdm.tqdm(range(len(new_probed_lims))):
        f_prob = new_probed_lims[fNo]
        filename = f'torch_{time_total}sec_{f_prob/1e6}MHz_{LOW_STEPSIZE}.mat'
        params = sio.loadmat(os.path.join(PROBE_FOLDER, EXPERIMENT, filename))
        amps = np.array(params['amps'].flatten())
        phs = np.array(params['phs'].flatten())
        frequencies = np.array(params['freqs'][0])
        freqs_ab_new.append(frequencies[amps>bound_val])
        amps_ab_new.append(amps[amps>bound_val])
        phs_ab_new.append(phs[amps > bound_val])
    #Join all the data into one numpy array
    freqs_np_new = np.hstack(freqs_ab_new)
    amps_np_new = np.hstack(amps_ab_new)
    phs_np_new = np.hstack(phs_ab_new)

    mdic_new = {'freqs' : freqs_np_new, 'amps': amps_np_new, 'phs' : phs_np_new}
    sio.savemat(os.path.join(RECONSTRUCTION_FOLDER, 'swept_freqs_ab_0.1s.mat'), mdic_new)
if __name__ == '__main__':
    detect_frequency(25)