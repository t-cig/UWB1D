from typing import List
import numpy as np
import scipy.io as sio
import os
import copy
import torch
from constants import *
from utils import *

def probe(probe_times: List, max_sizes: List, intervals: List, stepsizes: List, STORE_PROBE=True):
    """
    Given a list of probing timescales, sizes of frequency for each plot, number of interval (plots) 
    and stepsizes of probing, generate matlab files of amplitudes and phases for each probed frequency.

    @param probe_times: list of probing timescales (exposure times)
    @param max_sizes: list of maximum sizes of the frequency intervals for each timescale
    @param intervals: list of frequency intervals to probe for each timescale
    @param stepsizes: list of stepsizes for probing each timescale

    Note: len(probe_times) = len(max_sizes) = len(intervals) = len(stepsizes)
    """

    # Initialize variables to be used
    data_all = probe_init(EXPERIMENT, PROBE_FILE, DATA_NAME)
    bound_val = 0
    freqs_ab = []
    amps_ab = []
    phs_ab = []

    with torch.no_grad():
        for i in range(len(probe_times)):
            print(f'{PINK}Probing for {probe_times[i]} seconds{ENDC}')
            # Extract timestamps that are in the interval [0, time_total), and thinning them through a 
            # binomial distribution with keep probability PROB_KEEP. 
            stamps = extract_probing_stamps(data_all, probe_times[i])
            stamps_in = torch.tensor(stamps, device=device)
            time_total = probe_times[i]
            # For each frequency interval of maxSize, divide according to stepsize, and collect all probing frequencies.
            for curMax in np.arange(0, intervals[i], 1):
                maxFreq = (curMax+1)*max_sizes[i]
                stepsize = stepsizes[i]
                if curMax == 0:
                    minFreq = 0
                else:
                    minFreq = curMax*max_sizes[i] + stepsize

                frequencies = torch.arange(minFreq, maxFreq+stepsize, stepsize, device=device)

                #Compute lower bounds for both probe times, only on first iteration of each probe time
                if(curMax == 0):
                    bound_val = compute_bound_torch(frequencies, stamps, intervals[i], probe_times[i])
                
                print(f'{YELLOW}Min Freq: {torch.min(frequencies)}, Max Freq: {torch.max(frequencies)}{ENDC}')
                # Filename to store the probing files
                filename = f'torch_{time_total}sec_{maxFreq/1e6}MHz_{stepsize}.mat'
                # If the file exists, we get the amplitudes and frequencies directly from the file.
                if os.path.isfile(os.path.join(PROBE_FOLDER, EXPERIMENT, f'torch_{time_total}sec_{maxFreq/1e6}MHz_{stepsize}.mat')): 
                    print("Target file " + f'torch_{time_total}sec_{maxFreq/1e6}MHz_{stepsize}.mat' + " exists. As a cautionary measure, skip replacing the file.")
                    params = sio.loadmat(os.path.join(PROBE_FOLDER, EXPERIMENT, filename))
                    amps2 = np.array(params['amps'].flatten())
                    phs2 = np.array(params['phs'].flatten())
                    freqs2 = np.array(params['freqs'].flatten())
                else:
                    ## Sweep thru all probing frequencies, and compute reconstructed amplitudes and phases.
                    amps1, phs1 = compute_coefficients_sweep_torch(time_total, frequencies, stamps_in, device, verbose=True)
                    amps2 = copy.deepcopy(amps1.cpu().numpy()).flatten()
                    phs2 = copy.deepcopy(phs1.cpu().numpy()).flatten()
                    freqs2 = copy.deepcopy(frequencies.cpu().numpy()).flatten()

                if(curMax == 0 and probe_times[i] == 0.1):
                    freqs_ab.append(freqs2[(amps2 > bound_val) * (freqs2 > 3e6)])
                    amps_ab.append(amps2[(amps2 > bound_val) * (freqs2 > 3e6)])
                    phs_ab.append(phs2[(amps2 > bound_val) * (freqs2 > 3e6)])
                elif(curMax == 0 and probe_times[i] == 1):
                    freqs_ab.append(freqs2[(amps2 > bound_val) * (freqs2 <= 3e6)])
                    amps_ab.append(amps2[(amps2 > bound_val) * (freqs2 <= 3e6)])
                    phs_ab.append(phs2[(amps2 > bound_val) * (freqs2 <= 3e6)])
                else:
                    freqs_ab.append(freqs2[amps2 > bound_val])
                    amps_ab.append(amps2[amps2 > bound_val])
                    phs_ab.append(phs2[amps2 > bound_val])
                
                # If STORE_PROBE=True (storing probing files), copy the amplitudes and phases, and save them in a matlab file. 
                if STORE_PROBE:
                    mdic = {AMPS_NAME:amps2, FREQS_NAME: frequencies.cpu().numpy(), PHS_NAME:phs2}
                    sio.savemat(os.path.join(PROBE_FOLDER, EXPERIMENT, filename), mdic)

                torch.cuda.empty_cache()

    freqs_np = np.hstack(freqs_ab)
    amps_np = np.hstack(amps_ab)
    phs_np = np.hstack(phs_ab)

    p = freqs_np.argsort()

    mdic = {'freqs' : freqs_np[p], 'amps': amps_np[p], 'phs' : phs_np[p]}
    os.makedirs(RECONSTRUCTION_FOLDER, exist_ok=True)
    sio.savemat(os.path.join(RECONSTRUCTION_FOLDER, 'swept_freqs_ab.mat'), mdic)
    

if __name__ == '__main__':
    ## Probing times, maxSize for each one, number of intervals (files) to be created for each one, and stepsize for each one
    probe_times = [1, 0.1]
    max_sizes = [40e6, 400e6]
    intervals = [1, 25]
    stepsizes = [1, 6]

    ## Create probing files using main function probe above
    probe(probe_times, max_sizes, intervals, stepsizes)