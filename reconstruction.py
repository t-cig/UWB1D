import numpy as np
from constants import *
from utils import *

# Main function used for reconstructing frequencies
def reconstruction(exposures, names):
    """
    Calculate and create reconstructed rate figures.
    @param exposures: list of different exposure times corresponding to different devices
    @param names: list of names for different devices
    @amps_torch: torch tensor of amplitudes
    @frequencies_torch: torch tensor of frequencies
    @phs_torch: torch tensor of phase_shifts
    """
    name_index = 0
    # Load the MatLab data file, extract frequencies, amplitudes and phase shifts data, then change them
    # into torch tensor formats. 
    for i in range(len(exposures)):
        if i == 3:
            amps_torch, frequencies_torch, phs_torch = reconstruction_init(RECONSTRUCTION_FILE2)
        else:
            amps_torch, frequencies_torch, phs_torch = reconstruction_init(RECONSTRUCTION_FILE)
        
        # Get everything need for time, including time unit, timescale, start_time, end_time and the
        # uniformly sampled NUM_SAMPLES number of timestamps. 
        unit_time, time_scale, start_time, end_time, times = reconstruction_time(exposures[i], i==len(exposures)-1)
        # Calculate the maximum frequency based on the exposure time, and the current device.
        if name_index == 1:
            max_freq = 100e3
        else:
            max_freq = 0.5 / exposures[i] * NUM_SAMPLES
        print(f'{YELLOW}Device name: {names[name_index]}. Maximum frequency: {max_freq}{ENDC}')

        # Reconstruct the rate function based on function reconstruct_rate_function_torch
        rate_fn = reconstruct_rate_function_torch(times, max_freq, frequencies_torch, amps_torch, phs_torch)
        
        # Convert back the timestamps, and rates function into numpy formats, for plotting
        times_np = times.cpu().numpy()
        rate_np = rate_fn.cpu().numpy()

        # Create plots for frequency reconstruction of the current device
        create_plots(times_np, unit_time, time_scale, rate_np, exposures[i], start_time, end_time, names[name_index])
        name_index += 1
    
    # Esimate FWHM (full width half maximum) of the rate_function near its maximum value.
    fwhm_estimate = np.ptp(times_np[np.where(rate_np>=rate_np.max()/2)])
    print(f'Estimated FWHM: {fwhm_estimate * 1e12} ps')

if __name__ == '__main__':
    # Exposure time and corresponding device names. 
    exposures = np.array([1, 5/900, 3/3e6, 1/360e6])
    names = np.array(['anybeam', '900Hzbulb', '3MHzLaser', '40MHzLaser'])
    reconstruction(exposures, names)
