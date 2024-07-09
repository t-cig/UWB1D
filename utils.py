import numpy as np
import scipy.io as sio
import os
from constants import *
import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import chi


def probe_init(experiment_name: str, data_file: str, data_name: str):
    """Initialize the probing experiment and load the corresponding datafile. 
    @param experiment_name: name of the experiment currently running (the folder name where the data
    file is stored)
    @param data_file: name of the data_file
    @param data_name: the name of the data for probing frequencies in the data_file
    """
    data = sio.loadmat(os.path.join(PROBE_FOLDER, experiment_name, data_file))
    data = data.get(data_name)
    data_all = np.array(data)[:, 0]
    return data_all

def extract_probing_stamps(data_all, time_total):
    """
    Extract timestamps that are in the interval [0, time_total), and thinning them through a binomial 
    distribution with keep probability PROB_KEEP. 
    """
    np.random.seed(SEED_NUMBER)
    stamps = data_all[data_all < time_total]
    keep = np.random.binomial(1, PROB_KEEP, len(stamps)) 
    stamps = stamps[keep>0]
    return stamps

def compute_coefficients_sweep_torch(time_total, freq, stamps, device, verbose=True):
    """
    Sweeps across all the provided frequencies.
    @param time_total: exposure time in seconds
    @param freq: torch 1D tensor containing the probing frequencies.
    @param stamps: torch 1D tensor containing the timestamps.
    @param device: torch.device
    @param verbose: whether to display estimated time of completion or not
    @return:
        amplitudes: 1D tensor
        phases: 1D tensor
    """
    y = torch.zeros_like(freq.reshape(-1, 1), dtype=torch.cdouble, device=device)
    if verbose:
        for stp in tqdm.tqdm(stamps):
            tmp = 1/time_total * torch.exp(-torch.tensor(2j, device=device, dtype=torch.cdouble) * torch.pi * freq.reshape(-1, 1) * stp)
            y += tmp
        amplitudes = torch.abs(y) * 2
        phases = torch.atan2(torch.imag(y), torch.real(y))
    else:
        for stp in stamps:
            tmp = 1/time_total * torch.exp(-torch.tensor(2j, device=device) * torch.pi * freq.reshape(-1, 1) * stp)
            y += tmp
        amplitudes = torch.abs(y) * 2
        phases = torch.atan2(torch.imag(y), torch.real(y))
    amplitudes[0] /= 2
    return amplitudes, phases

def compute_bound_torch(frequencies, stamps, num_files, probe_time):
    """"
    Compute bound function when frequencies is a pytorch tensor.
    @param frequencies: pytorch 1D tensor of probed frequencies
    @param stamps: torch 1D tensor containing the timestamps.
    @param num_files: number of files of associated probe time
    @param probe_time: probe time, 0.1 or 1
    @return:
        bound_val: the estimated bound
    """
    if(probe_time == 0.1):
        num_freqs_thisfile = torch.sum(frequencies > 3e6).item()
        num_freqs_otherfiles = len(frequencies)
        num_freqs = (num_files - 1) * num_freqs_otherfiles + num_freqs_thisfile
        percentage_in = 1 - 1/num_freqs
        bound_val = compute_amplitude_bound(percentage_in, stamps.size/stamps[-1]/probe_time)
        print(f'Bound: {bound_val}')
    elif(probe_time == 1):
        num_freqs_smaller = torch.sum(frequencies <= 3e6).item()
        percentage_in = 1 - 1/num_freqs_smaller
        bound_val = compute_amplitude_bound(percentage_in, stamps.size/stamps[-1]/probe_time)
        print(f'Small bound: {bound_val}')
    return bound_val

def compute_bound_np(frequencies, stamps, num_files, probe_time):
    """"
    Compute bound function when frequencies is a numpy array. Used when data is loaded from .mat files.
    @param frequencies: numpy array of frequencies
    @param stamps: torch 1D tensor containing the timestamps.
    @param num_files: number of files of associated probe time
    @param probe_time: probe time, 0.1 or 1
    @return:
        bound_val: the estimated bound
    """
    if(probe_time == 0.1):
        num_freqs_thisfile = np.sum(frequencies > 3e6)
        num_freqs_otherfiles = len(frequencies)
        num_freqs = (num_files - 1) * num_freqs_otherfiles + num_freqs_thisfile
        percentage_in = 1 - 1/num_freqs
        bound_val = compute_amplitude_bound(percentage_in, stamps.size/stamps[-1]/probe_time)
        print(f'Bound: {bound_val}')
    elif(probe_time == 1):
        num_freqs_smaller = np.sum(frequencies <= 3e6)
        percentage_in = 1 - 1/num_freqs_smaller
        bound_val = compute_amplitude_bound(percentage_in, stamps.size/stamps[-1]/probe_time)
        print(f'Small bound: {bound_val}')
    return bound_val

def compute_amplitude_bound(percentage_in, counts_per_sec):
    """
    Computes the bound value based on the chi distribution of the estimated amplitudes.
    @param percentage_in:
    @param counts_per_sec: Normalised counts per second, e.g  timestamps.size / timestamps[-1] / exposure
    @return:
        bound_val: the estimated bound
    """

    alpha_1 = chi.ppf(percentage_in, 2)
    bound_val = alpha_1 * np.sqrt(counts_per_sec * 2)
    return bound_val

def reconstruction_init(data_file: str):
    """
    Given the name of the reconstruction data_file, load the frequencies, amplitudes and phase shifts 
    data in the matlab file, and convert them to torch tensors. 
    @param: data_file: name of the reconstruction data file. 
    """
    mdic = sio.loadmat(os.path.join(RECONSTRUCTION_FOLDER, data_file))
    freqs_np = mdic[FREQS_NAME].flatten()
    amps_np = mdic[AMPS_NAME].flatten()
    phs_np = mdic[PHS_NAME].flatten()
    amps_torch = torch.tensor(amps_np, device=device)
    frequencies_torch = torch.tensor(freqs_np, device=device)
    phs_torch = torch.tensor(phs_np, device=device)
    return amps_torch, frequencies_torch, phs_torch

def reconstruction_time(exposure: float, laser_40MHz: bool):
    """
    Given the exposure time (in seconds) and whether the device is 40Hz laser, calculate the unit for
    time, scale of time, start_time, end_time (so that 0.5s is the midpoint of time), and then generate
    NUM_SAMPLES samples uniformly in the range [start_time, end_time]. 

    @param: exposure: exposure time (in seconds)
    @param: laser_40Hz: whether the current device is 40Hz laser. 
    """
    # Set units for time and timescale.
    if exposure >= 1:
        unit_time = 's'
        time_scale = 1
    elif exposure >= 1e-3:
        unit_time = 'ms'
        time_scale = 1e3
    elif exposure >= 1e-6:
        unit_time = 'us'
        time_scale = 1e6
    elif exposure >= 1e-9:
        unit_time = 'ns'
        time_scale = 1e9
    elif exposure >= 1e-12:
        unit_time = 'ps'
        time_scale = 1e12
    
    # If the device is laser_40Hz, center time by LASER_EXP, instead of 0.5
    if laser_40MHz:
        start_time = LASER_EXP - exposure/2
        end_time = LASER_EXP + exposure/2
    else:
        start_time = 0.5 - exposure/2
        end_time = 0.5 + exposure/2
    times = torch.linspace(start_time, end_time, NUM_SAMPLES, device=device)
    return unit_time, time_scale, start_time, end_time, times

def reconstruct_rate_function_torch(times: torch.tensor, max_freq: float, frequencies: torch.tensor, amplitudes: torch.tensor, phases: torch.tensor, verbose = True):
    """
    Reconstructs the 1D rate function of a non-homohegeneous Poisson (NHPP) distribution.
    @param ts: 1D tensor of the time values
    @param frequencies: 1D tensor of frequencies used for the reconstruction
    @param amplitudes: 1D tensor of the corresponding amplitudes of the frequencies
    @param phases: 1D tensor of the corresponding phases
    @param verbose: whether to show progression
    @return:
        rate: 1D tensor of the NHPP rate
    """

    rate = torch.zeros_like(times, device=device)
    torch.pi = torch.tensor(math.pi, device = device, dtype=torch.float64)
    
    if verbose:
        for freq_id in tqdm.tqdm(range(frequencies.shape[0] - 1)):
            if frequencies[freq_id+1] > max_freq:
                continue
            rate += amplitudes[freq_id + 1] * torch.cos(2 * torch.pi * frequencies[freq_id + 1] * times + phases[freq_id + 1])
    else:
        for freq_id in (range(frequencies.shape[0] - 1)):
            if frequencies[freq_id+1] > max_freq:
                continue
            rate += amplitudes[freq_id + 1] * torch.cos(2 * torch.pi * frequencies[freq_id + 1] * times + phases[freq_id + 1])
    rate += amplitudes[0]
    rate[rate < 0] = 0
    return rate

def create_plots(times_np: np.array, unit_time: str, time_scale: int, rate_np: np.array, exposure: float, start_time:float, end_time:float, name:str):
    """
    Given the timestamps, time_unit, time_scale, reconstructed rate function, exposure time, start and
    end times and name of the device, create the corresponding figure for reconstructed rate_function.
    @param times_np: numpy array of timestamps
    @param unit_time: current unit of time
    @param time_scale: current scale of time

    """
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    plt.plot(times_np * time_scale, rate_np, color = INTEGRAL_COLOR, linewidth=RATE_LINEWIDTH)

    ax.spines['bottom'].set_linewidth(AXES_WIDTH)
    ax.spines['left'].set_linewidth(AXES_WIDTH)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(labelleft=False, labelbottom=True)
    plt.xlim([times_np.min()*time_scale, times_np.max()*time_scale])
    plt.xlabel(f'Time [{unit_time}]')
    
    ticks_val = np.linspace(0, exposure*time_scale, 6)
    ticks = np.linspace(start_time*time_scale, end_time*time_scale, 6)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticker.FormatStrFormatter('%.2f').format_ticks(ticks_val))
    plt.tick_params(left = False)
    plt.savefig(os.path.join(RECONSTRUCTION_FOLDER, f'rate_recon_{name}_figure.pdf') ,dpi=300,bbox_inches='tight', transparent = True)
