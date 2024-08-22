import subprocess
import argparse
# More imports down below at if __name__ == '__main__'

def uwb1d(probe_args, reconstruction_args, PROBED=False):
    ## Probing pipeline
    if not PROBED:
        probe_times = probe_args['times']
        max_sizes = probe_args['max_sizes']
        intervals = probe_args['intervals']
        stepsizes = probe_args['stepsizes'] 
        assert len(probe_times) == len(max_sizes) == len(intervals) == len(stepsizes), "There should be one max_size, interval, and stepsize corresponding to each probing timescale. Please make sure all lists in probe_args have the same length. "   
        probe(probe_times, max_sizes, intervals, stepsizes, STORE_PROBE)
    else:
        intervals = probe_args['intervals']
        detect_frequency(intervals[0])
    print(BLUE + "Probing and frequency thresholding finished. Starting reconstruction.." + ENDC)

    ## Reconstruction pipeline
    exposure_times = np.array(reconstruction_args['exposure_times'])
    device_names = np.array(reconstruction_args['device_names'])
    assert len(exposure_times) == len(device_names), "There should be one device corresponding to each exposure_time. Please make sure the lists in reconstruction_args have the same length."
    reconstruction(exposure_times, device_names)
    print(PINK + "Reconstruction Finished. Please go to " + RECONSTRUCTION_FOLDER + " to see the reconstructed flux for each device. " + ENDC)

if __name__ == '__main__':
    ## Parse user argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--conda', action='store_true', help='Enable Conda flag')
    parser.add_argument('--probed', action='store_true', help='Skip Probing Step')
    parser.add_argument('--probestore', action='store_true', help='Skip Probing Storage')
    args = parser.parse_args()

    ## Whether or not to create a brand new anaconda environment for running the pipeline. Set CONDA=True 
    # to enable automatic creation of conda environment, activating it and installing all requirements.
    CONDA = args.conda
    # Whether or not the probe files are already created, and we only need reconstruction. Setting
    # PROBE=True will not run the probing part of the pipeline.
    PROBED = args.probed
    
    ## Whether or not to create and store intermediate probing files during the pipeline. Setting 
    # STORE_PROBE=False will significantly reduce the memory cost of the pipeline, but when 
    # the pipeline is interupted, it is harder to recover. 
    if args.probestore:
        STORE_PROBE = True
    else:
        STORE_PROBE = False
    
    if CONDA:
    ## Create conda environment named UWB, activate the environment, and install all requirements for the pipeline
        subprocess.run(['chmod', 'a+x', 'init.bash'])
        subprocess.run(['./init.bash', str(PROBED), str(STORE_PROBE)])
    else:
        subprocess.run(['pip3', 'install', '-r', 'requirements.txt'])
        print("All requirements satisfied.")
        ## After all required packages installed, the imports
        from probe import probe
        from detect_frequencies import detect_frequency
        from reconstruction import reconstruction
        import numpy as np
        from constants import *
        
        probe_args = {'times': [0.1, 1], 
                    'max_sizes': [400e6, 40e6], 
                    'intervals': [25, 1], 
                    'stepsizes': [6, 1]}
        reconstruction_args = {'exposure_times': [1, 5/900, 3/3e6, 1/360e6],
                            'device_names': ['anybeam', '900Hzbulb', '3MHzLaser', '40MHzLaser']}

        uwb1d(probe_args, reconstruction_args, PROBED)