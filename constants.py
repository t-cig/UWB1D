import torch
import math
import matplotlib

# Set device (default cuda) to run this project, and set default tensor type to float64.
device = torch.device('cuda:0')
torch.set_default_dtype(torch.float64)

## Constants to be used in the pipeline
# USE_PAPER_PI is set to True to initialize constant torch.pi with type float32 (as consistent with our paper), 
# however, if would like to use type float64 for torch.pi instead, change USE_PAPER_PI to False. 
# Note that changing to float64 will produce slight variations in the created probing files, 
# and consequently others.
USE_PAPAR_PI = True

if USE_PAPAR_PI:
    torch.pi = torch.tensor(math.pi, device=device, dtype=torch.float32)
else:
    torch.pi = torch.tensor(math.pi, device=device)

## Set seed and probablities used for thinning timestamps
SEED_NUMBER = 0
PROB_KEEP = 0.1

##################### Change Below if Needed #########################################################
## Directory names. 
PROBE_FOLDER = ''
RECONSTRUCTION_FOLDER = 'figures'

## Experiment setup, probing and reconstruction datafiles. The PROBE_FILE needs to be contained under 
## PROBE_FOLDER/EXPERIMENT. 
## The RECONSTRUCTION_FILE will be contained under RECONSTRUCTION_FOLDER
EXPERIMENT = 'experiment1'
PROBE_FILE = 'scan_posX001_posY001.mat'
## Key to find desired data
DATA_NAME = 'data_2'
RECONSTRUCTION_FILE = 'swept_freqs_ab.mat'
# Freqs, amps, phs data variables (keys) that will be stored in RECONSTRUCTION_FILE
FREQS_NAME = 'freqs'
AMPS_NAME = 'amps'
PHS_NAME = 'phs'
###################################################################################################

## Probing constants (only needs to be changed if executing the pipeline without the probing part)
# Stepsize for lower timescale (in our case, 0.1s)
LOW_STEPSIZE = 6
# Stepsize for higher timescale (in our case, 1s)
HIGH_STEPSIZE = 1
NUM_FILES_TOTAL = 26
# Frequency for lower timescale (in our case, 0.1s) in MHz
FREQ_LOW = 400.0
# Frequency for higher timescale (in our case, 0.1s) in MHz
FREQ_HIGH = 40.0

## Plotting constants
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
INTEGRAL_COLOR = 'b'
RATE_LINEWIDTH = 1
AXES_WIDTH = 1
FIG_SIZE = (16, 4)

# Number of reconstruction samples
NUM_SAMPLES = 10000

# Exposure time for 40MHz laser
LASER_EXP = 0.500983

# Other constants that can be set for plots
# SHIFT_AMOUNT = 0
# RATE_COLOR = 'black'
# COUNTS_COLOR = 'r'

# NUM_EXPERIMENTS = 50
# COUNTING_OPACITY = 0.3
# MARKER_SIZE = 10
# FILE_FORMAT = '.pdf'
# X_SHIFT = 0.5

## Constants for output in terminal
# Define color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
PINK = '\033[95m'
ENDC = '\033[0m'  # Reset color
