#!/bin/bash
NAME="UWB"
CONDA_DIR=~/.conda # Change this if needed
CONDA_HOME_DIR=/apps/anaconda3/bin # Change this if needed
directory_path="$CONDA_DIR/envs/$NAME"s

# Color variables
BLUE='\033[0;34m'
PINK='\033[1;35m'
NC='\033[0m' # No Color

# Creating conda environment
if [ -d "$directory_path" ]; then 
    echo -e "${BLUE}Conda environment named $NAME already exists, skip creating environment.${NC}"
else
    echo -e "${PINK}Creating conda environment named $NAME.. This might take several minutes.${NC}"
    conda create --prefix $CONDA_DIR/envs/$NAME python=3.10.0 -y
    echo -e "${BLUE}Conda Environment Created Successfully!${NC}"
fi

# Activating conda environment with correct name
source $CONDA_HOME_DIR/activate $CONDA_DIR/envs/$NAME
echo -e "${BLUE}Conda Environment Activated.${NC}"
echo -e "${PINK}Installing pipeline requirements..${NC}"

# Install correct version of pytorch with cuda enabled. Change if your cuda version if below 12.
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

pip3 install -r requirements.txt
echo -e "${PINK}All requirements satisfied.${NC}"
echo -e "${BLUE}Starting overall pipeline..${NC}"

# Call python3 uwb.py with correct flag
arg1=$1
arg2=$2
if [[ "$arg1" == "True" ]]; then
    python3 uwb.py --probed
elif [[ "$arg2" != "True" ]]; then
    python3 uwb.py --noprobestore
else
    python3 uwb.py
fi