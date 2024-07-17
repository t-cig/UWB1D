# Passive Ultra-Wideband Single-Photon Imaging

This is the 1D data and codebase for the ICCV 2023 paper "Passive Ultra-Wideband Single-Photon Imaging". For more information, see our paper [here](https://www.dgp.toronto.edu/projects/ultra-wideband/static/docs/paper.pdf) and our website [here](https://www.dgp.toronto.edu/projects/ultra-wideband/). 

### Introduction to the repository
* The current repository represents a real, captured experiment, where an unsynchronized single-photon avalanche diode (SPAD) passively records indirect light coming from multiple sources operating asynchronously from each other (unsynchronized picosecond lasers, projectors, etc.) 
* The final output of this code pipeline is the center four figures in Figure 1 of the [paper](https://www.dgp.toronto.edu/projects/ultra-wideband/static/docs/paper.pdf). The actual scene can be found in Figure 5 (row 2, middle). 
* Data captured by the SPAD is current stored in the MatLab file `experiment1/scan_posX001_posY001.mat`.

### Running the 1D Ultra-Wideband Pipeline



#### Running the pipeline with anaconda3
If you are running the codebase on Windows command line, please skip this section and proceed directly to the section "Running the pipeline directly from terminal". 

Before executing the overall pipeline, you should have anaconda3 installed and fill in the `.conda` and home directories of anaconda3 in file `init.bash`. 

The default setting of the pipeline assumes that cuda is enabled, and has version 12.*. If you would like to run it on cpu, proceed directly to "Hardware requirement for the pipeline" section. If you cuda version is lower, change the line 27 in `init.bash` to install correct cuda-enabled version of torch. 

Then, the begin-to-end pipeline can be simply executed with `python3 uwb.py --conda`.  Runnning this will automatically create a new anaconda3 environment called `UWB`, activate this environment, install all required packages in this environment and then execute the pipeline.

#### Running the pipeline directly from terminal
You can also simply run `python3 uwb.py` to run the overall pipeline from the terminal or already activated conda environment. Note that you will need a python version of `3.10.0` and above. Also, make sure to install the correct version of torch you need (for specific cuda version or cpu) before running `python3 uwb.py`.

All other required packages for the codebase will automatically be installed. 

#### Hardware requirement for the pipeline
Default hardware for running the pipeline is `torch.device('cuda:0')` with cuda version `12.*`, and you can modify it in the variable `device` of `constants.py` (e.g. cpu). Note that with cpu, the overall pipeline would take several hours to finish execution.

#### Several options with running the pipeline
1. Without the `--conda` flag, the pipeline assumes that the desired conda environment is activated or that the pipeline is run directly through terminal without a conda environment. The pipeline will still check whether all required packages are installed and will install the missing ones. If you would like to create/activate the conda environment UWB on Mac/Linux shell, you can add this flag. 
2. If you have all probed files in MatLab format ready and would only like to run the frequency thresholding and flux reconstruction parts of the pipeline, you can add the `--probed` flag when running `python3 uwb.py`. This will only run the frequency thresholding and flux reconstruction parts of the pipeline.
3. Creating and saving all probing files (default) will take several (~20-30GB) of storage. To run the pipeline in a more memory-efficient way, you can add in the `--noprobestore` flag when running `python3 uwb.py`. With this flag, no probing intermediate files will be saved. Note that since the pipeline execution takes time, with this flag, you need to wait for the overall pipeline to re-run if the session disconnects.
4. By default, all outputs of flux reconstruction will be stored in the folder `figures` inside current directory. However, you can also change the `RECONSTRUCTION_FOLDER` inside the `constants.py` file. To probe data files other than the data file provided by us in this repo, you can also change `EXPERIMENT, PROBE_FILE, DATA_NAME` in `constants.py`. Please make sure that `PROBE_FILE` is contained in the `PROBE_FOLDER/EXPERIMENT` directory. 

When using this code in your projects, please cite:

```bibtex
@inproceedings{wei2023ultrawideband,
  author    = {Wei, Mian and Nousias, Sotiris and Gulve, Rahul and Lindell, David B and Kutulakos, Kiriakos N},
  title     = {Passive Ultra-Wideband Single-Photon Imaging},
  booktitle = {Proc. ICCV},
  year      = {2023},
}
```
