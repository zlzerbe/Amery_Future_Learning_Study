This repository contains the code, and some of the data employed and generated in the paper "Amery Ice Shelf Future Learning Study" (working title). Below is a description of the contents. Feel free to reach out if you have any questions about the repository or paper. 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17923081.svg)](https://doi.org/10.5281/zenodo.17923081)

/Computational Notebooks 
This directory consists of notebooks used to compute various quantities in the statistical workflow employed by this study.

GVC_Emulator_Validation_And_Training.ipynb – This notebook validates 19 scalar emulators trained on ensembles of MALI input parameters and the corresponding cumulative grounded mass change projection data from Jantre 2024 using 5-fold cross validation.
Amery_Future_Observations.ipynb – This notebook is used to generate the future realistic observations of cumulative grounded mass change, as discussed in the Methods section. It requires the use of trained and validated scalar Gaussian Process Emulators.	
Local_Calibration.ipynb – This notebook is a way for users of this repository to validate the Bayesian Calibration process we employed on their local machine. It produces Markov chains containing samples of the 6 MALI input parameter posteriors and the quantities generated during sampling. Note that generating sample sizes as seen in our manuscript locally can take many hours. To generate calibrated chains for further scientific analysis, it is recommended to use the high performance computing workflow in the HPC_Bayesian_Calibration directory.
SLR_PCA_Projection.ipynb – This notebook produces projections of volume above flotation and the corresponding sea level contribution, given a set of samples of the six MALI parameters. 


/Data
This directory contains all the data used or generated in this study, as well as trained and validated Gaussian Process emulators saved as JLD2 objects.

Future_Observation_Data – Contains the synthetic future observations of cumulative grounded mass change used to constrain priors in this study, as well as the MALI parameters which generated each trajectory of cumulative grounded mass change.
Posterior_Data – Contains 100 Dictionaries, each dictionary mapping the final year of calibrating observations to the posterior samples constrained with observations up to that year. 
Projection_Data – Contains 100 subfolders, each with 19 time series matrices of SLE projections from 2015 to 2300.
Training_Data – Contains various data used throughout the workflow.
*All emulators used in this study, saved as JLD2 objects, are stored in this directory.
HPC_Data – Currently empty, this was originally used to hold the turing.jl chain objects containing the markov chains of postertior samples and the quantities generated and saved in that sampling process (log_priors, log_likelihoods, etc.). That directory was very large in size, and so has not been included in this release, but the posterior samples themselves are available in /Posterior_Data.


/HPC_Bayesian_Calibration
This directory contains all of the julia, bash, and SLURM scripts used to do large scale MCMC sampling on a high performance computing cluster. These are included for transparency, but the code will need to be modified to work on the user’s specific cluster.

/Plots
This directory contains all of the plots generated for the analysis of the results of this study. Some are contained within the main body of the paper, and others are included in its supplementary material.


/Plotting_Notebooks
This directory contains all of the notebooks needed to generate the plots included in the main body and supplementary materials.


/Utils
This directory contains a few Julia scripts used in various parts of the workflow employed in this study.
















				
