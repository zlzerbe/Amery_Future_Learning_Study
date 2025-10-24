/Computational Notebooks 
This directory consists of notebooks used to compute various quantities in the statistical workflow employed by this study.

GVC_Emulator_Validation_And_Training.ipynb – This notebook validates 19 scalar emulators trained on ensembles of MALI input parameters and the corresponding cumulative grounded mass change projection data from Jantre 2024 using 5-fold cross validation.
Amery_Future_Observations.ipynb – This notebook is used to generate the future realistic observations of cumulative grounded mass change, as discussed in the Methods section. It requires the use of trained and validated scalar Gaussian Process Emulators.	
Local_Calibration.ipynb – This notebook is a way for users of this repository to validate the Bayesian Calibration process we employed on their local machine. It produces Markov chains containing samples of the 6 MALI input parameter posteriors and the quantities generated during sampling. Note that generating sample sizes as seen in our manuscript locally can take many hours. To generate calibrated chains for further scientific analysis, it is recommended to use the high performance computing workflow in the HPC_Bayesian_Calibration directory.
SLR_PCA_Projection.ipynb – This notebook produces projections of volume above flotation and the corresponding sea level contribution, given a set of samples of the six MALI parameters. 
Parameter_Learning_Tables.ipynb & SLR_ Learning_Tables.ipynb – These notebooks generate tables of numeric values for the results of this study. In-depth descriptions of these tables and their quantities can be found in the supporting material.

/Data
This directory contains all the data used or generated in this study, as well as trained and validated Gaussian Process emulators saved as JLD2 objects.

Future_Observation_Data – Contains the synthetic future observations used to constrain priors in this study, as well as the MALI parameters which generated the trajectories.
Posterior_Data – Contains 100 Dictionaries, each dictionary mapping the final year of calibrating observations to the posterior samples constrained with observations up to that year. 
Projection_Data – Contains 100 subfolders, each with 19 time series matrices of SLE projections from 2015 to 2300.
Training_Data – Contains various data used throughout the workflow.
Parameter_Learning_Tables & SLR_Learning_Tables – CSV files containing the numerical results generated in this study.
*All emulators used in this study, saved as JLD2 objects, are stored in this directory.


/HPC_Bayesian_Calibration
This directory contains all of the julia, bash, and SLURM scripts used to do large scale MCMC sampling on a high performance computing cluster. These are included for transparency, but the code will need to be modified to work on the user’s specific cluster.

/Plots
This directory contains all of the plots generated for the analysis of the results of this study. Some are contained within the main body of the paper, and others are included in its supplementary material.


/Plotting_Notebooks
This directory contains all of the notebooks needed to generate the plots included in the main body and supplementary materials.


/Utils
This directory contains a few Julia scripts used in various parts of the workflow employed in this study.
















				
