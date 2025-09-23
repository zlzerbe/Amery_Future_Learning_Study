# Import libraries
using Turing
using LinearAlgebra
using Distributions
using MultivariateStats
import MultivariateStats: reconstruct
using GaussianProcesses
using Optim
using StatsBase
using Statistics
using Suppressor
using JLD2
using CSV
using DataFrames, DataFramesMeta
using SplitApplyCombine
using KernelFunctions
using MCMCChains
using QuantileRegressions
using PyCall
using PyPlot
using Printf
import PyCall.pyfunction

include("../Utils/trapezoidal.jl")

include("../Utils/scale_utils.jl")
using .ScaleUtils

include("../HPC_Bayesian_Calibration/amery_calibration.jl")
using .amery_sequential_calibration 

include("../Utils/gp_utils.jl")
using .GPUtils

scipy = pyimport("scipy")
np = pyimport("numpy")


# Set a seed for reproducibility
using Random
Random.seed!(11);

#####################################################################################################################

# BATCH SPECIFIC INFORMATION 

if length(ARGS) < 3
        error("Usage: julia run_mcmc_sampler.jl <job_id> <realization_id> <saveout_directory>")
end

#Should be number 0 - 19
job_id_str = ARGS[1]
job_idx = parse(Int, job_id_str)
years_selected = [((i*15)+2030) for i in 0:job_idx]
println(years_selected)
#Should be an int (1, 2, 3, .... 100)
realization_id_str = ARGS[2]
realization_id = parse(Int, realization_id_str)
#Should be a string: "/home/zzerbe/ProjectFolder/BrookhavenCode/ProjectNotebooks/OUTPUT_R_1"
saveout_directory = ARGS[3]

#Make a directory for saving the returned Markov Chains
if !isdir("$(saveout_directory)/chains_dir")
        mkdir("$(saveout_directory)/chains_dir")
        chains_dir = "$(saveout_directory)/chains_dir"
        println(chains_dir)
else
        chains_dir = "$(saveout_directory)/chains_dir"
        println(chains_dir)
end

#Make a directory for saving the quantities generated during sampling (Log-priors, log-likelihood, log-posteriors, emulator outputs)
if !isdir("$(saveout_directory)/gq_dir")
        mkdir("$(saveout_directory)/gq_dir")
        gq_dir = "$(saveout_directory)/gq_dir"
        println(chains_dir)
else
        gq_dir = "$(saveout_directory)/gq_dir"
        println(chains_dir)
end


#####################################################################################################################

# "Global" VARIABLES

future_years = [2030, 2045, 2060, 2075, 2090, 2105, 2120, 2135, 2150, 2165, 2180, 2195, 2210, 2225, 2240, 2255, 2270, 2285, 2300]

# Minimum and maximum values of the datasets for the three observables used to calibrate the 2015 priors (Calving front, surface mass balance, grounding line), used for scaling purposes

Yc_mins  = [-96.34022417]
Yc_maxs  = [74.9385927]
Yg_mins  = [-21.10306973]
Yg_maxs  = [19.2289032]
Ym_mins  = [-23.91989209]
Ym_maxs  = [21.04901554]



#####################################################################################################################


        #####################################################################################################################
        #####################################################################################################################
        #####################################################################################################################

                                                # Variables reusable across all calibrations #

        #####################################################################################################################
        #####################################################################################################################
        #####################################################################################################################
        
        #RELX Inputs
        X_raw = CSV.read("../Data/Training_Data/Amery_GP_Emulator_RELX_Ensemble_Filtered.csv", DataFrame);
        # 1) Grab all column‐names as Symbols
        cols = Symbol.(names(X_raw))

        # 2) Remove the index‐column symbol
        cols = filter(c -> c != :Column1, cols)

        # 3) Now call get_scaled_matrix on the remaining columns
        X_scaled_t, X_scalers, X_min_dict, X_max_dict = ScaleUtils.get_scaled_matrix(X_raw, cols)

        #####################################################################################################################

        #RELX Emulators
        cal_path = "../Data/calving_front_gap_15_emulators.jld2"
        grd_path = "../Data/grounding_line_gap_15_emulators.jld2"
        mass_path = "../Data/mass_balance_gap_15_emulators.jld2"


        # Load emulator dictionaries from disk:
        gp_cal_dict  = GPUtils.load_emulators(cal_path)
        gp_grd_dict  = GPUtils.load_emulators(grd_path)
        gp_mass_dict = GPUtils.load_emulators(mass_path)


        #####################################################################################################################

        # Cumulative Grounded Voume Change mins, maxs, and emulators

        y_vol_mins, y_vol_maxs = JLD2.load("../Data/Training_Data/grd_vol_training_mins_maxs", "y_grd_vol_mins", "y_grd_vol_maxs")

        #sort the mins and maxs by year, collect into vectors
        ordered_vol_mins = collect(values(sort(y_vol_mins)))
        ordered_vol_maxs = collect(values(sort(y_vol_maxs)))

        # Load cumulative variance data
        vol_uncertainties = CSV.read("../Data/Training_Data/sigma_grd_vol_change.csv", DataFrame);

        #load scalar volume emulators
        gps_Vol_all = GPUtils.load_emulators("../Data/Grd_vol_change_gap_15_emulators.jld2");



        #####################################################################################################################

        #subset the relx emulators for just the 2015 instances
        gps_cal  = [gp_cal_dict[2015]]
        gps_grd  = [gp_grd_dict[2015]]
        gps_mass = [gp_mass_dict[2015]]

        #Just renamed these for the main function call
        Yvol_mins = ordered_vol_mins
        Yvol_maxs = ordered_vol_maxs



        #####################################################################################################################
        #####################################################################################################################
        #####################################################################################################################

                                                # Function calls #

        #####################################################################################################################
        #####################################################################################################################
        #####################################################################################################################    
        
        #Load future observations
        fut_obs_path = "../Data/Future_Observation_Data/Official_Constraining_Observations.jld2"
        
        #Size 100x19
        all_future_obs = JLD2.load(fut_obs_path, "Y_grd_vol_obs") 

        #Subsetting the matrix of future observations according to the Realization ID, yielding a single future trajectory of 19 years worth of             #future grounded volume change observations
        future_obs = all_future_obs[realization_id,:]

        #Nothing returned by this function
        amery_sequential_calibration.calibration_func(future_obs, 
                                                        future_years,
                                                        years_selected,
                                                        chains_dir,
                                                        gq_dir,
                                                        X_min_dict, 
                                                        X_max_dict,
                                                        Yc_mins,
                                                        Yc_maxs,
                                                        Yg_mins,
                                                        Yg_maxs,
                                                        Ym_mins,
                                                        Ym_maxs,
                                                        Yvol_mins,
                                                        Yvol_maxs,
                                                        vol_uncertainties,
                                                        gps_cal,
                                                        gps_grd,
                                                        gps_mass,
                                                        gps_Vol_all )
                                                              

