module amery_sequential_calibration 
      
# Import libraries
using Turing
using LinearAlgebra
using Distributions
using MultivariateStats
import MultivariateStats: reconstruct
using GaussianProcesses
using StatsBase
using Statistics
using StatsPlots
using Suppressor
using JLD2
using CSV
using DataFrames, DataFramesMeta
using SplitApplyCombine
using KernelFunctions
using MCMCChains
using PyCall
using PyPlot
using Printf
import PyCall.pyfunction



scipy = pyimport("scipy")
np = pyimport("numpy")

include("../Utils/scale_utils.jl")
using .ScaleUtils

include("../Utils/gp_utils.jl")
using .GPUtils

include("../HPC_Bayesian_Calibration/calibration_model.jl")
using .Calibration_model

using Random

        
export calibration_func

    #––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

                            #CALIBRATION FUNCTION

    #––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    function calibration_func(
        future_obs::Vector{Float64},
        future_years::Vector{Int64},
        years_selected::Vector{Int64},
        chains_directory::String,
        generated_quantities_directory::String,
        X_mins::Dict{Symbol,Float64}, 
        X_maxs::Dict{Symbol,Float64},
        Yc_mins::Vector{Float64},
        Yc_maxs::Vector{Float64},
        Yg_mins::Vector{Float64},
        Yg_maxs::Vector{Float64},
        Ysmb_mins::Vector{Float64},
        Ysmb_maxs::Vector{Float64},
        Ymass_mins::Vector{Float64},
        Ymass_maxs::Vector{Float64},
        mass_uncertainties::DataFrame,
        gps_cal,
        gps_grd,
        gps_smb,
        gps_mass_all

    )

            #Indices required for the current number of years of data used 
            col_idx = findall(y -> y in years_selected, future_years)
            println(col_idx)
            #subset your observations according to years used
            Y_mass_sub = future_obs[col_idx[1]:col_idx[end]]

            #Concatenate the 2015 observations to the front
            Y_obs_w_present = vcat([-11.94, 0.9877, -1.656], Y_mass_sub )

            #Subset the observational variance
            Ymass_vars_df = filter(row -> row.year in years_selected , mass_uncertainties);
            Ymass_vars = Ymass_vars_df[!,:sigma];
            
            #Subset the grounded mass change emulators
            gps_mass = [gps_mass_all[y] for y in years_selected]

            #Subset the grounded mass change mins and maxs
            y_mass_mins_sub =  Ymass_mins[col_idx[1]:col_idx[end]] 
            y_mass_maxs_sub = Ymass_maxs[col_idx[1]:col_idx[end]]

            #Define your turing model
            model = Calibration_model.model_func(
                Y_obs_w_present,
                X_mins, X_maxs,
                Yc_mins, Yc_maxs,
                Yg_mins, Yg_maxs,
                Ysmb_mins, Ysmb_maxs,
                y_mass_mins_sub, y_mass_maxs_sub,
                Ymass_vars, gps_cal, 
                gps_grd, gps_smb, gps_mass
            )
            
            #chain = sample(model, NUTS(), MCMCSerial(),200000, 1, discard_initial=100000; progress=true)
            chain = sample(model, NUTS(), MCMCSerial(),1000, 1, discard_initial=500; progress=true)


            #Save the chain
            @save "$(chains_directory)/$(years_selected[end])_chain.jld2" chain
            
            #Since we're not running the most recent version of turing, this is the only way to get the generated quantities
            # chain_ret = generated_quantities(model, chain)
            
            # lpri = [r.log_pri for r in chain_ret]
            # llik = [r.log_lik for r in chain_ret]
            # lpost = [r.log_post for r in chain_ret]
            
            # μ = hcat([r.μ for r in chain_ret]...)'
            # σ = hcat([r.σ for r in chain_ret]...)';

            # #Save the generated quantities
            # @save "$(generated_quantities_directory)/$(years_selected[end])_GQs.jld2" lpri llik lpost μ σ

            n_fut_years = length(years_selected)
            println("Sampled and saved with $(n_fut_years) years of future constraining data")


    
    return nothing

    end






end



