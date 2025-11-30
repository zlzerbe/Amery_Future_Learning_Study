# recalibration.jl

module Calibration_model

using Turing, Distributions, GaussianProcesses

include("../Utils/trapezoidal.jl")

export model_func

#─────────────────────────────────────────────────────────────────────────────
# Module-level constants for observational noise
const σ_cal  = 21.95
const σ_grd  = 7.540
const σ_smb = 5.721
#─────────────────────────────────────────────────────────────────────────────


#─────────────────────────────────────────────────────────────────────────────
#
# A “vectorized” Turing model: at each θ‐draw it:
#   broadcasts `predict_y.` to get all calving‐front, surface mass balance, grounding line and cumulative grounded mass change means & variances,
# then transforms and applies Gaussian‐likelihoods in a short loop.
#─────────────────────────────────────────────────────────────────────────────
@model function model_func(
    Y_obs::Vector{Float64},    
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
    Ymass_vars::Vector{Float64},
    gp_cal,
    gp_grd,
    gp_smb,
    gps_mass
)

    # 1) Priors on the 6 input parameters, truncated to Sobol ranges:
    vmThresh    ~ Truncated(Normal(130000, 50000/3),  80000, 180000)
    fricExp     ~ Truncated(TrapezoidalDist(0.1,0.15,0.28,0.333), 0.1, 0.333)
    mu_scale    ~ Truncated(Normal(1.0, 0.1),  0.8, 1.2)
    stiff_scale ~ Truncated(Normal(1.0, 0.1),  0.8, 1.2)
    gamma0      ~ Truncated(LogNormal(10, 1),  9620, 471000)
    melt_flux   ~ Truncated(Normal(35, 11.5),  12, 58)                         

    # Calulcate the log priors of each individual parameter
    log_pri_1 = logpdf(Truncated(Normal(130000, 50000/3),  80000, 180000), vmThresh)
    log_pri_2 = logpdf(Truncated(TrapezoidalDist(0.1,0.15,0.28,0.333), 0.1, 0.333), fricExp)
    log_pri_3 = logpdf(Truncated(Normal(1.0, 0.1),  0.8, 1.2), mu_scale)
    log_pri_4 = logpdf(Truncated(Normal(1.0, 0.1),  0.8, 1.2), stiff_scale)
    log_pri_5 = logpdf(Truncated(LogNormal(10, 1),  9620, 471000), gamma0)
    log_pri_6 = logpdf(Truncated(Normal(35, 11.5),  12, 58), melt_flux) 

    # Compute the joint log prior 
    log_pri = log_pri_1 + log_pri_2 + log_pri_3 + log_pri_4 + log_pri_5 + log_pri_6
    
    

    # 2) Rescale inputs to [0,1] using X_mins / X_maxs
    p_vm     = (vmThresh    - X_mins[:vmThresh])   / (X_maxs[:vmThresh]    - X_mins[:vmThresh])
    p_fric   = (fricExp     - X_mins[:fricExp])    / (X_maxs[:fricExp]     - X_mins[:fricExp])
    p_mu     = (mu_scale    - X_mins[:mu_scale])   / (X_maxs[:mu_scale]    - X_mins[:mu_scale])
    p_stiff  = (stiff_scale - X_mins[:stiff_scale])/(X_maxs[:stiff_scale] - X_mins[:stiff_scale])
    p_gamma0 = (gamma0      - X_mins[:gamma0])     / (X_maxs[:gamma0]      - X_mins[:gamma0])
    p_melt   = (melt_flux   - X_mins[:melt_flux])  / (X_maxs[:melt_flux]   - X_mins[:melt_flux])

    θ = [p_vm;; p_fric;; p_mu;; p_stiff;; p_gamma0;; p_melt]'


    # 3) Broadcasted GP predictions for each observable:
    cal_preds  = predict_y.(gp_cal,  Ref(θ))   
    grd_preds  = predict_y.(gp_grd,  Ref(θ))
    smb_preds = predict_y.(gp_smb, Ref(θ))
    mass_preds =  predict_y.(gps_mass, Ref(θ))

    # 4) Unzip raw means/vars
    μc_raws   = only.([c[1] for c in cal_preds])   
    varc_raws = [c[2][1] for c in cal_preds]
    μg_raws   = only.([g[1] for g in grd_preds])
    varg_raws = [g[2][1] for g in grd_preds]
    μ_smb_raws   = only.([m[1] for m in smb_preds])
    var_smb_raws = [m[2][1] for m in smb_preds]
    μ_mass_raws   = only.([m[1] for m in mass_preds])
    var_mass_raws = [m[2][1] for m in mass_preds]
    

    # 5) Compute raw SDs
    sc_raw = sqrt.(varc_raws)
    sg_raw = sqrt.(varg_raws)
    s_smb_raw = sqrt.(var_smb_raws)
    
    s_mass_raw = sqrt.(var_mass_raws)
    

    # 6) Un‐scale to original units
    μc_un = μc_raws .* (Yc_maxs .- Yc_mins) .+ Yc_mins
    sc_un = sc_raw    .* (Yc_maxs .- Yc_mins)
    μg_un = μg_raws .* (Yg_maxs .- Yg_mins) .+ Yg_mins
    sg_un = sg_raw    .* (Yg_maxs .- Yg_mins)
    
    μ_smb_un = μ_smb_raws .* (Ysmb_maxs .- Ysmb_mins) .+ Ysmb_mins
    s_smb_un = s_smb_raw    .* (Ysmb_maxs .- Ysmb_mins)
    
    μ_mass_un = μ_mass_raws .* (Ymass_maxs .- Ymass_mins) .+ Ymass_mins
    s_mass_un = s_mass_raw    .* (Ymass_maxs .- Ymass_mins)

    # Concatenate the different emulator mean outputs
    means = vcat(μc_un,μg_un,μ_smb_un,μ_mass_un)

    # Concatenate the different emulator uncertainties
    sigma_all = vcat(sc_un, sg_un, s_smb_un, s_mass_un)
    # Concatenate the different observational uncertainties
    obs_noise_all = vcat(σ_cal, σ_grd, σ_smb, Ymass_vars)
    sigmas = sqrt.(obs_noise_all.^2 .+ sigma_all.^2)
    

    # 7) Likelihoods (vectorized)
    Y_obs .~ Normal.(means, sigmas)

    # Compute the log-likelihood function
    log_lik = sum(logpdf.(Normal.(means, sigmas), Y_obs) )
    # Compute the log posteriors
    log_post = log_lik + log_pri


    
    return (μ=means,σ=sigmas,log_pri=log_pri,log_lik=log_lik,log_post=log_post)
end

end # module Recalibration