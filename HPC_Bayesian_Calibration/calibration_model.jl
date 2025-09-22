# recalibration.jl

module Calibration_model

using Turing, Distributions, GaussianProcesses

include("./trapezoidal.jl")

export model_func

#─────────────────────────────────────────────────────────────────────────────
# Module-level constants for observational noise
const σ_cal  = 21.95
const σ_grd  = 7.540
const σ_mass = 5.721
#─────────────────────────────────────────────────────────────────────────────


#─────────────────────────────────────────────────────────────────────────────
#   @model recalibration_func(
#       Y_cal_obs, Y_grd_obs, Y_mass_obs,
#       X_mins, X_maxs,
#       Yc_mins, Yc_maxs,
#       Yg_mins, Yg_maxs,
#       Ym_mins, Ym_maxs,
#       gps_cal, gps_grd, gps_mass
#   )
#
# A “vectorized” Turing model: at each θ‐draw it:
#   •  broadcasts `predict_y.` to get all calving‐front means & variances,
#   •  broadcasts for grounding‐line,
#   •  broadcasts for mass‐change,
# then transforms and applies Gaussian‐likelihoods in a short loop.
#─────────────────────────────────────────────────────────────────────────────
@model function model_func(
    Y_obs::Vector{Float64},    # length = n_years, must take form (calObs, grdObs, massObs,volObs1, volObs2, .....)
    X_mins::Dict{Symbol,Float64},
    X_maxs::Dict{Symbol,Float64},
    Yc_mins::Vector{Float64},      # original‐scale min for calving‐front per year
    Yc_maxs::Vector{Float64},      # original‐scale max for calving‐front per year
    Yg_mins::Vector{Float64},      # original‐scale min for grounding‐line per year
    Yg_maxs::Vector{Float64},      # original‐scale max for grounding‐line per year
    Ym_mins::Vector{Float64},      # original‐scale min for mass‐change per year
    Ym_maxs::Vector{Float64},      # original‐scale max for mass‐change per year
    Yvol_mins::Vector{Float64},
    Yvol_maxs::Vector{Float64},
    Yvol_vars::Vector{Float64},
    gp_cal,
    gp_grd,
    gp_mass,
    gps_vol
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
    cal_preds  = predict_y.(gp_cal,  Ref(θ))   # Vector of ((μc, varc), …)
    grd_preds  = predict_y.(gp_grd,  Ref(θ))
    mass_preds = predict_y.(gp_mass, Ref(θ))
    vol_preds =  predict_y.(gps_vol, Ref(θ))
    

    # 4) Unzip raw means/vars
    μc_raws   = only.([c[1] for c in cal_preds])   # in [0,1]
    varc_raws = [c[2][1] for c in cal_preds]
    μg_raws   = only.([g[1] for g in grd_preds])
    varg_raws = [g[2][1] for g in grd_preds]
    μm_raws   = only.([m[1] for m in mass_preds])
    varm_raws = [m[2][1] for m in mass_preds]
    μVol_raws   = only.([m[1] for m in vol_preds])
    varVol_raws = [m[2][1] for m in vol_preds]
    

    # 5) Compute raw SDs
    sc_raw = sqrt.(varc_raws)
    sg_raw = sqrt.(varg_raws)
    sm_raw = sqrt.(varm_raws)
    sv_raw = sqrt.(varVol_raws)
    

    # 6) Un‐scale to original units
    μc_un = μc_raws .* (Yc_maxs .- Yc_mins) .+ Yc_mins
    sc_un = sc_raw    .* (Yc_maxs .- Yc_mins)
    μg_un = μg_raws .* (Yg_maxs .- Yg_mins) .+ Yg_mins
    sg_un = sg_raw    .* (Yg_maxs .- Yg_mins)
    μm_un = μm_raws .* (Ym_maxs .- Ym_mins) .+ Ym_mins
    sm_un = sm_raw    .* (Ym_maxs .- Ym_mins)
    
    μv_un = μVol_raws .* (Yvol_maxs .- Yvol_mins) .+ Yvol_mins
    sv_un = sv_raw    .* (Yvol_maxs .- Yvol_mins)

    # Concatenate the different emulator mean outputs
    means = vcat(μc_un,μg_un,μm_un,μv_un)

    # Concatenate the different emulator standard deviation outputs
    sigma_all = vcat(sc_un, sg_un, sm_un, sv_un)
    obs_noise_all = vcat(σ_cal, σ_grd, σ_mass, Yvol_vars)
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