# ── recalibration.jl ──
#
# The Turing calibration model. At each θ-draw it broadcasts the GP emulators for the three
# present-day observables (calving-front, grounding-line, mass-change) and the future
# grounded-volume trajectory, un-scales to physical units, and applies the likelihood: the
# present-day block is diagonal (independent Normals); the cumulative-volume block gets the dense
# (Wiener or independent) covariance plus the optional GP code-noise term.
#
# The error model, code-noise toggle, and emulator are all read from FL_* env vars — the SAME
# vars the generator (AmeryFutureObservations.jl) reads — so the likelihood covariance matches
# the generating covariance by construction.

module Recalibration

using Turing, Distributions, GaussianProcesses, LinearAlgebra

include("./trapezoidal.jl")

export recalibration_func


#========================================================================#
# Observational-noise constants (present-day observables)
#========================================================================#
const σ_cal  = 21.95
const σ_grd  = 7.540
const σ_mass = 5.721


#========================================================================#
# Experiment toggles — read from FL_* env vars (the generator reads the SAME
# env, so likelihood ≡ generator by construction).
#
#   FL_ERR_MODEL  = "wiener" (dense Brownian obs covariance) | "indep" (diagonal per-year)
#   FL_CODE_NOISE = "1" include GP predictive/code variance | "0" strip it EVERYWHERE
#                   (from BOTH the 3 present-day observables AND the future-volume block)
#========================================================================#
const ERR_MODEL  = get(ENV, "FL_ERR_MODEL", "wiener")
const CODE_NOISE = get(ENV, "FL_CODE_NOISE", "1") == "1"
@assert ERR_MODEL in ("wiener", "indep") "FL_ERR_MODEL must be 'wiener' or 'indep', got $(ERR_MODEL)"

# Emulator query for the code-uncertainty term. DEFAULT = predict_f (deterministic-simulator
# convention: latent GP variance only — the fixed 0.01 nugget is a Cholesky regularizer, NOT code
# uncertainty, so it is excluded). Set FL_GP_PREDICT=y ONLY to reproduce the old predict_y behavior.
#   FL_GP_PREDICT = "f" (default) → predict_f : latent variance only
#                 = "y"           → predict_y : latent variance + the fixed 0.01 nugget
const GP_PRED = get(ENV, "FL_GP_PREDICT", "f") == "y" ? predict_y : predict_f


#========================================================================#
# recalibration_func — the vectorized Turing model.
#
#   Y_obs = [calObs, grdObs, massObs, volObs_2030, volObs_2045, …, volObs_2300]
#   plus the per-observable scaling min/max, the per-year volume σ (Yvol_vars),
#   and the GP emulator objects.
#========================================================================#
@model function recalibration_func(
    Y_obs::Vector{Float64},    # [cal, grd, mass, vol_2030, vol_2045, …]
    X_mins::Dict{Symbol,Float64},
    X_maxs::Dict{Symbol,Float64},
    Yc_mins::Vector{Float64},      # original-scale min/max, calving-front, per year
    Yc_maxs::Vector{Float64},
    Yg_mins::Vector{Float64},      # grounding-line
    Yg_maxs::Vector{Float64},
    Ym_mins::Vector{Float64},      # mass-change
    Ym_maxs::Vector{Float64},
    Yvol_mins::Vector{Float64},
    Yvol_maxs::Vector{Float64},
    Yvol_vars::Vector{Float64},
    gp_cal,
    gp_grd,
    gp_mass,
    gps_vol
)

    # Priors on the 6 input parameters, truncated to the Sobol design ranges.
    vmThresh    ~ Truncated(Normal(130000, 50000/3),  80000, 180000)
    fricExp     ~ Truncated(TrapezoidalDist(0.1,0.15,0.28,0.333), 0.1, 0.333)
    mu_scale    ~ Truncated(Normal(1.0, 0.1),  0.8, 1.2)
    stiff_scale ~ Truncated(Normal(1.0, 0.1),  0.8, 1.2)
    gamma0      ~ Truncated(LogNormal(10, 1),  9620, 471000)
    melt_flux   ~ Truncated(Normal(35, 11.5),  12, 58)

    # Explicit log-prior — returned as a diagnostic only. The ~ statements above are what actually
    # enter the sampled density; this sum is NOT added to the target again.
    log_pri_1 = logpdf(Truncated(Normal(130000, 50000/3),  80000, 180000), vmThresh)
    log_pri_2 = logpdf(Truncated(TrapezoidalDist(0.1,0.15,0.28,0.333), 0.1, 0.333), fricExp)
    log_pri_3 = logpdf(Truncated(Normal(1.0, 0.1),  0.8, 1.2), mu_scale)
    log_pri_4 = logpdf(Truncated(Normal(1.0, 0.1),  0.8, 1.2), stiff_scale)
    log_pri_5 = logpdf(Truncated(LogNormal(10, 1),  9620, 471000), gamma0)
    log_pri_6 = logpdf(Truncated(Normal(35, 11.5),  12, 58), melt_flux)
    log_pri = log_pri_1 + log_pri_2 + log_pri_3 + log_pri_4 + log_pri_5 + log_pri_6

    # Rescale inputs to [0,1] using the emulator-training min/max.
    p_vm     = (vmThresh    - X_mins[:vmThresh])   / (X_maxs[:vmThresh]    - X_mins[:vmThresh])
    p_fric   = (fricExp     - X_mins[:fricExp])    / (X_maxs[:fricExp]     - X_mins[:fricExp])
    p_mu     = (mu_scale    - X_mins[:mu_scale])   / (X_maxs[:mu_scale]    - X_mins[:mu_scale])
    p_stiff  = (stiff_scale - X_mins[:stiff_scale])/(X_maxs[:stiff_scale] - X_mins[:stiff_scale])
    p_gamma0 = (gamma0      - X_mins[:gamma0])     / (X_maxs[:gamma0]      - X_mins[:gamma0])
    p_melt   = (melt_flux   - X_mins[:melt_flux])  / (X_maxs[:melt_flux]   - X_mins[:melt_flux])

    θ = [p_vm;; p_fric;; p_mu;; p_stiff;; p_gamma0;; p_melt]'

    # Present-day observables: broadcasted GP predictions (mean + variance).
    cal_preds  = GP_PRED.(gp_cal,  Ref(θ))
    grd_preds  = GP_PRED.(gp_grd,  Ref(θ))
    mass_preds = GP_PRED.(gp_mass, Ref(θ))

    μc_raws   = only.([c[1] for c in cal_preds])
    varc_raws = [c[2][1] for c in cal_preds]
    μg_raws   = only.([g[1] for g in grd_preds])
    varg_raws = [g[2][1] for g in grd_preds]
    μm_raws   = only.([m[1] for m in mass_preds])
    varm_raws = [m[2][1] for m in mass_preds]

    sc_raw = sqrt.(varc_raws)
    sg_raw = sqrt.(varg_raws)
    sm_raw = sqrt.(varm_raws)

    # Un-scale present-day means and SDs to physical units.
    μc_un = μc_raws .* (Yc_maxs .- Yc_mins) .+ Yc_mins
    sc_un = sc_raw  .* (Yc_maxs .- Yc_mins)
    μg_un = μg_raws .* (Yg_maxs .- Yg_mins) .+ Yg_mins
    sg_un = sg_raw  .* (Yg_maxs .- Yg_mins)
    μm_un = μm_raws .* (Ym_maxs .- Ym_mins) .+ Ym_mins
    sm_un = sm_raw  .* (Ym_maxs .- Ym_mins)

    # Future cumulative-volume block — per-year GPs return μv_un in Gt.
    vol_preds   = GP_PRED.(gps_vol, Ref(θ))
    μVol_raws   = only.([m[1] for m in vol_preds])
    varVol_raws = [m[2][1] for m in vol_preds]
    sv_raw      = sqrt.(varVol_raws)
    μv_un = μVol_raws .* (Yvol_maxs .- Yvol_mins) .+ Yvol_mins
    sv_un = sv_raw    .* (Yvol_maxs .- Yvol_mins)

    means = vcat(μc_un, μg_un, μm_un, μv_un)

    # Present-day (and returned-diagnostic) marginal SDs; drop the GP code variance when code-off.
    sigma_all     = vcat(sc_un, sg_un, sm_un, sv_un)
    obs_noise_all = vcat(σ_cal, σ_grd, σ_mass, Yvol_vars)
    sigmas = CODE_NOISE ? sqrt.(obs_noise_all.^2 .+ sigma_all.^2) : obs_noise_all

    # Likelihood. Present-day block (cal, grd, mass) stays diagonal; the cumulative-volume block
    # gets the dense covariance.
    n_vol = length(μv_un)             # number of future cumulative-volume observations
    n_now = length(Y_obs) - n_vol     # present-day observables (here 3: cal, grd, mass)

    # Present-day: independent Normals.
    Y_obs[1:n_now] .~ Normal.(means[1:n_now], sigmas[1:n_now])

    # Cumulative-volume covariance = obs part (+ code part when code-on), matched term-for-term to
    # the generator draw (see AmeryFutureObservations.jl):
    #   obs part:  wiener → Σ_o[i,j]=min(Yvol_vars[i]²,Yvol_vars[j]²)=σ_a²·min(τ_i,τ_j)
    #                       (since Yvol_vars = σ_a·√(year−2016));
    #              indep  → Diagonal(Yvol_vars²)  (per-year independent, SAME marginals).
    #   code part: + Diagonal(sv_un²) only when CODE_NOISE (per-year GP predictive variance).
    Σ_obs = ERR_MODEL == "wiener" ?
                [min(Yvol_vars[i]^2, Yvol_vars[j]^2) for i in 1:n_vol, j in 1:n_vol] :
                diagm(Yvol_vars.^2)
    if CODE_NOISE
        code_mat = Diagonal(sv_un.^2)
        Σ_vol = Symmetric(Σ_obs .+ code_mat)
    else
        Σ_vol = Symmetric(Σ_obs)
    end
    Y_obs[n_now+1:end] ~ MvNormal(μv_un, Σ_vol)

    # Explicit log-likelihood / log-posterior — returned as diagnostics only.
    log_lik = sum(logpdf.(Normal.(means[1:n_now], sigmas[1:n_now]), Y_obs[1:n_now])) +
              logpdf(MvNormal(μv_un, Σ_vol), Y_obs[n_now+1:end])
    log_post = log_lik + log_pri

    return (μ=means, σ=sigmas, log_pri=log_pri, log_lik=log_lik, log_post=log_post)
end

end # module Recalibration