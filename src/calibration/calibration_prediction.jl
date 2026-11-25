# ── calibration_prediction.jl ──
#
# The per-realization calibration driver. `calibration_func` assembles the observation
# vector for the requested number of future years, builds the Turing model from
# Recalibration.recalibration_func, runs a single warm-started NUTS chain, and writes the
# chain plus the model's generated quantities (μ, σ, log-prior / -likelihood / -posterior)
# to disk. The sampling knobs (burn-in, adaptation, post-burn-in count, target acceptance)
# are ENV-overridable via FL_* so a smoke test can shrink them without editing this file.

module amery_calibration_prediction

using Turing, AdvancedHMC, AbstractMCMC, DynamicPPL
using MCMCChains
using DataFrames
using JLD2
using Printf
using Random

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "..", "utils", "recalibration.jl"))
using .Recalibration

export calibration_func


#========================================================================#
# Calibration parameters (declaration order matches the Turing model)
#========================================================================#
const PARAMS = Symbol[:vmThresh, :fricExp, :mu_scale, :stiff_scale, :gamma0, :melt_flux]


#========================================================================#
# Sampling knobs.
#
# A fixed post-burn-in of N_POST samples. BURNIN / N_ADAPTS / N_POST are ENV-overridable
# (FL_BURNIN / FL_N_ADAPTS / FL_N_POST) so a smoke test can shrink them WITHOUT editing this
# file (avoids the risk of shipping reduced consts into the real 1900-job batch).
#
# NUTS target acceptance: default 0.8 reproduces run B. The no-code-uncertainty experiments
# (C/D) have a much tighter likelihood (sharper posterior curvature) → raise to ~0.95 via
# FL_TARGET_ACCEPT to curb divergences / step-size collapse.
#========================================================================#
const BURNIN     = parse(Int, get(ENV, "FL_BURNIN",  "100000"))
const N_ADAPTS   = parse(Int, get(ENV, "FL_N_ADAPTS", "50000"))   # discard_initial > N_ADAPTS keeps adaptation out of the kept chain
const N_POST     = parse(Int, get(ENV, "FL_N_POST",  "100000"))   # post-burn-in sample count
const TARGET_ESS = 10_000          # reporting threshold: warn if the min parameter ESS falls below this
const TARGET_ACCEPT = parse(Float64, get(ENV, "FL_TARGET_ACCEPT", "0.8"))


#========================================================================#
# calibration_func — calibrate against `future_obs` for the selected years.
#
#   Assembles Y_obs = [calObs, grdObs, massObs, volObs_2030, …] (the fixed present-day
#   observed triple + the future grounded-volume sub-vector for `years_selected`), builds
#   the Recalibration Turing model, warm-starts NUTS at θ_init (truth basin) in linked
#   space, samples one chain, and saves the chain + generated quantities keyed by the last
#   selected year.
#========================================================================#
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
    Ym_mins::Vector{Float64},
    Ym_maxs::Vector{Float64},
    Yvol_mins::Vector{Float64},
    Yvol_maxs::Vector{Float64},
    vol_uncertainties::DataFrame,
    gps_cal,
    gps_grd,
    gps_mass,
    gps_Vol_all,
    θ_init::Vector{Float64}
)

    # Indices required for the current number of years used.
    col_idx = findall(y -> y in years_selected, future_years)
    println(col_idx)
    # Subset the observations according to the years used.
    Y_vol_sub = future_obs[col_idx[1]:col_idx[end]]
    Y_obs = vcat([-11.94, 0.9877, -1.656], Y_vol_sub)

    Yvol_vars_df = filter(row -> row.year in years_selected, vol_uncertainties)
    Yvol_vars = Yvol_vars_df[!, :sigma]

    gps_Vol = [gps_Vol_all[y] for y in years_selected]

    y_vol_mins_sub = Yvol_mins[col_idx[1]:col_idx[end]]
    y_vol_maxs_sub = Yvol_maxs[col_idx[1]:col_idx[end]]

    model = Recalibration.recalibration_func(
        Y_obs,
        X_mins, X_maxs,
        Yc_mins, Yc_maxs,
        Yg_mins, Yg_maxs,
        Ym_mins, Ym_maxs,
        y_vol_mins_sub, y_vol_maxs_sub,
        Yvol_vars, gps_cal,
        gps_grd, gps_mass, gps_Vol
    )

    # target_accept from FL_TARGET_ACCEPT (default 0.8 = run B).
    nuts = AdvancedHMC.NUTS(TARGET_ACCEPT; metric = :dense)

    # Warm-start at θ_init (truth basin) in unconstrained/linked space; kwarg is `init_params`.
    pri_lo = [80000.0,  0.1,   0.8, 0.8, 9620.0,   12.0]
    pri_hi = [180000.0, 0.333, 1.2, 1.2, 471000.0, 58.0]
    Xmn = [X_mins[p] for p in PARAMS]; Xmx = [X_maxs[p] for p in PARAMS]
    θ_phys = clamp.(θ_init .* (Xmx .- Xmn) .+ Xmn,
                    pri_lo .+ 1e-4 .* (pri_hi .- pri_lo),
                    pri_hi .- 1e-4 .* (pri_hi .- pri_lo))
    init0 = DynamicPPL.link!!(DynamicPPL.unflatten(DynamicPPL.VarInfo(model), θ_phys), model)[:]
    @printf("warm-start θ_phys = %s\n", string(round.(θ_phys; digits = 4))); flush(stdout)

    # One warm-started NUTS chain of N_POST post-burn-in draws; report the min parameter ESS.
    N_post = N_POST
    chain = sample(
        model, externalsampler(nuts), MCMCSerial(), N_post, 1;
        discard_initial = BURNIN,
        n_adapts = N_ADAPTS,
        init_params = [init0],
        progress = true,
    )
    df = DataFrame(MCMCChains.summarystats(chain; sections = :parameters))
    keep = in.(Symbol.(df.parameters), Ref(Set(PARAMS)))
    ess_min = minimum(collect(df.ess[keep]))
    @printf("N_post=%d  min ESS=%.1f\n", N_post, ess_min); flush(stdout)

    if ess_min < TARGET_ESS
        @warn "min parameter ESS below target" ess_min TARGET_ESS nsamples=N_post
    end

    # Save the chain.
    @save "$(chains_directory)/$(years_selected[end])_chain.jld2" chain

    chain_params = MCMCChains.get_sections(chain, :parameters)
    chain_ret = generated_quantities(model, chain_params)

    lpri = [r.log_pri for r in chain_ret]
    llik = [r.log_lik for r in chain_ret]
    lpost = [r.log_post for r in chain_ret]

    μ = hcat([r.μ for r in chain_ret]...)'
    σ = hcat([r.σ for r in chain_ret]...)'

    # Save the generated quantities.
    @save "$(generated_quantities_directory)/$(years_selected[end])_GQs.jld2" lpri llik lpost μ σ

    n_fut_years = length(years_selected)
    println("Sampled and saved with $(n_fut_years) years of future constraining data")

    return nothing
end

end # module amery_calibration_prediction
