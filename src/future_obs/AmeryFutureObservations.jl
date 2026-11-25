# ── AmeryFutureObservations.jl ──
#
# Synthetic future-observation generator (headless twin of
# notebooks/Future_Observation_Trajectory_Sampling.ipynb).
#
# Re-synced to the notebook's EXACT seed/draws so the headless and interactive paths produce
# identical trajectories:
#   • SEED is single-sourced (must equal the notebook's SEED).
#   • The canonical emulator is loaded, never retrained.
#   • The RNG is re-seeded immediately before the draw loop (idempotent).
#   • Draws come from the 100 fixed rows of inputs/params/present_day_2015_posterior_100_draws.npy
#     (the notebook's posterior draws), NOT a randperm of the 100k combined set.
# Output is written under a DETERMINISTIC serial ("seed9_wiener_code" for the default run B), not a random key, so re-running
# overwrites the same files. The Wiener covariance is built in the observational-error section.
#
# Generation-only dependencies: this script does not plot or run MCMC, so the heavy
# Turing/PyPlot/MCMCChains imports of the original are intentionally dropped.

using LinearAlgebra          # Diagonal, Symmetric (Wiener covariance)
using Distributions          # Normal, MvNormal
using GaussianProcesses      # predict_y / predict_f on the loaded emulators
using JLD2
using CSV
using DataFrames             # DataFrame, select, Not
using PyCall                 # numpy load/save
using Random

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

np = pyimport("numpy")

include(joinpath(@__DIR__, "..", "utils", "scale_utils.jl")); using .ScaleUtils
include(joinpath(@__DIR__, "..", "utils", "gp_utils.jl"));    using .GPUtils


#========================================================================#
# Reproducibility: SEED is the single source (MUST match the notebook's SEED) — it seeds the
# draw-loop RNG (below) and the output serial. Pin BLAS to one thread to avoid thread-race drift.
#========================================================================#
SEED = 9
BLAS.set_num_threads(1)


#========================================================================#
# Experiment toggles — read from FL_* env vars (recalibration.jl reads the SAME
# env, so generator ≡ likelihood by construction).
#
#   FL_ERR_MODEL  = "wiener" (Brownian Σ_o)  |  "indep" (Diagonal(σ_grd_vol²), same marginals)
#   FL_CODE_NOISE = "1" add GP code variance to the draw  |  "0" obs-noise-only draw
#========================================================================#
ERR_MODEL  = get(ENV, "FL_ERR_MODEL", "wiener")
CODE_NOISE = get(ENV, "FL_CODE_NOISE", "1") == "1"
@assert ERR_MODEL in ("wiener", "indep") "FL_ERR_MODEL must be 'wiener' or 'indep', got $(ERR_MODEL)"

# Emulator query for the code-uncertainty draw (must match the likelihood's FL_GP_PREDICT). DEFAULT
# = predict_f (deterministic-simulator convention, latent variance only); FL_GP_PREDICT=y ONLY to
# reproduce the old predict_y behavior.
GP_PRED = get(ENV, "FL_GP_PREDICT", "f") == "y" ? predict_y : predict_f


#========================================================================#
# Inputs — parameter scaling
#========================================================================#
X_raw = CSV.read(joinpath(PROJECT_ROOT, "inputs", "data", "Amery_Input_Parameters_Filtered.csv"), DataFrame)
cols = filter(c -> c != :Column1, Symbol.(names(X_raw)))
println("Column names: ", cols)
X_scaled_t, X_scalers, X_mins, X_maxs = ScaleUtils.get_scaled_matrix(X_raw, cols)
X_scaled = X_scaled_t'


#========================================================================#
# Inputs — future grounded-volume-change time series → per-year output scaling
#========================================================================#
y_grd_vol_raw = CSV.read(joinpath(PROJECT_ROOT, "inputs", "data", "Amery_Grd_Vol_Change_time_series_2300_ssp5.csv"), DataFrame)
y_grd_vol_fut = collect(Matrix(select(y_grd_vol_raw, Not(1))))

gap = 15
years = [y for y in 2015:gap:2300 if y != 2015]     # [2030, 2045, …, 2300]  (19 analysis years)
col_idx = (years .- 2016)
y_grd_vol_sub = y_grd_vol_fut[:, col_idx]

years_all = years
colnames_grd_vol = Symbol.("y_grd_vol_" .* string.(years_all))
df_y_grd_vol_all = DataFrame(hcat(y_grd_vol_sub), colnames_grd_vol)

y_grd_vol_scaled, y_grd_vol_scalers, y_grd_vol_mins, y_grd_vol_maxs =
    ScaleUtils.get_scaled_matrix(df_y_grd_vol_all, colnames_grd_vol)


#========================================================================#
# Canonical pre-trained per-year emulator — LOAD ONLY, never retrain
#========================================================================#
grd_vol_path = joinpath(PROJECT_ROOT, "inputs", "emulators", "cumulative_grounded_volume_change_gap_15_emulators.jld2")
@assert isfile(grd_vol_path) "Canonical emulator not found: $grd_vol_path"
gps_Vol_all = GPUtils.load_emulators(grd_vol_path)
println("Emulated years: ", sort(collect(keys(gps_Vol_all))))


#========================================================================#
# Posterior draws — the SAME 100 fixed draws the notebook uses
#========================================================================#
θ_posterior_expert = np.load(joinpath(PROJECT_ROOT, "inputs", "params", "present_day_2015_posterior_100_draws.npy"))
post = DataFrame(vmThresh_post=θ_posterior_expert[:,1], fricExp_post=θ_posterior_expert[:,2],
    mu_scale_post=θ_posterior_expert[:,3], stiff_scale_post=θ_posterior_expert[:,4],
    gamma0_post=θ_posterior_expert[:,5], melt_flux_post=θ_posterior_expert[:,6])
total_rows = nrow(post)


#========================================================================#
# Observational-error covariance
#
# Per-year Brownian marginal σ:  σ_a·√(year − 2016),  σ_a = 5.720 Gt/yr.
#
# Temporally-correlated (discrete-Wiener / Brownian) covariance for the CUMULATIVE
# grounded-volume observations:
#     Σ_o[i,j] = σ_a² · min(τ_i, τ_j),   τ = year − 2016.
# Its diagonal equals the per-year σ² (marginals unchanged); the off-diagonals supply
# the cross-year correlation the independent-draw model was missing.
#========================================================================#
σ_a = 5.720
years_sig = 2017:2300
sig_grd_vol = DataFrame(year = collect(years_sig), sigma = σ_a .* sqrt.(years_sig .- 2016))

years_selected = years_all[1:end]
n_fut = length(years_selected)
σ_grd_vol = [sig_grd_vol[sig_grd_vol.year .== y, :sigma][1] for y in years_selected]
println("Sigma values for selected years: ", σ_grd_vol)

τ_wiener = years_selected .- 2016
Σ_o = [σ_a^2 * min(τ_wiener[i], τ_wiener[j]) for i in 1:n_fut, j in 1:n_fut]
@assert maximum(abs.(diag(Σ_o) .- σ_grd_vol.^2)) < 1e-6   # diagonal ≡ marginal σ²

gps_grd_vol = [gps_Vol_all[y] for y in years_selected]
Y_gv_mins = [y_grd_vol_mins[Symbol("y_grd_vol_$(y)")] for y in years_selected]
Y_gv_maxs = [y_grd_vol_maxs[Symbol("y_grd_vol_$(y)")] for y in years_selected]


#========================================================================#
# Generate synthetic observations
#========================================================================#
n_draws = 100
@assert n_draws <= total_rows "n_draws ($n_draws) exceeds available posterior rows ($total_rows)."

# The obs-file serial varies per experiment (FL_SERIAL, e.g. "seed9_indep_nocode"); each error-model /
# code-noise experiment writes its own synthetic-obs file. The generating draws are NOT re-saved: the
# NUTS warm-start reads them straight from inputs/params/present_day_2015_posterior_100_draws.npy.
SERIAL = get(ENV, "FL_SERIAL", "seed$(SEED)_wiener_code")

Y_grd_vol_obs = zeros(n_draws, n_fut)

# Re-seed immediately before drawing so this script is idempotent and matches the notebook
# draw-for-draw (any bookkeeping draws above are decoupled).
rng = MersenneTwister(SEED)

for j in 1:n_draws
    x = eachrow(post)[j]

    # Rescale the six inputs into [0,1]:
    p_vm     = (x.vmThresh_post    - X_mins[:vmThresh])   / (X_maxs[:vmThresh]    - X_mins[:vmThresh])
    p_fric   = (x.fricExp_post     - X_mins[:fricExp])    / (X_maxs[:fricExp]     - X_mins[:fricExp])
    p_mu     = (x.mu_scale_post    - X_mins[:mu_scale])   / (X_maxs[:mu_scale]    - X_mins[:mu_scale])
    p_stiff  = (x.stiff_scale_post - X_mins[:stiff_scale])/(X_maxs[:stiff_scale] - X_mins[:stiff_scale])
    p_gamma0 = (x.gamma0_post      - X_mins[:gamma0])     / (X_maxs[:gamma0]      - X_mins[:gamma0])
    p_melt   = (x.melt_flux_post   - X_mins[:melt_flux])  / (X_maxs[:melt_flux]   - X_mins[:melt_flux])

    θ = [p_vm;; p_fric;; p_mu;; p_stiff;; p_gamma0;; p_melt]'    # 1×6 row-vector

    grd_vol_preds = GP_PRED.(gps_grd_vol, Ref(θ))
    μ_gv_raws   = only.([gv[1] for gv in grd_vol_preds])
    var_gv_raws = [gv[2][1] for gv in grd_vol_preds]

    # Un-scale into original units (Gt):
    μ_gv_un = μ_gv_raws .* (Y_gv_maxs .- Y_gv_mins) .+ Y_gv_mins
    s_gv_un = sqrt.(var_gv_raws) .* (Y_gv_maxs .- Y_gv_mins)
    Σ_code  = Diagonal(s_gv_un.^2)

    # Draw the whole trajectory as ONE Gaussian vector.
    #   obs part:  wiener → Σ_o (Brownian);  indep → Diagonal(σ_grd_vol²)  (same marginals)
    #   code part: + Σ_code only when CODE_NOISE  (Diagonal per-year GP predictive variance).
    Σ_obs  = ERR_MODEL == "wiener" ? Σ_o : diagm(σ_grd_vol.^2)
    Σ_draw = CODE_NOISE ? Symmetric(Σ_obs .+ Σ_code) : Symmetric(Σ_obs)
    Y_grd_vol_obs[j, :] = rand(rng, MvNormal(μ_gv_un, Σ_draw))
end

@save joinpath(PROJECT_ROOT, "intermediate_data", "future_observations", "$(SERIAL)_future_observations.jld2") Y_grd_vol_obs
println("Wrote $(SERIAL)_future_observations.jld2 ($(n_draws) draws × $(n_fut) years).")