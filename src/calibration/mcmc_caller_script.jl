# ── mcmc_caller_script.jl ──
#
# Per-task calibration driver: run one NUTS calibration of the six Amery ice-shelf
# parameters against a single synthetic future-observation realization.
#
# Invoked once per SLURM array task:
#     julia mcmc_caller_script.jl <job_id> <realization_id> <saveout_directory>
#
#   job_id          number of future years to assimilate, 0..18 (the sequential-
#                   learning knob). job_id = i selects years 2030 : 15 : (2030 + 15i).
#   realization_id  which synthetic obs draw to calibrate against (row of the
#                   FL_SERIAL obs file), 1-based.
#   saveout_dir     output root; chains land in <dir>/chains_dir and generated
#                   quantities in <dir>/gq_dir.
#
# The obs file is selected by FL_SERIAL (default "seed9_wiener_code" = run B); each error-model /
# code-noise experiment points at its own generated trajectories. The driver loads
# the per-year GP emulators and present-day scalers, warm-starts NUTS from the
# matching seed-9 present-day posterior draw (θ_init_scaled), then hands everything
# to amery_calibration_prediction.calibration_func (which writes the chain and the
# generated quantities to disk).

using CSV, DataFrames, JLD2, Random, LinearAlgebra, PyCall

BLAS.set_num_threads(1)   # bit-reproducibility must not depend on the caller's thread count (cf. AmeryFutureObservations.jl)

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "..", "utils", "scale_utils.jl"));   using .ScaleUtils
include(joinpath(@__DIR__, "..", "utils", "gp_utils.jl"));      using .GPUtils
include(joinpath(@__DIR__, "calibration_prediction.jl"));       using .amery_calibration_prediction

Random.seed!(9)   # seed the global RNG NUTS draws from, for reproducibility (seed 9 everywhere)


#========================================================================#
# Command-line arguments (one SLURM array task)
#========================================================================#
if length(ARGS) < 3
    error("Usage: julia mcmc_caller_script.jl <job_id> <realization_id> <saveout_directory>")
end

job_idx           = parse(Int, ARGS[1])                 # 0..18: number of future years
years_selected    = [i * 15 + 2030 for i in 0:job_idx]  # 2030, 2045, … the assimilated years
realization_id    = parse(Int, ARGS[2])                 # which synthetic obs draw (1-based)
saveout_directory = ARGS[3]

chains_dir = joinpath(saveout_directory, "chains_dir")
gq_dir     = joinpath(saveout_directory, "gq_dir")
isdir(chains_dir) || mkdir(chains_dir)
isdir(gq_dir)     || mkdir(gq_dir)


#========================================================================#
# Fixed analysis grid + present-day (2015) observation scalers
#========================================================================#
future_years = [2030, 2045, 2060, 2075, 2090, 2105, 2120, 2135, 2150,
                2165, 2180, 2195, 2210, 2225, 2240, 2255, 2270, 2285, 2300]

# Min/max scalers for the present-day calving-front (Yc), grounding-line (Yg),
# and mass-balance (Ym) emulator outputs.
Yc_mins, Yc_maxs = [-96.34022417], [74.9385927]
Yg_mins, Yg_maxs = [-21.10306973], [19.2289032]
Ym_mins, Ym_maxs = [-23.91989209], [21.04901554]


#========================================================================#
# Calibration reusables: scaled design, present-day + volume emulators,
# per-year volume uncertainties
#========================================================================#
# Parameter design → the six parameters' min/max for the [0,1] calibration transform. Reads the
# SAME full-precision CSV as the generator (AmeryFutureObservations.jl), so the input scaling is
# identical → generator ≡ likelihood exactly. (Amery_RELX_Ensemble_Filtered.csv is the rounded
# reference copy of this design that also carries the 3 present-day observable columns.)
X_raw = CSV.read(joinpath(PROJECT_ROOT, "inputs", "data", "Amery_Input_Parameters_Filtered.csv"), DataFrame)
cols  = filter(c -> c != :Column1, Symbol.(names(X_raw)))
_, _, X_min_dict, X_max_dict = ScaleUtils.get_scaled_matrix(X_raw, cols)

# Present-day (2015) calving-front, grounding-line, and mass-balance emulators. Only the 2015
# emulator of each is ever used (the calibration constrains the present-day observables), so each
# artifact holds a SINGLE GP stored under the variable name <variable>_emulator. The one-element
# vector wrapper is kept because the likelihood broadcasts predict_f over these.
gps_cal  = [JLD2.load(joinpath(PROJECT_ROOT, "inputs", "emulators", "calving_front_emulator.jld2"),  "calving_front_emulator")]
gps_grd  = [JLD2.load(joinpath(PROJECT_ROOT, "inputs", "emulators", "grounding_line_emulator.jld2"), "grounding_line_emulator")]
gps_mass = [JLD2.load(joinpath(PROJECT_ROOT, "inputs", "emulators", "mass_balance_emulator.jld2"),   "mass_balance_emulator")]

# Cumulative grounded-volume change: per-year min/max (sorted by year), the per-year
# σ table (σ_a·√τ), and the 19 per-year volume emulators.
y_vol_mins, y_vol_maxs = JLD2.load(joinpath(PROJECT_ROOT, "inputs", "data", "grd_vol_change_mins_maxs.jld2"), "y_grd_vol_mins", "y_grd_vol_maxs")
Yvol_mins = collect(values(sort(y_vol_mins)))
Yvol_maxs = collect(values(sort(y_vol_maxs)))
vol_uncertainties = CSV.read(joinpath(PROJECT_ROOT, "inputs", "data", "sigma_grd_vol_change.csv"), DataFrame)
gps_Vol_all = GPUtils.load_emulators(joinpath(PROJECT_ROOT, "inputs", "emulators", "cumulative_grounded_volume_change_gap_15_emulators.jld2"))


#========================================================================#
# Future observations for this realization + NUTS warm-start
#========================================================================#
# Obs-file serial from FL_SERIAL (default seed9_wiener_code = run B). Each experiment (A/C/D,
# ±code noise) points at its own generated trajectories; the warm-start stays seed9-data.
obs_serial     = get(ENV, "FL_SERIAL", "seed9_wiener_code")
all_future_obs = JLD2.load(joinpath(PROJECT_ROOT, "intermediate_data", "future_observations", "$(obs_serial)_future_observations.jld2"), "Y_grd_vol_obs")
future_obs     = all_future_obs[realization_id, :]

# Present-day posterior draw for this realization (rescaled to [0,1]) — NUTS warm-start. Read straight
# from the single source of truth (the 100 physical draws) and apply the SAME min–max scaling the
# generator used / the likelihood inverts — no separate emulator_pred cache to fall out of sync.
_draws = convert(Matrix{Float64}, pyimport("numpy").load(joinpath(PROJECT_ROOT, "inputs", "params", "present_day_2015_posterior_100_draws.npy")))
_row   = _draws[realization_id, :]                       # physical, column order [vmThresh…melt_flux]
_ord   = [:vmThresh, :fricExp, :mu_scale, :stiff_scale, :gamma0, :melt_flux]
θ_init_scaled = [(_row[k] - X_min_dict[p]) / (X_max_dict[p] - X_min_dict[p]) for (k, p) in enumerate(_ord)]


#========================================================================#
# Run the calibration (chain + generated quantities are written to disk)
#========================================================================#
amery_calibration_prediction.calibration_func(future_obs,
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
                                              gps_Vol_all,
                                              θ_init_scaled)