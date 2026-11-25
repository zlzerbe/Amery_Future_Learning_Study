# present_day_calibration.jl — year-2015 calibration of the six Amery ice-shelf parameters.
# Trains the three present-day observable emulators from the ensemble CSV, saves them, then runs the
# NUTS calibration (predict_f code uncertainty) and writes the posterior plus two 100-draw subsets.

using CSV, DataFrames, JLD2, Random, Statistics, Printf
using Turing, Distributions, GaussianProcesses, LinearAlgebra, Optim
using PyCall

BLAS.set_num_threads(1)   # bit-reproducibility must not depend on the caller's thread count (cf. AmeryFutureObservations.jl)

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(@__DIR__, "..", "utils", "scale_utils.jl")); using .ScaleUtils
include(joinpath(@__DIR__, "..", "utils", "trapezoidal.jl"))
const np = pyimport("numpy")

const IN_COLS  = [:vmThresh, :fricExp, :mu_scale, :stiff_scale, :gamma0, :melt_flux]
const OUT_COLS = [:Calving_Front_Position, :Grounding_Line_Position, :Mass_Change]
const σ_obs    = [21.95, 7.540, 5.721]         # observation SDs: calving front, grounding line, mass change
const OBSERVED = [-11.94, 0.9877, -1.656]      # year-2015 observed values
const N_SAMPLES = parse(Int, get(ENV, "FL_PD_NSAMPLES", "100000"))
const BURNIN    = parse(Int, get(ENV, "FL_PD_BURNIN",   "50000"))
const SEED      = parse(Int, get(ENV, "FL_PD_SEED",     "9"))
const SAVE_FULL = get(ENV, "FL_PD_SAVE_FULL", "0") == "1"   # also write the full N_SAMPLES×6 posterior (off by default)

# ── data + [0,1] scaling from the ensemble CSV ──
data = CSV.read(joinpath(ROOT, "inputs", "data", "Amery_RELX_Ensemble_Filtered.csv"), DataFrame)
Xsc, _, in_min,  in_max  = ScaleUtils.get_scaled_matrix(data, IN_COLS)    # N×6
Ysc, _, out_min, out_max = ScaleUtils.get_scaled_matrix(data, OUT_COLS)   # N×3
const X       = collect(Xsc')                                             # 6×N (GP orientation)
const IN_MIN  = [in_min[c]  for c in IN_COLS];  const IN_MAX  = [in_max[c]  for c in IN_COLS]
const OUT_MIN = [out_min[c] for c in OUT_COLS]; const OUT_MAX = [out_max[c] for c in OUT_COLS]

# ── train + save the three emulators (Matérn-5/2 ARD, linear mean, fixed 0.01 nugget) ──
# Fit by doubling the optimizer budget 100→200→400→800, retraining from scratch until Optim reports
# convergence (errors if 800 iterations still don't converge).
function train_emulator(y; base_iters=100, max_iters=800)
    iters = base_iters
    while true
        gp  = GaussianProcesses.GP(X, Float64.(y), MeanLin(zeros(6)), Mat52Ard(zeros(6), 0.001), -4.605170185985)
        res = GaussianProcesses.optimize!(gp; domean=true, kern=true, noise=false, iterations=iters)
        Optim.converged(res) && return gp
        iters *= 2
        iters > max_iters && error("emulator failed to converge within $max_iters iterations")
    end
end
const gp_cal  = train_emulator(Ysc[:, 1])
const gp_grd  = train_emulator(Ysc[:, 2])
const gp_mass = train_emulator(Ysc[:, 3])
const GPS     = (gp_cal, gp_grd, gp_mass)

const EMU_DIR = joinpath(ROOT, "inputs", "emulators"); mkpath(EMU_DIR)
jldsave(joinpath(EMU_DIR, "calving_front_emulator.jld2");  calving_front_emulator  = gp_cal)
jldsave(joinpath(EMU_DIR, "grounding_line_emulator.jld2"); grounding_line_emulator = gp_grd)
jldsave(joinpath(EMU_DIR, "mass_balance_emulator.jld2");   mass_balance_emulator   = gp_mass)

# ── calibration model ──
@model function present_day(Y)
    vmThresh    ~ Truncated(Normal(130000, 50000/3), 80000, 180000)
    fricExp     ~ TrapezoidalDist(0.1, 0.15, 0.28, 0.333)
    mu_scale    ~ Truncated(Normal(1.0, 0.1), 0.8, 1.2)
    stiff_scale ~ Truncated(Normal(1.0, 0.1), 0.8, 1.2)
    gamma0      ~ Truncated(LogNormal(10, 1), 9620, 471000)
    melt_flux   ~ Truncated(Normal(35, 11.5), 12, 58)

    p = ([vmThresh, fricExp, mu_scale, stiff_scale, gamma0, melt_flux] .- IN_MIN) ./ (IN_MAX .- IN_MIN)
    θ = reshape(p, 6, 1)
    for i in 1:3
        μ, v = predict_f(GPS[i], θ)                      # latent GP variance only (no nugget)
        mu   = only(μ) * (OUT_MAX[i] - OUT_MIN[i]) + OUT_MIN[i]
        sig  = sqrt(only(v)) * (OUT_MAX[i] - OUT_MIN[i])
        Y[i] ~ Normal(mu, sqrt(σ_obs[i]^2 + sig^2))
    end
end

# ── sample; save the 100-draw subset (+ the full posterior when FL_PD_SAVE_FULL=1) ──
Random.seed!(SEED)
chain = sample(present_day(OBSERVED), NUTS(), MCMCSerial(), N_SAMPLES, 1; discard_initial=BURNIN, progress=true)
display(chain)

θ_post = Matrix(DataFrame(chain)[!, IN_COLS])            # N×6, physical units, canonical column order

# 100-draw subset (seeded random): justified by the chain's very high ESS (ESS/N ≈ 0.5–0.8,
# τ ≈ 1.5 iters) — the draws are already near-independent, so a random 100 is as decorrelated as a
# thinned 100. Seeded, so the selection is itself reproducible. Written straight to inputs/params —
# the location the downstream future-obs generator + four-configuration calibration consume. The full
# posterior / chain are not persisted: the 100 draws are all downstream needs for now.
n = size(θ_post, 1)
rand_idx = randperm(MersenneTwister(SEED), n)[1:100]
draws_path = joinpath(ROOT, "inputs", "params", "present_day_2015_posterior_100_draws.npy")
np.save(draws_path, θ_post[rand_idx, :])

# Optional full-posterior dump (env FL_PD_SAVE_FULL=1). Off by default so the shipped output stays the
# lean 100-draw subset; the full N_SAMPLES×6 chain is deterministic (seed SEED) so this reproduces it.
if SAVE_FULL
    full_path = joinpath(ROOT, "inputs", "params", "present_day_2015_posterior_samples.npy")
    np.save(full_path, θ_post)
    println("full posterior $(size(θ_post,1))×$(size(θ_post,2)) → $(full_path)")
end

@printf("\n%-12s %14s %14s %12s\n", "param", "mean", "sd", "ess")
ss = DataFrame(summarystats(chain))
for c in IN_COLS
    i = findfirst(==(c), Symbol.(ss.parameters))
    @printf("%-12s %14.5g %14.5g %12.1f\n", String(c), ss.mean[i], ss.std[i], ss.ess[i])
end
println("\n100 random draws from a $(size(θ_post,1))-sample chain → $(draws_path)")
println("DONE")