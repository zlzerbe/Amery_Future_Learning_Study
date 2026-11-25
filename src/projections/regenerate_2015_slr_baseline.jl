# regenerate_2015_slr_baseline.jl — regenerate the year-2015 SLR baselines
# (present2015_post_mm_ssp5, 285×5000, mm) by pushing 5000 systematically-thinned draws of the
# present-day (2015) posterior through the VAF PCA GP emulator. Writes TWO baselines, one per
# projection convention, so each configuration's 2015 baseline matches its own finals:
#   • 2015_SLR_Projections_codeon.jld2  — emulator code uncertainty ON  (PC ~ Normal(mean, σ)),
#       used by configurations A and B (code-on projections).
#   • 2015_SLR_Projections_codeoff.jld2 — emulator code uncertainty OFF (PC = GP mean; matches
#       appendix Fig 8), used by configurations C and D (code-off projections).
# Each mirrors `project_realization` (Amery_Future_Learning_Plots.ipynb) for its convention. Run
# ONCE; rerun whenever inputs/params/present_day_2015_posterior_samples.npy changes:
#     julia --project=. src/projections/regenerate_2015_slr_baseline.jl
using JLD2, CSV, DataFrames, Random, LinearAlgebra, Statistics, GaussianProcesses, MultivariateStats, Distributions, PyCall
include(joinpath(@__DIR__, "..", "utils", "scale_utils.jl")); using .ScaleUtils

const ROOT   = normpath(joinpath(@__DIR__, "..", ".."))
const PARAMS = [:vmThresh, :fricExp, :mu_scale, :stiff_scale, :gamma0, :melt_flux]
const N_KEEP = 5000        # posterior draws pushed through VAF (systematic thinning)
const GT2MM  = -362        # VAF(Gt) → SLR(mm) sign/scale convention (matches project_realization)

# input scaling — same X_mins/X_maxs as the calibration / generator / notebook
X_raw = CSV.read(joinpath(ROOT, "inputs", "data", "Amery_Input_Parameters_Filtered.csv"), DataFrame)
_, _, X_mins, X_maxs = ScaleUtils.get_scaled_matrix(X_raw, filter(c -> c != :Column1, Symbol.(names(X_raw))))
param_mins = [X_mins[p] for p in PARAMS]
param_maxs = [X_maxs[p] for p in PARAMS]

# VAF PCA GP emulator (5 per-PC GPs + 285-step PCA basis); PCA object is stored under key "p"
gps, PC_min, PC_max, pca, num_pcs = JLD2.load(joinpath(ROOT, "inputs", "emulators", "VAF_PCA_GP_Emulator.jld2"),
    "gps", "PC_min", "PC_max", "p", "num_pcs")
predict_gp_eml(gp, X) = (mv = GaussianProcesses.predict_f(gp, X); (mv[1], sqrt.(mv[2])))
reconstruct_pcs(pca, W, J=length(W)) =
    MultivariateStats.decentralize(MultivariateStats.projection(pca)[:, 1:J] * W, mean(pca))

# present-day posterior (100k physical, PARAMS order) → systematic thin to 5000
chain = convert(Matrix{Float64}, pyimport("numpy").load(joinpath(ROOT, "inputs", "params", "present_day_2015_posterior_samples.npy")))
step  = max(1, fld(size(chain, 1), N_KEEP))
draws = chain[1:step:end, :]
size(draws, 1) > N_KEEP && (draws = draws[1:N_KEEP, :])
αs = (draws .- param_mins') ./ (param_maxs' .- param_mins')          # N×6 scaled to [0,1]
ntime = size(MultivariateStats.projection(pca), 1)                  # 285

# code-OFF baseline — GP posterior MEAN per PC (no Normal draw); consumes no RNG.
base_codeoff = Matrix{Float64}(undef, ntime, size(draws, 1))
for s in 1:size(αs, 1)
    α  = reshape(αs[s, :], :, 1)
    pc = [ only(predict_gp_eml(gps[i], α)[1]) * (PC_max[i] - PC_min[i]) + PC_min[i] for i in 1:num_pcs ]
    base_codeoff[:, s] = reconstruct_pcs(pca, pc, num_pcs)
end
base_codeoff ./= GT2MM

# code-ON baseline — PC ~ Normal(mean, σ) (same recipe/seed as project_realization's code-on path).
Random.seed!(11)                                                    # reproducible PC code-noise draws
base_codeon = Matrix{Float64}(undef, ntime, size(draws, 1))
for s in 1:size(αs, 1)
    α  = reshape(αs[s, :], :, 1)
    pc = [ let (μ, σ) = predict_gp_eml(gps[i], α)
               rand(Normal(only(μ) * (PC_max[i] - PC_min[i]) + PC_min[i],
                           only(σ) * (PC_max[i] - PC_min[i])))
           end for i in 1:num_pcs ]
    base_codeon[:, s] = reconstruct_pcs(pca, pc, num_pcs)
end
base_codeon ./= GT2MM

let present2015_post_mm_ssp5 = base_codeon
    @save joinpath(ROOT, "inputs", "params", "2015_SLR_Projections_codeon.jld2") present2015_post_mm_ssp5
end
let present2015_post_mm_ssp5 = base_codeoff
    @save joinpath(ROOT, "inputs", "params", "2015_SLR_Projections_codeoff.jld2") present2015_post_mm_ssp5
end
println("Wrote 2015_SLR_Projections_{codeon,codeoff}.jld2  ($(size(base_codeon)), mm, [year, sample])")
