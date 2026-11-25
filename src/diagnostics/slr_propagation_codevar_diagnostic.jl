# slr_propagation_codevar_diagnostic.jl — DIAGNOSTIC ONLY (no calibration/NUTS/trajectory rerun).
#
# Question: in the EXISTING code-on configurations A (indep_code) and B (wiener_code), how much of the
# projected sea-level-contribution (SLE) 90% CI width comes from the PCA-GP VAF emulator's
# predictive/code variance in the FINAL propagation step, versus the parameter posterior itself?
#
# Method (holds the A/B posteriors FIXED; changes ONLY the last PCA-GP propagation step):
#   For each existing A/B posterior draw θ at each trajectory r and analysis year yr, we form the
#   projected SLE sample two ways, sharing the SAME θ draws and the SAME GP predict_f means:
#     • code-on  : PC_i ~ Normal(m_i(θ), σ_i(θ))         (exactly project_realization, cfg.code=true)
#     • prop-off : PC_i =        m_i(θ)                   (exactly project_realization, cfg.code=false,
#                                                          i.e. the operation configurations C/D already use)
#   width = quantile(.95) − quantile(.05) at the prediction-year row, averaged across trajectories.
#   Because code-on = prop-off + independent emulator noise on the SAME θ/means, the pair is an exact
#   conditional comparison and their gap isolates the emulator propagation variance.
#
# It does NOT touch: the scalar grounded-mass / 3 present-day GP code uncertainty used in
# generating+calibrating A/B, the synthetic trajectories, or the NUTS posteriors. It only re-runs the
# final VAF PCA-GP propagation, from the cached A/B posterior draws, into a NEW output dir.
#
# Run:
#   julia --project=. src/diagnostics/slr_propagation_codevar_diagnostic.jl [MAXR]
#   (MAXR optional: limit to first MAXR realizations for a quick smoke test; default all 100.)

using JLD2, CSV, DataFrames, Random, LinearAlgebra, Statistics,
      GaussianProcesses, MultivariateStats, Distributions
include(joinpath(@__DIR__, "..", "utils", "scale_utils.jl")); using .ScaleUtils

const ROOT     = normpath(joinpath(@__DIR__, "..", ".."))
const PARAMS   = [:vmThresh, :fricExp, :mu_scale, :stiff_scale, :gamma0, :melt_flux]
const GAPPED   = collect(2030:15:2300)                 # 19 analysis years
const PRED     = [2100, 2200, 2300]                    # horizons
const ALLYEARS = collect(2016:2300)                    # 285 projection steps
const ROW      = Dict(py => findfirst(==(py), ALLYEARS) for py in PRED)   # 85,185,285
const NKEEP    = 5000                                  # PROJ_SAMPLE_SIZE (same thinning as pipeline)
const GT2MM    = -362
const OUTDIR   = joinpath(ROOT, "outputs", "diagnostics", "slr_propagation_codevar")
cal_range(py)  = collect(range(2015, step=15, length=fld(py - 2015, 15) + 1))   # analysis years ≤ py
MAXR = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 100
REAL = 1:MAXR

# ── shared setup: identical scaling + emulator load as the notebook / regenerate script ──────────
X_raw = CSV.read(joinpath(ROOT, "inputs", "data", "Amery_Input_Parameters_Filtered.csv"), DataFrame)
_, _, X_mins, X_maxs = ScaleUtils.get_scaled_matrix(X_raw, filter(c -> c != :Column1, Symbol.(names(X_raw))))
param_mins = [X_mins[p] for p in PARAMS]
param_maxs = [X_maxs[p] for p in PARAMS]

gps, PC_min, PC_max, pca, num_pcs = JLD2.load(joinpath(ROOT, "inputs", "emulators", "VAF_PCA_GP_Emulator.jld2"),
    "gps", "PC_min", "PC_max", "p", "num_pcs")
const J     = num_pcs
const BASIS = MultivariateStats.projection(pca)[:, 1:J]        # 285 × J loadings
const PMEAN = MultivariateStats.mean(pca)                       # 285 vector
Brows(py)   = BASIS[ROW[py], :]                                # J vector (one prediction-year row)
const BR    = Dict(py => Brows(py) for py in PRED)
const MR    = Dict(py => PMEAN[ROW[py]] for py in PRED)

width(v) = quantile(v, 0.95) - quantile(v, 0.05)

# thin a posterior chain (iters×6) to ≤NKEEP, exactly as project_realization
function thin(chain)
    step  = max(1, fld(size(chain, 1), NKEEP))
    draws = chain[1:step:end, :]
    size(draws, 1) > NKEEP && (draws = draws[1:NKEEP, :])
    draws
end

# GP predict_f means (unscaled PC) and scaled SDs for all samples at once → (Wmean N×J, Ssd N×J)
function pc_mean_sd(draws)
    αs = (draws .- param_mins') ./ (param_maxs' .- param_mins')        # N×6 in [0,1]
    N  = size(αs, 1)
    Wmean = Matrix{Float64}(undef, N, J); Ssd = Matrix{Float64}(undef, N, J)
    Xt = permutedims(αs)                                              # 6×N
    for i in 1:J
        μ, v = GaussianProcesses.predict_f(gps[i], Xt)               # vectors length N
        Wmean[:, i] = μ .* (PC_max[i] - PC_min[i]) .+ PC_min[i]
        Ssd[:,   i] = sqrt.(v) .* (PC_max[i] - PC_min[i])
    end
    Wmean, Ssd
end

# SLE(mm) samples at prediction-year `py` for prop-off (means) and code-on (means + emulator noise Z)
function sle_row(Wmean, Ssd, Z, py)
    b   = BR[py]; m0 = MR[py]
    off = (Wmean * b .+ m0) ./ GT2MM                                 # N   (prop-off)
    on  = ((Wmean .+ Ssd .* Z) * b .+ m0) ./ GT2MM                   # N   (code-on)
    # per-sample emulator variance contributed at this row (Var over Z of the added term)
    ev  = ((Ssd .^ 2) * (b .^ 2)) ./ (GT2MM^2)                       # N
    off, on, ev
end

# ── VERIFICATION 1: prop-off == the stored C/D projection (deterministic, no RNG) ────────────────
function verify_codeoff_matches_CD(; scen="indep_nocode", r=1, yr=2300, tol=1e-6)
    post = JLD2.load(joinpath(ROOT, "outputs", "posteriors_$(scen)", "R_$(r)_Posterior_Dict.jld2"), "post_data")
    haskey(post, yr) || (return (ok=false, msg="no year $yr for $scen R$r"))
    Wmean, _ = pc_mean_sd(thin(post[yr]))
    mine = Dict(py => (Wmean * BR[py] .+ MR[py]) ./ GT2MM for py in PRED)      # my prop-off rows
    stored = JLD2.load(joinpath(ROOT, "outputs", "projections_$(scen)", "R_$(r)", "$(r)-year$(yr)pred_VAF.jld2"),
                       "sample_post_mm_ssp5")
    maxabs = maximum(py -> maximum(abs.(mine[py] .- stored[ROW[py], :])), PRED)
    (ok = maxabs < tol, maxabs = maxabs, scen=scen, r=r, yr=yr)
end

# ── main sweep over A and B ─────────────────────────────────────────────────────────────────────
SCEN = [(:A, "indep_code"), (:B, "wiener_code")]
# width_on[scen][py], width_off[scen][py]  :: (nreal × ncal) matrices, col1 = 2015 baseline
results = Dict{Symbol,Any}()

# 2015 baselines (per convention). code-on baseline for A/B; also load code-off baseline for reference.
base_on  = JLD2.load(joinpath(ROOT, "inputs", "params", "2015_SLR_Projections_codeon.jld2"),  "present2015_post_mm_ssp5")
base_off = JLD2.load(joinpath(ROOT, "inputs", "params", "2015_SLR_Projections_codeoff.jld2"), "present2015_post_mm_ssp5")
base_on_w  = Dict(py => width(base_on[ROW[py],  :]) for py in PRED)
base_off_w = Dict(py => width(base_off[ROW[py], :]) for py in PRED)

for (sym, tag) in SCEN
    pdir = joinpath(ROOT, "outputs", "posteriors_$(tag)")
    won  = Dict(py => fill(NaN, length(REAL), length(cal_range(py))) for py in PRED)
    woff = Dict(py => fill(NaN, length(REAL), length(cal_range(py))) for py in PRED)
    vparam = Dict(py => fill(NaN, length(REAL), length(cal_range(py))) for py in PRED)  # Var of prop-off
    vemul  = Dict(py => fill(NaN, length(REAL), length(cal_range(py))) for py in PRED)  # mean emulator Var
    for py in PRED, i in 1:length(REAL)          # col 1 = 2015 baseline (same for all r)
        won[py][i, 1] = base_on_w[py]; woff[py][i, 1] = base_off_w[py]
    end
    Random.seed!(11)                              # reproducible emulator-noise draws for our code-on
    for (i, r) in enumerate(REAL)
        f = joinpath(pdir, "R_$(r)_Posterior_Dict.jld2")
        isfile(f) || (@warn "missing $f"; continue)
        post = JLD2.load(f, "post_data")
        for yr in GAPPED
            haskey(post, yr) || continue
            Wmean, Ssd = pc_mean_sd(thin(post[yr]))
            Z = randn(size(Wmean))                # emulator noise ~ N(0,1), added ONLY in code-on
            for py in PRED
                yr ≤ py || continue               # mirror build_slr_widths (analysis year ≤ horizon)
                c = findfirst(==(yr), cal_range(py)); c === nothing && continue
                off, on, ev = sle_row(Wmean, Ssd, Z, py)
                woff[py][i, c] = width(off); won[py][i, c] = width(on)
                vparam[py][i, c] = var(off); vemul[py][i, c] = mean(ev)
            end
        end
        i % 10 == 0 && println("  $(sym): done $i/$(length(REAL)) realizations")
    end
    results[sym] = (won=won, woff=woff, vparam=vparam, vemul=vemul, tag=tag)
    println("Finished configuration $sym ($tag).")
end

# ── cross-check: our recomputed code-on final-year mean widths vs the pipeline's cached widths ──
function cached_final_mean(tag, py)
    f = joinpath(ROOT, "outputs", "projections_$(tag)", "slr_learning_widths.jld2")
    isfile(f) || return NaN
    sw = JLD2.load(f, "slr_widths")            # Dict py => 100×ncal
    haskey(sw, py) || return NaN
    col = filter(!isnan, sw[py][:, end]); isempty(col) ? NaN : mean(col)
end

# ── summarize + write ────────────────────────────────────────────────────────────────────────────
# All reported quantities are CONDITIONAL on the fixed code-on A/B posterior and isolate ONLY the
# PCA-GP VAF emulator predictive variance in the FINAL propagation step. Per (configuration, horizon), at
# that horizon's final analysis year, across trajectories:
#   mean_width_codeon / mean_width_propoff — across-trajectory mean 90% CI width (mm), propagation
#                                            code variance ON / OFF.
#   relative_width_reduction_from_prop_code = 1 − propoff/codeon — the RELATIVE reduction in the 90%
#                                            width from switching the propagation code variance off
#                                            (a relative reduction, NOT a fraction of the CI width).
#   propagation_variance_fraction = Vemul / Vtotal, from the law of total variance on the conditional
#                                            predictive distribution: Vparam = Var_θ[g(θ)] (prop-off
#                                            sample variance), Vemul = E_θ[s_c²(θ)] (mean per-draw
#                                            emulator variance), Vtotal = Vparam + Vemul; averaged
#                                            across trajectories.
mkpath(OUTDIR)
meanskip(v) = (x = filter(!isnan, v); isempty(x) ? NaN : mean(x))
rows = DataFrame(config=String[], horizon=Int[], analysis_year=Int[],
                 mean_width_codeon=Float64[], mean_width_propoff=Float64[],
                 diff=Float64[], ratio_off_over_on=Float64[],
                 relative_width_reduction_from_prop_code=Float64[],
                 propagation_variance_fraction=Float64[], baseline_codeon=Float64[],
                 codeon_cached_pipeline=Float64[])
for (sym, tag) in SCEN
    R = results[sym]
    for py in PRED
        finalyr = cal_range(py)[end]
        c = length(cal_range(py))                         # final analysis-year column
        on  = meanskip(R.won[py][:, c]); off = meanskip(R.woff[py][:, c])
        vp  = meanskip(R.vparam[py][:, c]); ve = meanskip(R.vemul[py][:, c])
        push!(rows, (string(sym), py, finalyr, on, off, on - off,
                     off / on, 1 - off / on, ve / (vp + ve),
                     base_on_w[py], cached_final_mean(tag, py)))
    end
end
CSV.write(joinpath(OUTDIR, "summary_finalyear.csv"), rows)
@save joinpath(OUTDIR, "curves.jld2") results base_on_w base_off_w
open(joinpath(OUTDIR, "summary_finalyear.json"), "w") do io
    println(io, "{")
    for (k, r) in enumerate(eachrow(rows))
        println(io, "  \"$(r.config)_$(r.horizon)\": {",
            "\"analysis_year\": $(r.analysis_year), ",
            "\"mean_width_codeon\": $(round(r.mean_width_codeon, sigdigits=6)), ",
            "\"mean_width_propoff\": $(round(r.mean_width_propoff, sigdigits=6)), ",
            "\"diff\": $(round(r.diff, sigdigits=6)), ",
            "\"ratio_off_over_on\": $(round(r.ratio_off_over_on, sigdigits=6)), ",
            "\"relative_width_reduction_from_prop_code\": $(round(r.relative_width_reduction_from_prop_code, sigdigits=6)), ",
            "\"propagation_variance_fraction\": $(round(r.propagation_variance_fraction, sigdigits=6))}",
            k < nrow(rows) ? "," : "")
    end
    println(io, "}")
end

println("\n================ VERIFICATION ================")
v = verify_codeoff_matches_CD()
println("V1 prop-off == stored C/D projection (indep_nocode R1 y2300): ok=$(v.ok)  max|Δ|=$(v.maxabs)")

println("\n================ SUMMARY (final analysis year per horizon; across-trajectory mean 90% CI width, mm) ================")
println(rpad("config",10), rpad("horiz",7), rpad("yr",6),
        rpad("code-on",11), rpad("prop-off",11), rpad("Δwidth",10),
        rpad("off/on",9), rpad("rel_W_reduc",13), rpad("prop_var_frac",14), "codeon_cached")
for r in eachrow(rows)
    println(rpad(r.config,10), rpad(r.horizon,7), rpad(r.analysis_year,6),
            rpad(round(r.mean_width_codeon,digits=3),11), rpad(round(r.mean_width_propoff,digits=3),11),
            rpad(round(r.diff,digits=3),10), rpad(round(r.ratio_off_over_on,digits=3),9),
            rpad(round(1-r.ratio_off_over_on,digits=3),13), rpad(round(r.propagation_variance_fraction,digits=3),14),
            round(r.codeon_cached_pipeline,digits=3))
end
println("\nWrote: $(joinpath(OUTDIR, "summary_finalyear.csv")), summary_finalyear.json, curves.jld2")