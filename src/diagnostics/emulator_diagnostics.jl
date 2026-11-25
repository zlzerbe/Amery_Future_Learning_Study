# ── emulator_diagnostics.jl ──
#
# Out-of-sample K-fold cross-validation of the 19 per-year grounded-volume GP emulators
# (2030, 2045, …, 2300). For each year it refits a fresh GP on each training fold (never the
# pre-fit full-data emulator), predicts the held-out rows, renders the 6-panel diagnostic used for
# appendix Figures A1/A2 (Residuals-vs-Fitted, True-vs-Fitted, four residual-coloured parameter
# scatters + a shared colorbar), and records CV RMSE / nRMSE / R² per year.
#
# Figures → outputs/plots/emulator_diagnostics/emulator_cv_grd_vol_<year>.pdf
# Summary → outputs/plots/emulator_diagnostics/cv_summary.csv
#
# GP recipe: MeanLin + Mat52Ard(0.001) kernel, obs-noise fixed at log(0.01) (not trained),
# optimize! for 100 iters, doubling up to N_RETRIES times on non-convergence.
# Folds: sklearn KFold(n_splits=N_FOLDS, shuffle=true, random_state=42), reproduced via numpy so
# they match the published A1/A2 panels. Both input and output scalers are fit on the TRAIN fold
# only (no CV leakage). Inputs = the 6-parameter design (Amery_Input_Parameters_Filtered.csv, same
# [0,1] transform as the calibration); outputs = the grounded-volume series (column y-2016, Gt).
#
# Run from the repo root (Python deps: numpy/scipy/pandas/matplotlib/seaborn/scienceplots):
#   julia --project=. src/diagnostics/emulator_diagnostics.jl
#   FL_DIAG_YEARS=2300 (comma-separated) runs a subset; otherwise all 19.

using CSV
using DataFrames
using StatsBase
using Statistics
using Distributions
using GaussianProcesses
using Optim
using Printf
using PyCall
using LinearAlgebra
BLAS.set_num_threads(1)   # bit-reproducibility must not depend on the caller's thread count
using PyPlot

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "..", "utils", "scale_utils.jl"))
using .ScaleUtils


#========================================================================#
# Top-of-file constants
#========================================================================#
# 19 analysis years: 2015:15:2300 excluding 2015 → 2030, 2045, …, 2300.
const GAP        = 15
const ALL_YEARS  = [y for y in 2015:GAP:2300 if y != 2015]

# CV fold count. The published appendix figures A1/A2 use 5-fold CV (see the header note);
# change here only to compare other fold counts.
const N_FOLDS    = 5
const KF_SEED    = 42          # sklearn KFold random_state (+ np global seed)
const KF_SHUFFLE = true

# GP convergence handling.
const BASE_ITERS = 100
const N_RETRIES  = 3
const LOG_OBS_NOISE = -4.605170185985   # log(0.01); noise held constant

# Residual COLOUR-scale cap = this percentile of |residual| (outliers saturate to the colormap
# extremes so smaller residuals show more contrast, per appendix Fig A1/A2). Override: FL_DIAG_CLIPQ.
const CLIP_Q     = parse(Float64, get(ENV, "FL_DIAG_CLIPQ", "0.95"))

# Parameter columns, in the canonical pipeline order, with the display labels used in the
# notebook panels (C_μ=mu_scale, q=fricExp, C_φ=stiff_scale, σ_max=vmThresh, m̄=melt_flux,
# γ_0=gamma0).
const PARAM_COLS = [:vmThresh, :fricExp, :mu_scale, :stiff_scale, :gamma0, :melt_flux]

const INPUT_CSV  = joinpath(PROJECT_ROOT, "inputs", "data",
                            "Amery_Input_Parameters_Filtered.csv")
const OUTPUT_CSV = joinpath(PROJECT_ROOT, "inputs", "data",
                            "Amery_Grd_Vol_Change_time_series_2300_ssp5.csv")
const OUT_DIR    = joinpath(PROJECT_ROOT, "outputs", "plots", "emulator_diagnostics")

# Allow a quick single/subset-year smoke test via FL_DIAG_YEARS=2300 (or a list).
function _years_to_run()
    env = get(ENV, "FL_DIAG_YEARS", "")
    isempty(env) && return copy(ALL_YEARS)
    req = parse.(Int, strip.(split(env, ",")))
    bad = setdiff(req, ALL_YEARS)
    @assert isempty(bad) "FL_DIAG_YEARS contains non-analysis years: $bad (valid: $ALL_YEARS)"
    return req
end


#========================================================================#
# Plot style (matches the notebook)
#========================================================================#
const seaborn = pyimport("seaborn")
try
    pyimport("scienceplots")
    PyPlot.matplotlib.style.use(["default", "science", "no-latex"])
catch err
    @warn "scienceplots style unavailable; falling back to default matplotlib style." err
    PyPlot.matplotlib.style.use("default")
end

const SIZE      = 18
const FONTSIZE  = 22.5
const DOTCOLOR  = "green"
const EDGECOLOR = "gray"


#========================================================================#
# Data load & scaling
#========================================================================#
@info "Loading data" INPUT_CSV OUTPUT_CSV

X_raw = CSV.read(INPUT_CSV, DataFrame)
# Confirm the 6 expected parameter columns exist (drop any index / extra columns).
for c in PARAM_COLS
    @assert String(c) in names(X_raw) "Input CSV is missing expected column :$c"
end

y_out_raw = CSV.read(OUTPUT_CSV, DataFrame)
Y_out = collect(Matrix(select(y_out_raw, Not(1))))   # N × 284, drops the index column

# Row-alignment guard: STOP rather than guess if the ensembles are not aligned.
@assert nrow(X_raw) == size(Y_out, 1) """
Row-count mismatch between inputs ($(nrow(X_raw)) rows) and outputs \
($(size(Y_out,1)) rows). The two CSVs must be row-aligned (same ensemble members). \
Aborting rather than guessing an alignment."""
const N_MEMBERS = nrow(X_raw)
@info "Row-aligned ensembles" N_MEMBERS n_output_years=size(Y_out, 2)

# Scale the 6 inputs to [0,1] with the SAME transform the calibration uses.
X_scaled_NxD, _, _, _ = ScaleUtils.get_scaled_matrix(X_raw, PARAM_COLS)  # N × 6
const X_SCALED = collect(Matrix(X_scaled_NxD)')                          # 6 × N
# Raw 6×N inputs (same column order) for LEAK-FREE per-fold scaling inside run_cv: the input min–max
# is refit on each training fold rather than the whole ensemble.
const X_RAW = collect(Matrix(select(X_raw, PARAM_COLS))')                # 6 × N (raw units)

# Raw parameter columns (original row order) for the scatter panels.
const P_vmThresh   = Float64.(X_raw.vmThresh)
const P_fricExp    = Float64.(X_raw.fricExp)
const P_mu_scale   = Float64.(X_raw.mu_scale)
const P_stiff_scale = Float64.(X_raw.stiff_scale)
const P_gamma0     = Float64.(X_raw.gamma0)
const P_melt_flux  = Float64.(X_raw.melt_flux)


#========================================================================#
# CV core
#========================================================================#
const np = pyimport("numpy")

"""
    kfold_splits(n, k, seed) -> Vector{Tuple{Vector{Int},Vector{Int}}}

Reproduce sklearn.model_selection.KFold(n_splits=k, shuffle=true, random_state=seed)
byte-for-byte, so the folds match those used for appendix Figures A1/A2. sklearn's algorithm:
shuffle 0:n-1 with numpy's legacy RandomState(seed), then hand out consecutive blocks; the
first (n % k) folds get one extra sample. Returns 1-based (train_idx, test_idx) index vectors.
"""
function kfold_splits(n::Int, k::Int, seed::Int)
    rng = np.random.RandomState(seed)
    idx = np.arange(n)              # 0-based numpy array
    rng.shuffle(idx)               # in-place Fisher–Yates (as sklearn does)
    perm = Int.(idx) .+ 1          # → 1-based Julia indices
    base, rem = divrem(n, k)
    fold_sizes = fill(base, k)
    fold_sizes[1:rem] .+= 1        # first (n % k) folds get one extra
    splits = Tuple{Vector{Int},Vector{Int}}[]
    cur = 1
    for fs in fold_sizes
        test_idx = perm[cur:cur+fs-1]
        train_idx = vcat(perm[1:cur-1], perm[cur+fs:end])
        push!(splits, (train_idx, test_idx))
        cur += fs
    end
    return splits
end

"""
    fit_fold_gp(Xtr, ytr) -> gp

Fit ONE GP on the fold's training rows using the exact pipeline recipe (linear mean +
Matern-5/2 ARD, fixed obs-noise). Doubles iterations up to N_RETRIES times on non-convergence,
then returns the last GP with a warning.
"""
function fit_fold_gp(Xtr::AbstractMatrix, ytr::AbstractVector)
    d = size(Xtr, 1)
    iters = BASE_ITERS
    local gp
    for attempt in 0:N_RETRIES
        gp = GaussianProcesses.GP(Xtr, ytr, MeanLin(zeros(d)),
                                  Mat52Ard(zeros(d), 0.001), LOG_OBS_NOISE)
        res = optimize!(gp; domean=true, kern=true, noise=false, iterations=iters)
        Optim.converged(res) && return gp
        if attempt < N_RETRIES
            iters *= 2
        else
            @warn "Fold GP did not converge after $(N_RETRIES+1) attempts " *
                  "(up to $iters iters); using last state for diagnostics."
        end
    end
    return gp
end

"""
    run_cv(year) -> (actual, fitted, std_fit, residual, metrics)

K-fold, out-of-sample CV for one year's grounded-volume emulator. Returns the per-member
actual (Gt), fitted mean (Gt), predictive std (Gt), residual (actual − fitted, Gt), all in
original row order, plus a metrics NamedTuple.
"""
function run_cv(year::Int)
    ycol = year - 2016
    @assert 1 <= ycol <= size(Y_out, 2) "Year $year → column $ycol out of range."
    y_actual = Float64.(Y_out[:, ycol])                 # RAW Gt, original order

    # K-fold splits identical to the template notebook's sklearn KFold
    # (shuffle=true, random_state=42), reproduced via numpy's RandomState.
    splits = KF_SHUFFLE ? kfold_splits(N_MEMBERS, N_FOLDS, KF_SEED) :
                          error("Only the shuffled KFold recipe is supported.")

    fitted  = zeros(N_MEMBERS)
    std_fit = zeros(N_MEMBERS)
    for (train_idx, test_idx) in splits
        # Fit BOTH scalers on the TRAIN fold ONLY — held-out members never influence preprocessing
        # (no CV leakage): input min–max per feature and output min–max, both from the train fold,
        # applied to the held-out fold; predictions are unscaled with the train output scaler.
        xmin  = vec(minimum(X_RAW[:, train_idx]; dims = 2)); xmax = vec(maximum(X_RAW[:, train_idx]; dims = 2))
        xspan = ifelse.(xmax .== xmin, 1.0, xmax .- xmin)
        Xtr   = (X_RAW[:, train_idx] .- xmin) ./ xspan
        Xte   = (X_RAW[:, test_idx]  .- xmin) ./ xspan
        ymin, ymax = extrema(@view y_actual[train_idx]); yspan = ymax == ymin ? 1.0 : ymax - ymin
        ytr   = (y_actual[train_idx] .- ymin) ./ yspan

        gp    = fit_fold_gp(Xtr, ytr)
        μ, σ² = predict_f(gp, Xte)                        # predict_f (deterministic-simulator convention)
        fitted[test_idx]  = μ .* yspan .+ ymin            # unscale mean with the train output scaler
        std_fit[test_idx] = sqrt.(σ²) .* yspan            # unscale predictive std by the train span
    end

    residual = y_actual .- fitted

    # Summary metrics (all in Gt / dimensionless).
    rmse      = sqrt(mean(residual .^ 2))
    rms_act   = sqrt(mean(y_actual .^ 2))
    nrmse     = rms_act == 0 ? NaN : rmse / rms_act
    ss_res    = sum(residual .^ 2)
    ss_tot    = sum((y_actual .- mean(y_actual)) .^ 2)
    r2        = ss_tot == 0 ? NaN : 1 - ss_res / ss_tot
    pearson   = cor(y_actual, fitted)

    metrics = (year=year, rmse_gt=rmse, nrmse=nrmse, r2=r2,
               rms_actual_gt=rms_act, pearson=pearson,
               max_abs_resid_gt=maximum(abs, residual))
    return y_actual, fitted, std_fit, residual, metrics
end


#========================================================================#
# 6-panel figure (notebook layout)
#========================================================================#
"""
    make_figure(year, actual, fitted, std_fit, residual)

Render and save the 6-panel CV diagnostic for one year, faithful to the notebook
layout/labels. Unlike the grounding-line template, the residual/fitted axis limits and the
colorbar range are auto-scaled to the data because grounded-volume magnitudes (and their
spread) differ by orders of magnitude across years.
"""
function make_figure(year, actual, fitted, std_fit, residual)
    # Panel-(a) residual axis uses the full data range; the COLOUR scale is capped (default 95th
    # percentile of |residual|) so out-of-range residuals saturate to the colormap extremes and the
    # bulk of smaller residuals gets more colour contrast -- matching appendix Fig A1/A2.
    # (std_fit is no longer used: the predictive-interval error bars have been removed.)
    cmax = maximum(abs, residual)
    cmax = cmax == 0 ? 1.0 : cmax
    ccap = quantile(abs.(residual), CLIP_Q)
    ccap = ccap == 0 ? cmax : ccap
    fit_lo, fit_hi = extrema(vcat(actual, fitted))
    padf = 0.05 * (fit_hi - fit_lo == 0 ? 1.0 : fit_hi - fit_lo)
    fit_lo -= padf; fit_hi += padf

    fig, ax = subplots(nrows=2, ncols=3, figsize=(16, 10))

    # (a) Residuals vs Fitted with lowess smooth.
    try
        seaborn.regplot(x=fitted, y=residual, lowess=true,
            scatter_kws=Dict("alpha"=>0.85, "s"=>SIZE*2, "color"=>DOTCOLOR,
                             "edgecolor"=>EDGECOLOR, "linewidths"=>0.5,
                             "plotnonfinite"=>false, "zorder"=>2),
            line_kws=Dict("color"=>"red", "lw"=>1, "alpha"=>0.8), ax=ax[1,1])
    catch err
        @warn "seaborn.regplot lowess failed; plain scatter fallback." year err
        ax[1,1].scatter(fitted, residual, s=SIZE*2, c=DOTCOLOR, alpha=0.85,
                        edgecolor=EDGECOLOR, linewidths=0.5, zorder=2)
    end
    ax[1,1].set_ylabel("Residuals", fontsize=FONTSIZE)
    ax[1,1].set_xlabel("Fitted values", fontsize=FONTSIZE, labelpad=10)
    ax[1,1].grid(true, alpha=0.5, zorder=1)
    ax[1,1].ticklabel_format(style="sci", scilimits=(-4,4), useMathText=true)
    ax[1,1].tick_params(axis="both", which="major", labelsize=FONTSIZE*0.8)
    ax[1,1].xaxis.offsetText.set_fontsize(FONTSIZE*0.8)
    ax[1,1].yaxis.offsetText.set_fontsize(FONTSIZE*0.8)
    ax[1,1].set_ylim(-1.1*cmax, 1.1*cmax)
    ax[1,1].set_xlim(fit_lo, fit_hi)
    ax[1,1].locator_params(tight=true, nbins=6)
    ax[1,1].text(-0.2, 1, "(a)", ha="left", va="top",
                 transform=ax[1,1].transAxes, fontsize=FONTSIZE)

    # (b) True vs Fitted + 45° line. No predictive-interval error bars (code uncertainty excluded).
    ax[2,1].scatter(x=actual, y=fitted, s=SIZE*2, c=DOTCOLOR, alpha=0.85,
                    edgecolor=EDGECOLOR, linewidths=0.5, plotnonfinite=false, zorder=4)
    ax[2,1].plot([fit_hi, fit_lo], [fit_hi, fit_lo], "r-")
    ax[2,1].set_xlabel("True Values", fontsize=FONTSIZE)
    ax[2,1].set_ylabel("Fitted values", fontsize=FONTSIZE, labelpad=10)
    ax[2,1].locator_params(tight=true, nbins=6)
    ax[2,1].set_xlim(fit_lo, fit_hi)
    ax[2,1].set_ylim(fit_lo, fit_hi)
    ax[2,1].grid(true, alpha=0.5, zorder=1)
    ax[2,1].ticklabel_format(style="sci", scilimits=(-3,3), useMathText=true)
    ax[2,1].tick_params(axis="both", which="major", labelsize=FONTSIZE*0.8)
    ax[2,1].xaxis.offsetText.set_fontsize(FONTSIZE*0.8)
    ax[2,1].yaxis.offsetText.set_fontsize(FONTSIZE*0.8)
    ax[2,1].text(-0.2, 1, "(b)", ha="left", va="top",
                 transform=ax[2,1].transAxes, fontsize=FONTSIZE)

    # Helper for the four parameter-pair scatters coloured by residual.
    function param_panel(a, xv, yv, xlab, ylab; tag=nothing)
        im = a.scatter(xv, yv, s=SIZE*5, c=residual, edgecolor=EDGECOLOR,
                       linewidths=0.5, plotnonfinite=false, cmap="RdBu",
                       vmin=-ccap, vmax=ccap, zorder=2)
        a.set_xlabel(xlab, fontsize=FONTSIZE, labelpad=10)
        a.set_ylabel(ylab, fontsize=FONTSIZE, labelpad=10)
        a.grid(true, alpha=0.5, zorder=1)
        a.locator_params(tight=true, nbins=6)
        a.ticklabel_format(style="sci", scilimits=(-3,3), useMathText=true)
        a.tick_params(axis="both", which="major", labelsize=FONTSIZE*0.8)
        a.xaxis.offsetText.set_fontsize(FONTSIZE*0.8)
        a.yaxis.offsetText.set_fontsize(FONTSIZE*0.8)
        tag !== nothing && a.text(-0.24, 1, tag, ha="left", va="top",
                                  transform=a.transAxes, fontsize=FONTSIZE)
        return im
    end

    im12 = param_panel(ax[1,2], P_mu_scale,   P_fricExp,  L"C_{\mu}",  L"q"; tag="(c)")
    param_panel(ax[1,3], P_stiff_scale, P_vmThresh, L"C_{\phi}", L"\sigma_{max}")
    param_panel(ax[2,2], P_stiff_scale, P_mu_scale, L"C_{\phi}", L"C_{\mu}")
    param_panel(ax[2,3], P_melt_flux,   P_gamma0,   L"\overline{m}", L"\gamma_0")

    plt.tight_layout()

    # Shared "Residuals" colorbar (RdBu, symmetric).
    cbar_ax = fig.add_axes([1.03, 0.25, 0.02, 0.5])
    cbar = fig.colorbar(im12, cax=cbar_ax)
    cbar.mappable.set_clim(-ccap, ccap)
    cbar.ax.yaxis.offsetText.set_fontsize(FONTSIZE*0.8)
    cbar.ax.yaxis.offsetText.set_position((1.3, 0))
    cbar.ax.ticklabel_format(style="sci", scilimits=(-3,3), useMathText=true)
    cbar.ax.locator_params(tight=true, nbins=6)
    cbar.ax.set_title("Residuals", fontsize=FONTSIZE, pad=32)
    cbar.ax.tick_params(axis="both", which="major", labelsize=FONTSIZE*0.8)

    # No figure title (per no-titles preference); the year is in the output filename.

    base = joinpath(OUT_DIR, "emulator_cv_grd_vol_$(year)")
    fig.savefig(base * ".pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    return base
end


#========================================================================#
# Main sweep
#========================================================================#
function main()
    mkpath(OUT_DIR)
    years = _years_to_run()
    @info "Running emulator CV diagnostics" n_years=length(years) years N_FOLDS shuffle=KF_SHUFFLE seed=KF_SEED

    rows = NamedTuple[]
    for year in years
        @info "▶ Year $year"
        actual, fitted, std_fit, residual, m = run_cv(year)
        base = make_figure(year, actual, fitted, std_fit, residual)
        @info "   wrote $(basename(base)).pdf" rmse_gt=round(m.rmse_gt, digits=3) nrmse=round(m.nrmse, digits=5) r2=round(m.r2, digits=5)
        push!(rows, m)
    end

    summary = DataFrame(rows)
    summary = select(summary, :year, :rmse_gt, :nrmse, :r2, :rms_actual_gt,
                     :pearson, :max_abs_resid_gt)
    csv_path = joinpath(OUT_DIR, "cv_summary.csv")
    CSV.write(csv_path, summary)

    println("\n=================== PER-YEAR CV SUMMARY ===================")
    @printf("%-6s %14s %12s %10s %14s %9s\n",
            "year", "RMSE (Gt)", "nRMSE", "R2", "RMS_act (Gt)", "Pearson")
    for r in eachrow(summary)
        @printf("%-6d %14.3f %12.5f %10.5f %14.3f %9.5f\n",
                r.year, r.rmse_gt, r.nrmse, r.r2, r.rms_actual_gt, r.pearson)
    end
    println("==========================================================")
    println("\nWrote summary  -> ", csv_path)
    println("Wrote figures  -> ", OUT_DIR, "/emulator_cv_grd_vol_<year>.pdf")
    return summary
end

main()
