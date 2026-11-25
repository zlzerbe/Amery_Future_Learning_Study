# ─────────────────────────────────────────────────────────────────────────────
# MCMC effective-sample-size diagnostics (OPT-IN). Reads the per-realization
# posterior caches (R_<r>_Posterior_Dict.jld2 built by the notebook's cache cell)
# and reports per-parameter ESS by analysis year. NOT part of the figure pipeline.
#
# Requires packages already loaded by the notebook's Cell 0 (JLD2, MCMCChains,
# DataFrames, Printf) and the config-cell globals PARAMS / posteriors_directory /
# GAPPED_YEARS / REALIZATIONS.
#
# Usage from the notebook:
#   include("diagnostics/ess_report.jl")
#   ess_scan(posteriors_directory; realizations=REALIZATIONS, years=GAPPED_YEARS)  # flag any param ESS < 1000 across the batch
#   ess_report(posteriors_directory; realization=4, years=GAPPED_YEARS)            # full ESS/mean/sd table for one realization
# ─────────────────────────────────────────────────────────────────────────────
using JLD2, MCMCChains, DataFrames, Printf

# iters×P (or P×iters) chain matrix → per-parameter summarystats DataFrame
# (:parameters, :mean, :std, :ess or :ess_bulk, :rhat, …).
function _ess_df(A, params)
    if size(A, 2) != length(params) && size(A, 1) == length(params)
        @warn "Transposing chain matrix (looks like params × iters)" sizeA = size(A)
        A = permutedims(A)
    end
    ch = Chains(reshape(A, size(A, 1), size(A, 2), 1), params; start=1, thin=1)
    DataFrame(MCMCChains.summarystats(ch; sections=:parameters))
end

# ESS column name changed across MCMCChains versions (:ess → :ess_bulk); accept either.
_ess(df) = :ess in propertynames(df) ? df.ess : df.ess_bulk

"""
    ess_scan(base; realizations, years, threshold=1000.0, params=PARAMS) -> Vector{Tuple{Int,Int}}

Scan every (realization, year) posterior cache under `base` and print those where ANY
parameter's ESS < `threshold`. Missing caches / years are skipped. Returns the flagged
(realization, year) pairs.
"""
function ess_scan(base; realizations, years, threshold=1000.0, params=PARAMS)
    flagged = Tuple{Int,Int}[]
    for r in realizations
        cache = "$(base)/R_$(r)_Posterior_Dict.jld2"
        isfile(cache) || continue
        dict = JLD2.load(cache, "post_data")
        for y in years
            haskey(dict, y) || continue
            df = _ess_df(dict[y], params)
            if any(_ess(df) .< threshold)
                push!(flagged, (r, y))
                @printf("R%-3d %d   min ESS=%.1f < %.0f\n", r, y, minimum(_ess(df)), threshold)
                for (name, e) in zip(df.parameters, _ess(df))
                    @printf("    %-12s ESS=%8.1f\n", String(name), e)
                end
            end
        end
    end
    println(isempty(flagged) ?
        "All ESS ≥ $threshold  ($(length(realizations)) realizations × $(length(years)) years)." :
        "$(length(flagged)) (realization, year) pairs below $threshold.")
    return flagged
end

"""
    ess_report(base; realization, years, params=PARAMS)

Print the full per-year ESS / mean / sd table for a single `realization`.
"""
function ess_report(base; realization, years, params=PARAMS)
    dict = JLD2.load("$(base)/R_$(realization)_Posterior_Dict.jld2", "post_data")
    for y in years
        if !haskey(dict, y); println("Realization $realization — $y: (missing)"); continue; end
        df = _ess_df(dict[y], params)
        println("Realization $realization — $y")
        for (name, e, m, s) in zip(df.parameters, _ess(df), df.mean, df.std)
            @printf("    %-12s ESS=%8.1f  mean=%-12.5g sd=%-.5g\n", String(name), e, m, s)
        end
    end
end
