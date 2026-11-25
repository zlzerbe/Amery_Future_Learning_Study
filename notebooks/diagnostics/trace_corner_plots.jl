# ─────────────────────────────────────────────────────────────────────────────
# MCMC convergence diagnostics (OPT-IN): per-realization trace + corner (pairwise)
# plots for a chosen realization at chosen analysis years. NOT part of the normal
# figure pipeline — include + call this only when you want to eyeball convergence.
#
# Requires packages already loaded by the notebook's Cell 0 (StatsPlots, MCMCChains,
# JLD2) and the config-cell globals PARAMS / hpc_data_directory / TAG.
#
# Usage from the notebook:
#   include("diagnostics/trace_corner_plots.jl")
#   trace_corner(hpc_data_directory; realization=67, years=[2060,2075,2090,2105], tag=TAG)
# ─────────────────────────────────────────────────────────────────────────────
using StatsPlots, MCMCChains, JLD2

"Locate the <year>_chain.jld2 for (realization, year) under an MCMC output tree."
function _find_chain_path(base, realization, year)
    for d in filter(d -> occursin("_R$(realization)_", d), readdir(base))
        chain_dir = joinpath(base, d, "chains_dir")
        isdir(chain_dir) || continue
        for f in filter(f -> endswith(f, "_chain.jld2"), readdir(chain_dir))
            parse(Int, first(f, 4)) == year && return joinpath(chain_dir, f)
        end
    end
    error("No chain for realization $realization, year $year in $base")
end

# flatten a (iter, param, chain) Chains slice to a (draws × params) matrix
_flatten(ch, vars) = (A = Array(ch[:, vars, :]); reshape(A, size(A, 1) * size(A, 3), size(A, 2)))

"""
    trace_corner(base; realization, years, params=PARAMS, max_iter=5000, tag="",
                 trace_outdir="../outputs/plots/Trace_Plots",
                 corner_outdir="../outputs/plots/Pairwise")

Save a thinned trace plot and a corner (pairwise) plot for `realization` at each of `years`,
reading chains from the MCMC tree `base`. Chains are thinned to ~`max_iter` draws. Filenames are
suffixed with `tag` (pass the configuration's TAG so outputs don't collide across configurations).
"""
function trace_corner(base; realization, years, params=PARAMS, max_iter=5000, tag="",
                      trace_outdir="../outputs/plots/Trace_Plots",
                      corner_outdir="../outputs/plots/Pairwise")
    mkpath(trace_outdir); mkpath(corner_outdir)
    for y in years
        ch   = JLD2.load(_find_chain_path(base, realization, y), "chain")
        pars = filter(p -> p in names(ch, :parameters), params)      # keep PARAMS order
        step = max(1, cld(size(ch, 1), max_iter))
        ch_thin = ch[1:step:end, pars, :]

        # trace plot (one row per parameter)
        ch_plot = Chains(Array(ch_thin), collect(pars); start=1, thin=1)
        p_trace = StatsPlots.plot(ch_plot; seriestype=:traceplot, legend=false,
            title="Realization $realization — Year $y (thinned ×$step)",
            layout=(length(pars), 1), size=(1100, max(220 * length(pars), 300)))
        StatsPlots.savefig(p_trace, joinpath(trace_outdir, "R$(realization)_$(y)_trace$(tag).pdf"))

        # corner / pairwise plot
        if length(pars) ≥ 2
            X = _flatten(ch_thin, pars); n = size(X, 2); cell = 380
            p_corr = corrplot(X, compact=true; label=permutedims(String.(pars)),
                              grid=true, size=(cell * n, cell * n))
            StatsPlots.savefig(p_corr, joinpath(corner_outdir, "R$(realization)_$(y)_corner$(tag).pdf"))
        else
            @info "Skipping corner plot for year $y (need ≥2 parameters)"
        end
    end
    println("trace/corner plots for R$(realization) → $(trace_outdir) , $(corner_outdir)")
end
