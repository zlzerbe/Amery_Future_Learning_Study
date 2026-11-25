# ── scale_utils.jl ──

module ScaleUtils

using StatsBase
using DataFrames

"""
    scale_columns(df, cols) -> (scaled_df, scalers, col_mins, col_maxs)

Min–max scale each column in `cols` to [0,1] with a fitted `UnitRangeTransform`. Returns the
scaled columns (named `<col>_scaled`), the per-column fitted scalers, and the per-column original
min/max (used to invert the transform downstream).
"""
function scale_columns(df::DataFrame, cols::Vector{Symbol})
    scaled_df  = DataFrame()
    scalers    = Dict{Symbol,UnitRangeTransform}()
    col_mins   = Dict{Symbol,Float64}()
    col_maxs   = Dict{Symbol,Float64}()

    for col in cols
        arr = Float64.(df[!, col])
        mn = minimum(arr)
        mx = maximum(arr)
        col_mins[col] = mn
        col_maxs[col] = mx
        scaler = fit(UnitRangeTransform, arr)
        transformed = StatsBase.transform(scaler, arr)
        scaled_col_name = Symbol(string(col, "_scaled"))
        scaled_df[!, scaled_col_name] = transformed
        scalers[col] = scaler
    end

    return scaled_df, scalers, col_mins, col_maxs
end

"""
    get_scaled_matrix(df, cols) -> (X, scalers, col_mins, col_maxs)

`scale_columns` with the scaled DataFrame returned as a `Matrix` (columns = the `<col>_scaled`).
"""
function get_scaled_matrix(df::DataFrame, cols::Vector{Symbol})
    scaled_df, scalers, col_mins, col_maxs = scale_columns(df, cols)
    X = Matrix(scaled_df)
    return X, scalers, col_mins, col_maxs
end

end # module ScaleUtils
