# ── scale_utils.jl ──

module ScaleUtils

using StatsBase
using DataFrames

#========================================#
#  PART 1: Scaling DataFrame columns 
#========================================#

#= 
    scale_columns(df, cols) -> (scaled_df, scalers)

  - df::DataFrame
  - cols::Vector{Symbol}

For each column ∈ cols:
  1. Convert df[!, col] to Float64 vector
  2. Compute min, max; store in col_mins[col], col_maxs[col]
  3. Fit UnitRangeTransform on that Float64[]
  4. Transform into [0,1]
  5. Insert a new column "<col>_scaled" into `scaled_df`
  6. Store the fitted scaler in `scalers[col]`

Returns:
  - scaled_df::DataFrame
      contains only the "<orig>_scaled" columns
  - scalers::Dict{Symbol,UnitRangeTransform}
      so you can reuse the transforms on future data
  - col_mins::Dict{Symbol,Float64} 
      maps each original column → its minimum
  - col_maxs::Dict{Symbol,Float64}
      maps each original column → its maximum
=#
function scale_columns(df::DataFrame, cols::Vector{Symbol})
    scaled_df = DataFrame()
    scalers   = Dict{Symbol,UnitRangeTransform}()
    col_mins   = Dict{Symbol,Float64}()
    col_maxs   = Dict{Symbol,Float64}()

    for col in cols
        # 1) Convert to Float64
        arr = Float64.(df[!, col])

        # 2) Compute min and max
        mn = minimum(arr)
        mx = maximum(arr)
        col_mins[col] = mn
        col_maxs[col] = mx

        # 3) Fit the UnitRangeTransform
        scaler = fit(UnitRangeTransform, arr)

        # 4) Transform into [0,1]
        transformed = StatsBase.transform(scaler, arr)

        # 5) Insert into scaled_df under "<col>_scaled"
        scaled_col_name = Symbol(string(col, "_scaled"))
        scaled_df[!, scaled_col_name] = transformed

        # 6) Keep the scaler
        scalers[col] = scaler
    end

    return scaled_df, scalers, col_mins, col_maxs
end

#= 
    get_scaled_matrix(df, cols) -> (X, scalers)

Convenience wrapper that:
  • Calls scale_columns(df, cols)
  • Converts scaled_df to a Matrix
  • Transposes so each row = one "<orig>_scaled" column

Returns:
  - X::Matrix{Float64}         
      (rows correspond to each "<orig>_scaled" column)
  - scalers::Dict{Symbol,UnitRangeTransform}
  - col_mins::Dict{Symbol,Float64}
  - col_maxs::Dict{Symbol,Float64}
=#
function get_scaled_matrix(df::DataFrame, cols::Vector{Symbol})
    scaled_df, scalers, col_mins, col_maxs = scale_columns(df, cols)
    X = Matrix(scaled_df)   # now rows = each "<orig>_scaled" column
    return X, scalers, col_mins, col_maxs
end


#========================================#
#  PART 2: Scaling a 2D Array (per‐row)
#========================================#

#=
    scale_matrix_rows(mat::AbstractMatrix{<:Real})
      -> (mat_scaled, scalers, row_mins, row_maxs)

  Inputs:
    mat       – any real‐valued 2D array, size = (nrows, ncols)
                 (e.g. pred_pre where each row is one PC’s values)

  Outputs:
    mat_scaled::Matrix{Float64}         size (nrows, ncols),
      where each row_i is min–max–scaled to [0,1].
    scalers::Vector{UnitRangeTransform} length = nrows,
      the fitted scaler for each row.
    row_mins::Vector{Float64} length = nrows, each row’s original minimum.
    row_maxs::Vector{Float64} length = nrows, each row’s original maximum.
=#
function scale_matrix_rows(mat::AbstractMatrix{<:Real})
    nrows, ncols = size(mat)

    # Preallocate outputs
    mat_scaled = Matrix{Float64}(undef, nrows, ncols)
    scalers    = Vector{UnitRangeTransform}(undef, nrows)
    row_mins   = Vector{Float64}(undef, nrows)
    row_maxs   = Vector{Float64}(undef, nrows)

    for i in 1:nrows
        # Extract ith row, convert to Float64
        row_vals = Float64.(view(mat, i, :))

        # Record min/max
        row_mins[i], row_maxs[i] = minimum(row_vals), maximum(row_vals)

        # Fit + transform
        scaler = fit(UnitRangeTransform, row_vals)
        scalers[i] = scaler

        mat_scaled[i, :] = StatsBase.transform(scaler, row_vals)
    end

    return mat_scaled, scalers, row_mins, row_maxs
end

end # module ScaleUtils