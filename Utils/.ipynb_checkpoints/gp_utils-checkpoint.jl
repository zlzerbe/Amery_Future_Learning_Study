# ── gp_utils.jl ──

module GPUtils

using GaussianProcesses
using JLD2
using Optim


export train_all_emulators!, load_emulators, get_emulators_for_years

#========================================================================#
# train_all_emulators!(path::String, X, Ymat, years;
#                     init_noise=-4.605170185985, base_iters=100, max_retries=3)
#
#   - X      :: d×N matrix of inputs
#   - Ymat   :: N×J matrix of outputs (column j ↦ “year” years[j])
#   - years  :: Vector{Int} of length J (e.g. [2030, 2045, …, 2285])
#
# Trains one GP per year, stores them in a Dict{Int,Any} (value ≈ GP), 
# and saves that Dict to `path` via JLD2.  If `path` already exists, just load it.
#
# Keyword arguments:
#   init_noise  :: initial log‐noise (default ≈ log(0.01))  
#   base_iters  :: how many iterations to try first when calling optimize!  
#   max_retries :: how many times to double iterations if optimize! errors
#
# Behavior:
#   • If `path` already exists on disk, we simply load and return it (no retraining).  
#   • Otherwise, we train all J GPs, save them, and return the Dict.
#========================================================================#
function train_all_emulators!(
        path::String,
        X::AbstractMatrix{<:Real},
        Ymat::AbstractMatrix{<:Real},
        years::Vector{Int};
        init_noise::Float64 = -4.605170185985,
        base_iters::Int    = 100,
        max_retries::Int   = 3
    )::Dict{Int,Any}

    # If the file already exists, load and return it
    if isfile(path)
        @info "Loading existing emulator dictionary from $path"
        return load_emulators(path)
    end

    @info "File $path not found—training all $(length(years)) emulators now."

    N, J = size(Ymat)
    d, NX = size(X)
    @assert NX == N "Mismatch: Ymat has $N rows but X has $NX columns."
    @assert length(years) == J "Mismatch: length(years)=$(length(years)) but Ymat has $J columns."

    # Create an untyped Dict; we will insert GP objects as we go
    emu_dict = Dict() 

    for (j, year) in enumerate(years)
        yvec = Float64.(Ymat[:, j])

        # Build a linear‐mean + Matern(5/2)-ARD kernel
        m_zero = MeanLin(zeros(d))
        kern   = Mat52Ard( zeros(d), 0.001 )

        # Construct the GP object (here `GaussianProcesses.GP` is a type)
        gp = GaussianProcesses.GP(X, yvec, m_zero, kern, init_noise)

        iters    = base_iters
        success  = false
        last_err = nothing

        for attempt in 0:max_retries
            
            res = optimize!(gp; domean=true, kern=true, noise=false, iterations=iters)
            
            if Optim.converged(res)
                success = true
                break
            else
                success = false
                @warn "Year $year: optimize! did not converge on attempt $(attempt+1) with $iters iterations."
                if attempt < max_retries
                    iters *= 2
                    @info "Retrying year $year with $iters iterations..."
                    gp = GaussianProcesses.GP(X, yvec, m_zero, kern, init_noise)
                end
            end
        end

        if !success
            error("Failed to optimize GP for year $year after $(max_retries+1) attempts. Last error: $last_err")
        end

        @info "Successfully trained emulator for year $year in $iters iterations."
        emu_dict[year] = gp
        
    end

    # Save the entire dictionary to disk
    @info "Saving all emulators to $path"
    @save path emu_dict

    return emu_dict
end


#========================================================================#
# load_emulators(path::String) -> Dict{Int,Any}
#
#   Loads and returns the Dict from the JLD2 file at `path`.
#========================================================================#
function load_emulators(path::String)::Dict{Int,Any}
    @assert isfile(path) "File $path does not exist."
    emu_dict = Dict{Int,Any}()
    @load path emu_dict
    return emu_dict
end


#========================================================================#
# get_emulators_for_years(emu_dict::Dict{Int,Any}, selected_years::Vector{Int})
#     -> Vector{GaussianProcesses.GP}
#
#   Given a dictionary (year → GP), return a Vector of GP objects in that order.
#   Errors if any requested year is missing.
#========================================================================#
function get_emulators_for_years(
        emu_dict::Dict{Int,Any},
        selected_years::Vector{Int}
    )::Vector{GaussianProcesses.GP}

    missing_years = setdiff(selected_years, collect(keys(emu_dict)))
    if !isempty(missing_years)
        error("These years are not in the emulator dict: $missing_years")
    end

    # Extract and return the GP objects (Julia will infer the array’s element type as GP)
    return [emu_dict[y]::GaussianProcesses.GP for y in selected_years]
end

end # module GPUtils