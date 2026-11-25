# ── gp_utils.jl ──

module GPUtils

using JLD2

export load_emulators

"""
    load_emulators(path) -> Dict{Int,Any}

Load the year→GP emulator dictionary (saved under key `emu_dict` by JLD2) from `path`.
"""
function load_emulators(path::String)::Dict{Int,Any}
    @assert isfile(path) "File $path does not exist."
    emu_dict = Dict{Int,Any}()
    @load path emu_dict
    return emu_dict
end

end # module GPUtils
