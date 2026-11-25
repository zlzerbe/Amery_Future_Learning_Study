using Distributions

struct TrapezoidalDist{T<:Real} <: ContinuousUnivariateDistribution
    l::T
    b::T
    c::T
    u::T

    # inner constructor function to instantiate new TrapezoidalDist objects
    function TrapezoidalDist{T}(l::T, b::T, c::T, u::T; check_args = true) where {T<:Real}
        check_args && Distributions.@check_args(TrapezoidalDist, l <= b < c <= u)
        return new{T}(l, b, c, u)
    end
end

# constructor functions for implicitly supplied type
# constructor for no type and params Float64
function TrapezoidalDist(l::Float64, b::Float64, c::Float64, u::Float64; check_args = true)
    return TrapezoidalDist{Float64}(l,b,c,u,check_args = check_args)
end

# constructor for real params - use promote to make aprams the same type
TrapezoidalDist(l::Real, b::Real, c::Real, u::Real) = TrapezoidalDist(promote(l, b, c, u)...)
TrapezoidalDist(l::Integer, b::Integer, c::Integer, u::Integer) = TrapezoidalDist(float(l), float(b), float(c), float(u))

##### BEGIN EIGHT METHODS
import Base.rand, StatsBase.params
import Random, Distributions, Statistics, StatsBase
using Random

# 0 - helper function
#### Parameters
StatsBase.params(d::TrapezoidalDist) = (d.l, d.b, d.c, d.u)


#1 rand(::AbstractRNG, d::UnivariateDistribution)
function Base.rand(rng::AbstractRNG, d::TrapezoidalDist)
    (l, b, c, u) = params(d)
    p1 = (b - l) / (u + c - b - l)
    p2 = 1 - (u - c) / (u + c - b - l)
    r = rand(rng)
    if 0 ≤ r < p1
        return l + sqrt(r * (u + c - b - l) * (b - l))
    elseif p1 ≤ r < p2
        return (r * (u + c - b - l) + b + l) / 2
    elseif p2 ≤ r ≤ 1
        return u - sqrt((1 - r) * (u + c - b - l) * (u - c))
    end
end

#2 logpdf(d::UnivariateDistribution, x::Real)
function Distributions.pdf(d::TrapezoidalDist, x::Real)
    l, b, c, u = params(d)
    if x < b
        res = 2 * (x - l) / ((b - l) * (u + c - b - l))
        return x < l ? zero(res) : res
    elseif c > x ≥ b
        res = 2 / (u + c - b - l)
        return res
    else
        res = 2 * (u - x) / ((u - c) * (u + c - b - l))
        return x > u ? zero(res) : res
    end
end
Distributions.logpdf(d::TrapezoidalDist, x::Real) = log(pdf(d, x))

#3 cdf(d::UnivariateDistribution, x::Real)
function Distributions.cdf(d::TrapezoidalDist, x::Real)
    l, b, c, u = params(d)
    if x < b
        res = (x - l)^2 / ((b - l) * (u + c - b - l))
        return x < l ? zero(res) : res
    elseif c > x ≥ b
        res = (2*x - l - b) / (u + c - b - l)
        return res
    else
        res = 1 - (u - x)^2 / ((u - c) * (u + c - b - l))
        return x > u ? one(res) : res
    end
end

#4 quantile(d::UnivariateDistribution, q::Real)
function Statistics.quantile(d::TrapezoidalDist, p::Real)
    (l, b, c, u) = params(d)
    w = u + c - b - l                     
    p ≤ 0 && return l
    p ≥ 1 && return u
    if p ≤ (b - l) / w                    
        return l + sqrt(p * w * (b - l))
    elseif p ≤ (2c - b - l) / w           
        return (p * w + l + b) / 2
    else                                  
        return u - sqrt((1 - p) * w * (u - c))
    end
end

#5 minimum(d::UnivariateDistribution)
function Base.minimum(d::TrapezoidalDist)
    (l, b, c, u) = params(d)
    return(l)
end

#6 maximum(d::UnivariateDistribution)
function Base.maximum(d::TrapezoidalDist)
    (l, b, c, u) = params(d)
    return(u)
end