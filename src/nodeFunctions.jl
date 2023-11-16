export hasweights, weights
export setscope!, setobs!, addobs!, scope, obs
export nobs, nscope
export isnormalized
export addscope!
export removescope!
export hasscope, hasobs
export logweights
export updatescope!
export classes, children, parents, length, add!, remove!, cf!, cf, logpdf!, logpdf
export rand

function isnormalized(node::Node)
    if !hasweights(node)
        return mapreduce(child -> isnormalized(child), &, children(node))
    else
        return sum(weights(node)) ≈ 1.0
    end
end
isnormalized(node::Leaf) = true

hasweights(node::SumNode) = true
hasweights(node::Node) = false

weights(node::SumNode) = exp.(node.logweights)
logweights(node::SumNode) = node.logweights

getindex(node::Node, i...) = getindex(node.children, i...)

function setscope!(node::CCNode, scope::Vector{Int})
    if length(scope) > 0

        if maximum(scope) > length(node.scopeVec)
            resize!(node.scopeVec, maximum(scope))
        end

        fill!(node.scopeVec, false)
        node.scopeVec[scope] .= true
    else
        fill!(node.scopeVec, false)
    end
end

function setscope!(node::CCNode, scope::Int)

    if scope > length(node.scopeVec)
        @warn "New scope is larger than original scope, resize node scope..."
        resize!(node.scopeVec, scope)
    end

    fill!(node.scopeVec, false)
    node.scopeVec[scope] = true
end

function addscope!(node::Node, scope::Int)
    @assert scope <= length(node.scopeVec)
    node.scopeVec[scope] = true
end

function removescope!(node::Node, scope::Int)
    @assert scope <= length(node.scopeVec)
    node.scopeVec[scope] = false
end

@inline scope(node::Node) = findall(node.scopeVec)
@inline scope(node::Leaf) = node.scope
@inline nscope(node::Node) = sum(node.scopeVec)
@inline nscope(node::Leaf) = length(node.scope)
@inline hasscope(node::Node) = sum(node.scopeVec) > 0
@inline hasscope(node::Leaf) = true
@inline hassubscope(n1::ProductNode, n2::CCNode) = any(n1.scopeVec .& n2.scopeVec)

function addobs!(node::CCNode, obs::Int)
    @assert obs <= length(node.obsVec)
    node.obsVec[obs] = true
end

function setobs!(node::Node, obs::AbstractVector{Int})
    if length(obs) > 0
        if maximum(obs) > length(node.obsVec)
            @warn "New obs is larger than original obs field, resize node obs..."
            resize!(node.obsVec, maximum(obs))
        end

        fill!(node.obsVec, false)
        node.obsVec[obs] .= true
    else
        fill!(node.obsVec, false)
    end
end

function setobs!(node::Node, obs::Int)
    if obs > length(node.obsVec)
        @warn "New obs is larger than original obs field, resize node obs..."
        resize!(node.obsVec, obs)
    end

    fill!(node.obsVec, false)
    node.obsVec[obs] = true
end

@inline nobs(node::Node) = sum(node.obsVec)
@inline obs(node::Node) = findall(node.obsVec)
@inline hasobs(node::Node) = any(node.obsVec)

"""
    updatescope!(cc)
Update the scope of all nodes in the CC.
"""
updatescope!(cc::CharacteristicCircuit) = updatescope!(cc.root)

function updatescope!(node::SumNode)
    for child in children(node)
        updatescope!(child)
    end
    setscope!(node, scope(first(children(node))))
    return node
end

function updatescope!(node::ProductNode)
    for child in children(node)
        updatescope!(child)
    end
    setscope!(node, mapreduce(c -> scope(c), vcat, children(node)))
    return node
end

updatescope!(node::Leaf) = node

"""

children(node) -> children::CCNode[]

Returns the children of an internal node.

##### Parameters:
* `node::Node`: Internal CC node to be evaluated.
"""
function children(node::Node)
    node.children
end

"""

parents(node) -> parents::CCNode[]

Returns the parents of a CC node.

##### Parameters:
* `node::CCNode`: CC node to be evaluated.
"""
function parents(node::CCNode)
    node.parents
end

"""
Add a node to a finite sum node with given weight in place.
add!(node::FiniteSumNode, child::CCNode, weight<:Real)
"""
function add!(parent::SumNode, child::CCNode, logweight::T) where T <: Real
    if !(child in parent.children)
        push!(parent.children, child)
        push!(parent.logweights, logweight)
        push!(child.parents, parent)
    end
end

function add!(parent::ProductNode, child::CCNode)
    if !(child in parent.children)
        push!(parent.children, child)
        push!(child.parents, parent)
    end
end

"""
Remove a node from the children list of a sum node in place.
remove!(node::FiniteSumNode, index::Int)
"""
function remove!(parent::SumNode, index::Int)
    pid = findfirst(map(p -> p == parent, parent.children[index].parents))

    @assert pid > 0 "Could not find parent ($(node.id)) in list of parents ($(parent.children[index].parents))!"
    deleteat!(parent.children[index].parents, pid)
    deleteat!(parent.children, index)
    deleteat!(parent.logweights, index)
end

function remove!(parent::ProductNode, index::Int)
    pid = findfirst(parent .== parent.children[index].parents)
    @assert pid > 0 "Could not find parent ($(node.id)) in list of parents ($(parent.children[index].parents))!"
    deleteat!(parent.children[index].parents, pid)
    deleteat!(parent.children, index)
end

function length(node::Node)
    Base.length(node.children)
end

function cf!(cc::CharacteristicCircuit, x::AbstractVector{<:Real}, cfvals::AxisArray)

    fill!(cfvals, -Inf)

    # Call inplace functions.
    for layer in cc.layers
        Threads.@threads for node in layer
            cf!(node, x, cfvals)
        end
    end

    cc.info[:llh] = real(cfvals[cc.root.id])

    return cfvals[cc.root.id]
end

function cf(cc::CharacteristicCircuit, x::AbstractVector{<:Real})
    idx = Axis{:id}(collect(keys(cc)))
    cfvals = AxisArray(Vector{Complex}(undef, length(idx)), idx)
    return cf!(cc, x, cfvals)
end

"""
    cf(n::SumNode, x)

Compute the cf of a sum node.
"""
function cf(n::SumNode, x::AbstractArray{<:Real}; w::AbstractVector{<:Real}=weights(n))
    return sum(w .* cf.(children(n), Ref(x)))
end

function cf!(n::SumNode, x::AbstractVector{<:Real}, cfvals::AxisArray{U,1}) where {U<:Complex}
    @inbounds y = map(c -> cfvals[c.id], children(n))
    cfvals[n.id] = map(U, _cf(n, x, y, weights(n)))
    return cfvals
end

function _cf(n::SumNode, x::AbstractArray{<:Real}, y::AbstractVector{<:T}, w::AbstractVector{<:Real}=weights(n)) where {T<:Complex}
    return sum(y.*w) # careful with complex dot product 
end

"""
    cf(n::ProductNode, x)

Compute the cf of a product node.
"""
function cf(n::ProductNode, x::AbstractArray{<:Real})
    return prod(cf(c, x) for c in children(n))
end

function _cf(n::ProductNode, x::AbstractVector{<:Real}, y::AbstractVector{<:Complex})
    return mapreduce(k -> hasscope(n[k]) ? y[k] : 1.0, *, 1:length(n))
end

function cf!(n::ProductNode, x::AbstractVector{<:Real}, cfvals::AxisArray{U}) where {U<:Complex}
    @inbounds y = map(c -> cfvals[c.id], children(n))
    cfvals[n.id] = map(U, _cf(n, x, y))
    return cfvals
end

# ################## #
# Leaf distributions #
# ################## #
"""
    cf(n::Leaf, x)

Compute the cf of a leaf node.
"""
function cf!(n::Leaf, x::AbstractVector{<:Real}, cfvals::AxisArray{U}) where {U<:Complex}
    cfvals[n.id] = map(U, _cf(n, x))
    return cfvals
end

_cf(n::UnivariateNode, x::AbstractVector{<:Real}) = _cf(n, x, Distributions.params(n.dist)...)
function _cf(n::UnivariateNode, x::AbstractVector{<:Real}, θ...)
    if length(params_opt(n))==0 # non-optim case, use params directly
        return Distributions.cf(n.dist, x[scope(n)])
    else # optim case, use CF for n.dist with params_opt for appropriate domains
        if typeof(n.dist) == Normal{Float64}
            μ = params_opt(n)[1]
            σ = exp(params_opt(n)[2]) # apply parameter constraints
            return cis.(μ * x[scope(n)]) * exp.(-0.5 * σ^2 * x[scope(n)].^2)
        elseif typeof(n.dist) == Categorical{Float64, Vector{Float64}}
            p = softmax(params_opt(n)) # apply parameter constraints
            x_supp = 1:length(p)
            cf_cat(t) = sum(v -> v[1] * cis(t * v[2]), zip(p, x_supp))
            return cf_cat.(x[scope(n)])
        elseif typeof(n.dist) == AlphaStable{Float64}
            α = 2/(1 + exp(-params_opt(n)[1])) # apply parameter constraints
            β = 2/(1 + exp(-params_opt(n)[2])) - 1
            c = exp(params_opt(n)[3])
            μ = params_opt(n)[4]
            if α == one(α)
                return exp(im*x[scope(n)]*μ - abs(c*x[scope(n)])^α * (1 - im*β*sign(x[scope(n)])*(-2/π * log(abs(x[scope(n)])))))
            else
                return exp(im*x[scope(n)]*μ - abs(c*x[scope(n)])^α * (1 - im*β*sign(x[scope(n)])*tan(π*α/2)))
            end
            # return Distributions.cf(AlphaStable(α, β, c, μ), x[scope(n)])
        else
            @error("NotImplementedError")
        end
    end
end

_cf(n::UnivariateECF, t::AbstractVector{<:Real}) = _cf(n, t, n.data)
function _cf(n::UnivariateECF, t::AbstractVector{<:Real}, data_x)
    if isnan(t[scope(n)])
        return 1.0 # when marginalised, cf(0)=1
    end

    if data_x == params(n)
        # calculate the ECF given n.data
        return mean(exp.(im * t[scope(n)] .* n.data); dims=1)[1]
    else 
        return mean(exp.(im * t[scope(n)] .* data_x); dims=1)[1]
    end
end

@inline cf(n::Leaf, x::AbstractArray{<:Real}) = _cf(n, x)
@inline cf(n::Leaf, x::AbstractArray{<:Real}, y...) = _cf(n, x, y...)

function logpdf!(cc::CharacteristicCircuit, X::AbstractMatrix{<:Real}, llhvals::AxisArray)

    fill!(llhvals, -Inf)

    # Call inplace functions.
    for layer in cc.layers
        Threads.@threads for node in layer
            logpdf!(node, X, llhvals)
        end
    end

    cc.info[:llh] = mean(llhvals[:,cc.root.id])

    return llhvals[:,cc.root.id]
end

function logpdf(cc::CharacteristicCircuit, X::AbstractMatrix{<:Real})
    idx = Axis{:id}(collect(keys(cc)))
    llhvals = AxisArray(Matrix{Float64}(undef, size(X, 1), length(idx)), 1:size(X, 1), idx)
    return logpdf!(cc, X, llhvals)
end

function logpdf!(cc::CharacteristicCircuit, x::AbstractVector{<:Real}, llhvals::AxisArray)

    fill!(llhvals, -Inf)

    # Call inplace functions.
    for layer in cc.layers
        Threads.@threads for node in layer
            logpdf!(node, x, llhvals)
        end
    end

    cc.info[:llh] = llhvals[cc.root.id]

    return llhvals[cc.root.id]
end

function logpdf(cc::CharacteristicCircuit, x::AbstractVector{<:Real})
    idx = Axis{:id}(collect(keys(cc)))
    llhvals = AxisArray(Vector{Float64}(undef, length(idx)), idx)
    return logpdf!(cc, x, llhvals)
end

"""
    logpdf(n::SumNode, x)

Compute the logpdf of a sum node.
"""
logpdf(n::SumNode, x::AbstractArray{<:Real}; lw::AbstractVector{<:Real}=logweights(n)) = n(x, lw=lw)

function logpdf!(n::SumNode, x::AbstractVector{<:Real}, llhvals::AxisArray{U,1}) where {U<:Real}
    @inbounds y = map(c -> llhvals[c.id], children(n))
    llhvals[n.id] = map(U, _logpdf(n, x, y, logweights(n)))
    return llhvals
end

function logpdf!(n::SumNode, x::AbstractMatrix{<:Real}, llhvals::AxisArray{U,2}) where {U<:Real}
    @inbounds Y = mapreduce(c -> llhvals[:, c.id], hcat, children(n))
    @inbounds llhvals[:,n.id] = map(i -> map(U, _logpdf(n, x, view(Y, i, :), logweights(n))), 1:size(x,1))
    return llhvals
end

# ############################################ #
# Concrete implementation for finite sum node. #
# ############################################ #
function (n::FiniteSumNode)(x::AbstractArray{<:Real}, y::AbstractVector{<:Real}; lw::AbstractVector{<:Real}=logweights(n))
    return _logpdf(n, x, y, lw)
end

function _logpdf(n::SumNode, x::AbstractArray{<:Real}, y::AbstractVector{<:Real}, lw::AbstractVector{<:Real})
    l = lw+y
    m = maximum(l)
    lp = log(mapreduce(v -> exp(v-m), +, l)) + m
    return isfinite(lp) ? lp : -Inf
end

function (n::FiniteSumNode)(x::AbstractVector{<:Real}; lw::AbstractVector{<:Real} = logweights(n))
    y = map(c -> c(x), children(n))
    return _logpdf(n, x, y, lw)
end

function (n::FiniteSumNode)(x::AbstractMatrix{<:Real}; lw::AbstractVector{<:Real}=logweights(n))
    @inbounds Y = mapreduce(i -> map(c -> c(view(x, i, :)), children(n)), hcat, 1:size(x,1))
    return @inbounds map(i -> _logpdf(n, x, view(Y, :, i), lw), 1:size(x,1))
end

"""
    logpdf(n::ProductNode, x)

Compute the logpdf of a product node.
"""
logpdf(n::ProductNode, x::AbstractArray{<:Real}) = n(x)
(n::FiniteProductNode)(x::AbstractArray{<:Real}, y::AbstractVector{<:Real}) = _logpdf(n, x, y)
function (n::FiniteProductNode)(x::AbstractVector{<:Real})
    y = map(c -> c(x), children(n))
    return _logpdf(n, x, y)
end

function (n::FiniteProductNode)(x::AbstractMatrix{<:Real})
    @inbounds Y = mapreduce(i -> map(c -> c(view(x, i, :)), children(n)), hcat, 1:size(x,1))
    return _logpdf(n, x, Y)
end

function _logpdf(n::ProductNode, x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    if !hasscope(n)
        return 0.0
    else
        return mapreduce(k -> hasscope(n[k]) ? y[k] : 0.0, +, 1:length(n))
    end
end

function _logpdf(n::ProductNode, x::AbstractMatrix{<:Real}, y::AbstractMatrix{<:Real})
    N = size(x,1)
    if !hasscope(n)
        return zeros(N)
    else
        return mapreduce(k -> hasscope(n[k]) ? y[:,k] : zeros(N), +, 1:length(n))
    end
end

function _logpdf(n::ProductNode, x::AbstractMatrix{<:Real}, y::AbstractVector{<:Real})
    # special case, the product node has only one child.
    if !hasscope(n)
        return 0.0
    else
        return y
    end
end

function logpdf!(n::ProductNode, x::AbstractVector{<:Real}, llhvals::AxisArray{U}) where {U<:Real}
    @inbounds y = map(c -> llhvals[c.id], children(n))
    llhvals[n.id] = map(U, _logpdf(n, x, y))
    return llhvals
end

function logpdf!(n::ProductNode, x::AbstractMatrix{<:Real}, llhvals::AxisArray{U}) where {U<:Real}
    @inbounds Y = mapreduce(c -> llhvals[:, c.id], hcat, children(n))
    llhvals[:,n.id] = map(U, _logpdf(n, x, Y))
    return llhvals
end

# ################## #
# Leaf distributions #
# ################## #
"""
    logpdf(n::Leaf, x)

Compute the logpdf of a leaf node.
"""
function logpdf!(n::Leaf, x::AbstractVector{<:Real}, llhvals::AxisArray{U}) where {U<:Real}
    llhvals[n.id] = map(U, n(x))
    return llhvals
end

function logpdf!(n::Leaf, x::AbstractMatrix{<:Real}, llhvals::AxisArray{U}) where {U<:Real}
    llhvals[:,n.id] .= map(U, n(x))
    return llhvals
end

@inline (n::UnivariateNode)(x::AbstractVector{<:Real}) = n(x, Distributions.params(n.dist)...)
@inline (n::UnivariateNode)(x::AbstractVector{<:Real}, θ...) = _logpdf(n, x, θ...)

function _logpdf(n::UnivariateNode, x::AbstractVector{<:Real}, θ...)
    if isnan(x[scope(n)])
        return 0.0
    end

    ϵ = 1e-5 # parameter for Laplace smoothing for Categorical Leaf
    if typeof(n.dist) == Normal{Float64}
        if length(params_opt(n))==0 # non-optim case, use params directly
            return logpdf(n.dist, x[scope(n)])
        else # optim case, use params_opt for appropriate domains 
            μ = params_opt(n)[1]
            σ = exp(params_opt(n)[2]) 
            return logpdf(Normal(μ, σ), x[scope(n)])
        end

    elseif typeof(n.dist) == Categorical{Float64, Vector{Float64}}
        if length(params_opt(n))==0 # non-optim case, use params directly
            p_leaf = params(n)
        else # optim case
            p_leaf = softmax(params_opt(n))
        end
        # 1. Laplace smoothing for Categorical
        K = length(p_leaf) # size of support
        p_new = (p_leaf .+ ϵ) ./ (1 + K * ϵ)
        dist_eps = Categorical(p_new)
        # 2. Fix test set contains discrete values outside the support of training set
        if logpdf(dist_eps, x[scope(n)]) == -Inf 
            if x[scope(n)] > length(p_new)
                # Concatenate the n.dist.p with zeros, final length to be the new value in test set
                δ = Int(x[scope(n)] - length(p_new))
                p_new = ([n.dist.p; zeros(δ)] .+ ϵ) ./ (1 + (K + δ) * ϵ)
                dist_eps = Categorical(p_new)
            end
        end
        return logpdf(Categorical(p_new), x[scope(n)])

    elseif typeof(n.dist) == AlphaStable{Float64}
        # use Gauss-Hermite quadrature to approximate the pdf
        t, w = gausshermite(50)
        f(x,t) = Distributions.cf(n.dist, t) * exp(-im * x * t) * exp(t^2)
        I = dot(w, f.(x[scope(n)], t))/(2*π)
        return real(I)
    end
end

@inline (n::UnivariateECF)(x::AbstractVector{<:Real}) = n(x, n.data)
@inline (n::UnivariateECF)(x::AbstractVector{<:Real}, θ...) = _logpdf(n, x, θ...)

function _logpdf(n::UnivariateECF, x::AbstractVector{<:Real}, data_x)
    @error("no PDF for ECF leaf")
end

(n::UnivariateNode)(x::AbstractMatrix{<:Real}) = @inbounds map(i -> n(view(x,i,:)), 1:size(x,1))
(n::UnivariateNode)(x::AbstractMatrix{<:Real}, θ...) = @inbounds map(i -> n(view(x,i,:), θ...), 1:size(x,1))

@inline logpdf(n::Leaf, x::AbstractArray{<:Real}) = n(x)
@inline logpdf(n::Leaf, x::AbstractArray{<:Real}, y...) = n(x, y...)

rand(rng::AbstractRNG, cc::CharacteristicCircuit) = rand(cc.root)

function rand(node::Node)
    @assert isnormalized(node)
    @assert hasscope(node)
    v = rand_(node)
    return map(d -> v[d], sort(scope(node)))
end

rand(node::UnivariateNode) = rand(node.dist)

function rand_(node::ProductNode)
    @assert isnormalized(node)
    @assert hasscope(node)
    return mapreduce(c -> rand_(c), merge, filter(c -> hasscope(c), children(node)))
end

function rand_(node::SumNode)
    @assert isnormalized(node)
    @assert hasscope(node)
    w = Float64.(weights(node))
    z = rand(Categorical(w / sum(w))) # Normalisation due to precision errors.
    # Generate observation by drawing from a child.
    return rand_(children(node)[z])
end

rand_(node::UnivariateNode) = Dict(node.scope => rand(node.dist))

