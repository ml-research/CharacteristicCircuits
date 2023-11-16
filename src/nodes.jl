export CharacteristicCircuit
export CCNode, Node, Leaf, SumNode, ProductNode
export FiniteSumNode, FiniteProductNode
export UnivariateNode, UnivariateECF
export params_opt

export root, leaves

# Abstract definition of a CharacteristicCircuit node.
abstract type CCNode end
abstract type Node <: CCNode end
abstract type SumNode <: Node end
abstract type ProductNode <: Node end
abstract type Leaf <: CCNode end

header(node::CCNode) = "$(summary(node))($(node.id))"

function Base.show(io::IO, node::CCNode)
    println(io, header(node))
    if hasweights(node)
        println(io, "\tweights = $(weights(node))")
        println(io, "\tnormalized = $(isnormalized(node))")
    end
    if hasscope(node)
        println(io, "\tscope = $(scope(node))")
    else
        println(io, "\tNo scope set!")
    end
    if hasobs(node)
        println(io, "\tassigns = $(obs(node))")
    end
end

function Base.show(io::IO, node::Leaf)
    println(io, header(node))
    println(io, "\tscope = $(scope(node))")
    println(io, "\tparameters = $(params(node))")
end

struct CharacteristicCircuit
    root::Node
    nodes::Vector{<:CCNode}
    leaves::Vector{<:CCNode}
    idx::Dict{Symbol,Int}
    topological_order::Vector{Int}
    layers::Vector{AbstractVector{CCNode}}
    info::Dict{Symbol,<:Real}
end

function CharacteristicCircuit(root::Node)
    nodes = getOrderedNodes(root)
    leaves = filter(n -> isa(n, Leaf), nodes)
    idx = Dict(n.id => indx for (indx, n) in enumerate(nodes))
    toporder = collect(1:length(nodes))

    maxdepth = depth(root)
    nodedepth = map(n -> depth(n), nodes)
    layers = Vector{Vector{CCNode}}(undef, maxdepth+1)
    for d in 0:maxdepth
        layers[d+1] = nodes[findall(nodedepth .== d)]
    end

    return CharacteristicCircuit(root, nodes, leaves, idx, toporder, layers, Dict{Symbol, Real}())
end

Base.keys(cc::CharacteristicCircuit) = keys(cc.idx)
Base.values(cc::CharacteristicCircuit) = cc.nodes
Base.getindex(cc::CharacteristicCircuit, i...) = getindex(cc.nodes, cc.idx[i...])
Base.setindex!(cc::CharacteristicCircuit, v, i...) = setindex!(cc.nodes, v, cc.idx[i...])
Base.length(cc::CharacteristicCircuit) = length(cc.nodes)
function Base.show(io::IO, cc::CharacteristicCircuit)
    println(io, summary(cc))
    println(io, "\t#nodes = $(length(cc))")
    println(io, "\t#leaves = $(length(cc.leaves))")
    println(io, "\tdepth = $(length(cc.layers))")
end
leaves(cc::CharacteristicCircuit) = cc.leaves
root(cc::CharacteristicCircuit) = cc.root

"""
   FiniteSumNode <: SumNode

A sum node computes a convex combination of its weight and the CF of its children.

## Usage:

```julia
node = FiniteSumNode{Float64}(;D=4, N=1)
add!(node, ..., log(0.5))
add!(node, ..., log(0.5))
cf(node, rand(4))
```

"""
mutable struct FiniteSumNode{T<:Real} <: SumNode
    id::Symbol
    parents::Vector{<:Node}
    children::Vector{<:CCNode}
    logweights::Vector{T}
    scopeVec::BitArray{1}
    obsVec::BitArray{1}
end

function FiniteSumNode{T}(;D=1, N=1, parents::Vector{<:Node}=Node[]) where {T<:Real}
    return FiniteSumNode{T}(gensym(:sum), parents, CCNode[], T[], falses(D), falses(N))
end
function FiniteSumNode(;D=1, N=1, parents::Vector{<:Node}=Node[])
    return FiniteSumNode{Float64}(;D=D, N=N, parents=parents)
end

eltype(::Type{FiniteSumNode{T}}) where {T<:Real} = T
eltype(n::CCNode) = eltype(typeof(n))

"""
   FiniteProductNode <: ProductNode

A product node computes a product of the CF of its children.

## Usage:

```julia
node = FiniteProductNode(;D=4, N=1)
add!(node, ...)
add!(node, ...)
cf(node, rand(4))
```

"""
mutable struct FiniteProductNode <: ProductNode
    id::Symbol
    parents::Vector{<:Node}
    children::Vector{<:CCNode}
    scopeVec::BitArray{1}
    obsVec::BitArray{1}
end

function FiniteProductNode(;D=1, N=1, parents::Vector{<:Node} = Node[])
    return FiniteProductNode(gensym(:prod), parents, CCNode[], falses(D), falses(N))
end

"""
   UnivariateNode <: Leaf

A univariate node evaluates the CF of its univariate distribution.

## Usage:

```julia
distribution = Normal()
dimension = 1
node = UnivariateNode(distribution, dimension)
cf(node, rand(2)) # == cf(Normal(), rand(2))
```

"""
mutable struct UnivariateNode{T<:Real} <: Leaf
    id::Symbol
    parents::Vector{<:Node}
    dist::UnivariateDistribution
    dist_params::Vector{T}
    opt_params::Vector{T}
    scope::Int
end

function UnivariateNode(distribution::T, dim::Int; dist_params::Vector{<:Real} = Float64[], opt_params::Vector{<:Real} = Float64[], parents::Vector{<:Node} = Node[]) where {T<:UnivariateDistribution}
    return UnivariateNode(gensym(:univ), parents, distribution, dist_params, opt_params, dim)
end
params(n::UnivariateNode) = n.dist_params
params_opt(n::UnivariateNode) = n.opt_params
#params(n::UnivariateNode) = Distributions.params(n.dist)

"""
   UnivariateECF <: Leaf

A univariate node evaluates the empirical characteristic function of its univariate distribution.

## Usage:

```julia
data_x = rand(Float64, n)
dimension = 1
node = UnivariateECF(data_x, dimension)
cf(node, rand(2))
```

"""
mutable struct UnivariateECF <: Leaf
    id::Symbol
    parents::Vector{<:Node}
    data::AbstractVector{<:Real} 
    scope::Int
end

function UnivariateECF(data_x::T, dim::Int; parents::Vector{<:Node} = Node[]) where {T<:AbstractVector}
    return UnivariateECF(gensym(:univ), parents, data_x, dim)
end
params(n::UnivariateECF) = (n.data) 