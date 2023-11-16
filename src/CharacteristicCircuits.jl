module CharacteristicCircuits

# loading dependencies into workspaces
using Reexport
@reexport using Distributions
using StatsFuns
using SpecialFunctions
using AxisArrays
using DataStructures
using Distances
using Clustering
using FastGaussQuadrature
using AlphaStableDistributions
using StatsBase: countmap
using PyCall
using Combinatorics
using Graphs

import Base: getindex, map, parent, length, size, show, isequal, getindex, keys, eltype, rand
import Random: Repetition, SamplerSimple, Sampler
import Distributions.logpdf
import StatsBase.nobs
import StatsBase.denserank

# add Base modules
@reexport using Statistics
using LinearAlgebra
using SparseArrays
using Random
using Printf

# include general implementations
include("nodes.jl")
include("nodeFunctions.jl")

# include approach specific implementations
include("structureUtilities.jl")
include("pyRDC.jl")

# include utilities
include("utilityFunctions.jl")

end # module
