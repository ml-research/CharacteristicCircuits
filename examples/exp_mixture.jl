## experiments on heterogeneous data
using ArgParse
using Random
using FileIO
using JLD2
using Logging
using HDF5
using AlphaStableDistributions
using FastGaussQuadrature
using Revise # <- Revise keeps track of changes and updates the inluded libs if needed
include("../src/CharacteristicCircuits.jl")
using .CharacteristicCircuits

Logging.disable_logging(Logging.Warn)

s = ArgParseSettings()
@add_arg_table s begin
    "--outputfolder"
        #help = "Path to folder in which output files should be stored."
        default = "results/"
        arg_type = String
    "--datafolder"
        #help = "Path to folder containing datasets."
        default = "./data_predictions_scripts/datasets/islv-hdf5"
        arg_type = String
    "--leaftype"
        #help = "Leaf type of continuous RVs."
        default = "Normal"
        arg_type = String
    "--minclustersize"
        #help = "min_k in structure learning."
        default = 100
        arg_type = Int
    "--epsilon"
        #help = "minimal variance value of the data slice."
        default = 1e-5
        arg_type = Float64
    "--threshold"
        #help = "threshold in RDC adjacency matrix."
        default = 0.3
        arg_type = Float64
    "--rdc"
        #help = "choose algorithm of splitting at product node, Gtest or rdc"
        default = "Gtest"
        arg_type = String
    "dataset"
        #help = "Dataset name, assuming naming convention of 20 terrible datasets."
        required = true
        arg_type = String
    
end


args = parse_args(s)
Random.seed!(1234)

# Read the args
minclustersize = args["minclustersize"]
ϵ = args["epsilon"]
if args["leaftype"]=="Normal" 
    leaftype = Normal 
elseif args["leaftype"]=="AlphaStable"
    leaftype = AlphaStable
else
    throw(UndefVarError(:leaftype))
end
if args["rdc"]=="rdc" 
    use_RDC=true
else
    use_RDC=false
end
rdc_threshold = args["threshold"]

# Load the dataset
dataset = args["dataset"]
datafolder = args["datafolder"]

# Results dir
resultsdir = args["outputfolder"] * dataset
if !isdir(resultsdir)
    mkpath(resultsdir)
end

@info "Processing $dataset in $datafolder"
@info "Storing results in $resultsdir"

# Output files
llhfile = joinpath(resultsdir, "llhfile.jld2")

# Write out the configuration of the run
save(joinpath(resultsdir, "configuration.jld2"), Dict("configuration" => args))

# Start script.
log_file = "log_"*string(leaftype)*"_"*string(minclustersize)*"_"*string(ϵ)*".txt"
open(joinpath(resultsdir, log_file), "w+") do io
    logger = SimpleLogger(io)
    with_logger(logger) do
        if Threads.nthreads() == 1
            @info "Run export JULIA_NUM_THREADS=nproc before running julia to use multithreading."
        else
            @info "Using $(Threads.nthreads()) threads."
            # Test if all threads are available.
            thrsa = zeros(Threads.nthreads())
            Threads.@threads for t in 1:Threads.nthreads()
                thrsa[t] = Threads.threadid()
            end

            avThreads = length(unique(thrsa[thrsa .> 0]))
            @assert avThreads == Threads.nthreads() "Only $(avThreads) threads available."
        end

        @info log_file

        @info "Loading dataset: $dataset stored under: $datafolder"

        x_train = map(Float64, h5read(joinpath(datafolder, dataset*"PP", "train.hdf5"), "train"))
        x_train = Matrix(transpose(x_train))

        x_valid = map(Float64, h5read(joinpath(datafolder, dataset*"PP", "valid.hdf5"), "valid"))
        x_valid = transpose(x_valid)

        x_test = map(Float64, h5read(joinpath(datafolder, dataset*"PP", "test.hdf5"), "test"))
        x_test = transpose(x_test)

        (N_train, D) = size(x_train)
        (N_valid, _) = size(x_valid)
        (N_test, _) = size(x_test)

        @info "N_train: $N_train , N_test: $N_test , D: $D"
        println("N_train: ", N_train, "N_valid: ", N_valid, " N_test: ", N_test, " D: ", D)

        # Preprocessing
        @info "Processing dataset of size N=$N_train, D=$D."
    
        # mean imputation of NaN values
        if any(isnan, x_train)
            # find dims with NaN from the mean vector
            mean_vec = mean(x_train, dims=1)
            mean_index = findall(isnan, mean_vec)
            for j in 1:length(mean_index)
                data_index = findall(isnan, x_train[:,mean_index[j][2]])
                data_mean = round(mean(deleteat!(x_train[:,mean_index[j][2]], data_index)))
                println(data_mean)
                x_train[data_index, mean_index[j][2]].=data_mean
            end
        end 

        if any(isnan, x_valid)
            # find dims with NaN from the mean vector
            mean_vec = mean(x_valid, dims=1)
            mean_index = findall(isnan, mean_vec)
            for j in 1:length(mean_index)
                data_index = findall(isnan, x_valid[:,mean_index[j][2]])
                data_mean = round(mean(deleteat!(x_valid[:,mean_index[j][2]], data_index)))
                println(data_mean)
                x_valid[data_index, mean_index[j][2]].=data_mean
            end
        end 

        if any(isnan, x_test)
            # find dims with NaN from the mean vector
            mean_vec = mean(x_test, dims=1)
            mean_index = findall(isnan, mean_vec)
            for j in 1:length(mean_index)
                data_index = findall(isnan, x_test[:,mean_index[j][2]])
                data_mean = round(mean(deleteat!(x_test[:,mean_index[j][2]], data_index)))
                println(data_mean)
                x_test[data_index, mean_index[j][2]].=data_mean
            end
        end 
        
        # select discrete and continuous dims
        dids = findall(map(d -> all(isinteger, filter(x -> !isnan(x), x_train[:,d])), 1:D))
        dids = filter(d -> length(filter(x -> !isnan(x), unique(x_train[:,d]))) < 20, 1:D)
        for d in dids
            ix = findall(!isnan, x_train[:,d])
            l = unique(x_train[ix,d])

            x_train[ix,d] = map(x -> findfirst(l .== x), x_train[ix,d])

            ix = findall(!isnan, x_valid[:,d])
            x_valid[ix,d] = map(x -> x ∈ l ? findfirst(l .== x) : length(l)+1, x_valid[ix,d])
            ix = findall(!isnan, x_test[:,d])
            x_test[ix,d] = map(x -> x ∈ l ? findfirst(l .== x) : length(l)+1, x_test[ix,d])
        end

        # to check the support of discrete RVs
        for kk in dids
            println("******* D = ", kk, " **********")
            println(sort(unique(x_train[:,kk])))
            println(sort(unique(x_test[:,kk])))
            println("*****************")
        end

        dist_list = Array{UnionAll, 1}(undef, D)
        dist_list[:] .= leaftype
        dist_list[dids] .= Categorical
        C_list = 0 ./ float((dist_list .==leaftype)) .+ 1
        D_list = 0 ./ float((dist_list .==Categorical)) .+ 1

        println("Learn Characteristic Circuit")

        # learn circuit with parametric leaf 
        cc = generate_cc(x_train, :learncc, distributions = dist_list, minclustersize=100, ϵ=ϵ, use_RDC=use_RDC, threshold=rdc_threshold)  
        updatescope!(cc)
        println(cc)
        
        # estimate the logpdf on training/validation/test data
        llh_train = mean(logpdf(cc, x_train))
        println("logpdf from training set = ", llh_train)
        @info "logpdf from training set = $llh_train"

        llh_valid = mean(logpdf(cc, x_valid))
        println("logpdf from valid set = ", llh_valid)
        @info "logpdf from valid set = $llh_valid"

        llh_test = mean(logpdf(cc, x_test))
        println("logpdf from test set = ", llh_test)
        @info "logpdf from test set = $llh_test"

        save(llhfile, "train", llh_train, "test", llh_test)
    end
end
