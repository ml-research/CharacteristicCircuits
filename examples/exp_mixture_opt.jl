## experiments on heterogeneous data 
## with random structure and parameter learning
using ArgParse
using Random
using FileIO
using JLD2
using Logging
using HDF5
using AlphaStableDistributions
using FastGaussQuadrature
using Zygote
using Optimisers
using Revise # <- Revise keeps track of changes and updates the inluded libs if needed
using Plots
include("../src/CharacteristicCircuits.jl")
using .CharacteristicCircuits


s = ArgParseSettings()
@add_arg_table s begin
    "--outputfolder"
        #help = "Path to folder in which output files should be stored."
        default = "results_opt/"
        arg_type = String
    "--datafolder"
        #help = "Path to folder containing datasets."
        default = "./data_predictions_scripts/datasets/islv-hdf5"
        arg_type = String
    "--leaftype"
        #help = "Leaf type of continuous RVs."
        default = "AlphaStable"
        arg_type = String
    "--epsilon"
        #help = "minimal variance value of the data slice."
        default = 1e-5
        arg_type = Float64
    "--lr1"
        #help = "learning rate."
        default = 1.0
        arg_type = Float64
    "--lr2"
        #help = "learning rate."
        default = 1.0
        arg_type = Float64
    "--iter"
        #help = "iteration for training."
        default = 5
        arg_type = Int64
    "--structure"
        #help = "random or learncc."
        default = "random"
        arg_type = String
    "dataset"
        #help = "Dataset name, assuming naming convention of 20 terrible datasets."
        required = true
        arg_type = String
    
end


args = parse_args(s)
Random.seed!(1234)

# read the args
ϵ = args["epsilon"]
lr1 = args["lr1"]
lr2 = args["lr2"]
iter = args["iter"]
CCstructure = args["structure"]
if args["leaftype"]=="Normal" 
    leaftype = Normal 
elseif args["leaftype"]=="AlphaStable"
    leaftype = AlphaStable
else
    throw(UndefVarError(:leaftype))
end

# Load the dataset
dataset = args["dataset"]
datafolder = args["datafolder"]

# results dir
resultsdir = args["outputfolder"] * dataset
if !isdir(resultsdir)
    mkpath(resultsdir)
end

@info "Processing $dataset in $datafolder"
@info "Storing results in $resultsdir"

# Output files.
llhfile = joinpath(resultsdir, "llhfile.jld2")

# Write out the configuration of the run.
save(joinpath(resultsdir, "configuration.jld2"), Dict("configuration" => args))

# function to optimize the CC
function train_cc(cc, x; iter=10, lr1=5, lr2=1)
    # Collect all sum nodes.
    snodes = filter(n -> isa(n, SumNode), values(cc))
    θs = logweights.(snodes)
    # Collect all leaf nodes
    lnodes = filter(n -> isa(n, UnivariateNode), values(cc))
    θl = params_opt.(lnodes)

    # get data dim
    D = size(x, 2)
    # optimization
    loss = zeros(iter+1)
    loss_pdf = zeros(iter+1)
    loss[1] = CFD(cc, x; σ=1, d=D, n=100)
    loss_pdf[1] = mean(logpdf(cc.root, x))
    @info 0, "cfd", loss[1]
    println(0, " cfd ", loss[1])
    @info 0, "pdf", loss_pdf[1]
    println(0, " pdf ", loss_pdf[1])
    for i in 1:iter
        # set lr
        @assert lr1 >= lr2
        lr_cur =  (- (lr1 -lr2) * i + (lr1*iter-lr2))/(iter-1)
        # get gradient
        myg = Zygote.gradient(Params([θl; θs])) do
            CFD(cc, x; σ=1, d=D, n=100)
        end
        
        for (js, n) in enumerate(snodes)
            if !isnothing(myg[θs[js]])
                n.logweights[:] = log.(projectToPositiveSimplex!(exp.(n.logweights[:] - lr_cur * myg[θs[js]])))
            end
        end
        for (jl, n) in enumerate(lnodes)
            if !isnothing(myg[θl[jl]])
                n.opt_params[:] = n.opt_params[:] - lr_cur * myg[θl[jl]]
            end
            if typeof(n.dist) == AlphaStable{Float64}
                α = 2/(1 + exp(-n.opt_params[1])) # apply parameter constraints
                β = 2/(1 + exp(-n.opt_params[2])) - 1
                c = exp(n.opt_params[3])
                μ = n.opt_params[4]
                n.dist = AlphaStable(α, β, c, μ)
            end
        end
        loss[i+1] = CFD(cc, x; σ=1, d=D, n=100)
        loss_pdf[i+1] = mean(logpdf(cc.root, x))
        if mod(i,20)==0
            @info i, "cfd", loss[i+1]
            println(i, " cfd ", loss[i+1])
            @info i, "pdf", loss_pdf[i+1]
            println(i, " pdf ", loss_pdf[i+1])
        end
    end
    return cc, loss, loss_pdf
end

function inv_softmax(x; c=1.0, ϵ=1e-5)
    return log.(x.+ϵ) .+ c
end

# Start script.
log_file = "log_"*string(leaftype)*"_random_"*string(ϵ)*".txt"
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

        println("Initialise CC structure")

        if CCstructure == "random"
            # random circuit with parametric leaf 
            cc = generate_cc(x_train, :random, distributions = dist_list)  
            updatescope!(cc)
        elseif CCstructure == "learncc"
            # structure learning
            cc = generate_cc(x_train, :learncc, distributions = dist_list, minclustersize=100, ϵ=ϵ)  
            updatescope!(cc)
            llh_train_0 = mean(logpdf(cc, x_train))
            println("logpdf from training set = ", llh_train_0)
            # set opt_params for parameter learning
            lnodes = filter(n -> isa(n, UnivariateNode), values(cc))
            for (jl, n) in enumerate(lnodes)
                if typeof(n.dist) == Categorical{Float64, Vector{Float64}}
                    n.opt_params = inv_softmax(Distributions.params(n.dist)[1])
                elseif typeof(n.dist) == Normal{Float64}
                    n.opt_params = [Distributions.params(n.dist)[1], log(Distributions.params(n.dist)[2])]
                elseif typeof(n.dist) == AlphaStable{Float64}
                    x_1 = -log(2/Distributions.params(n.dist)[1] - 1 + 1e-5)
                    x_2 = -log(2/(Distributions.params(n.dist)[2]+1+ 1e-5) - 1 + 1e-5)
                    n.opt_params = [x_1, x_2, log(Distributions.params(n.dist)[3]+1e-5), Distributions.params(n.dist)[4]]
                end
                n.dist_params = Float64[]
            end
        else
            @error("only random or learncc supported")
        end

        # estimate the logpdf on training/validation/test data before training
        llh_train_0 = mean(logpdf(cc, x_train))
        println("logpdf from training set = ", llh_train_0)
        @info "logpdf from training set = $llh_train_0"

        llh_valid_0 = mean(logpdf(cc, x_valid))
        println("logpdf from valid set = ", llh_valid_0)
        @info "logpdf from valid set = $llh_valid_0"

        llh_test_0 = mean(logpdf(cc, x_test))
        println("logpdf from test set = ", llh_test_0)
        @info "logpdf from test set = $llh_test_0"

        # optimize the parameters of the random circuit
        cc, loss, loss_pdf = train_cc(cc, x_train; iter=iter, lr1=lr1, lr2=lr2)

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

        save(llhfile, "train", llh_train, "test", llh_test, "cfd_loss", loss, "pdf_loss", loss_pdf)
    end
end

