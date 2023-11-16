using Zygote
using Optimisers
using LinearAlgebra
using Statistics
using Random
using BayesNets
using AlphaStableDistributions
using FastGaussQuadrature
using UnicodePlots
using Revise # <- Revise keeps track of changes and updates the inluded libs if needed
include("../src/CharacteristicCircuits.jl")
using .CharacteristicCircuits
include("exp_helper.jl")

function train_cc(cc, x; iter=10, lr1=5, lr2=1)
    # Collect all sum nodes.
    snodes = filter(n -> isa(n, SumNode), values(cc))
    θs = logweights.(snodes)
    # Collect all leaf nodes
    lnodes = filter(n -> isa(n, UnivariateNode), values(cc))
    θl = params_opt.(lnodes)

    D = size(x, 2)
    # optimization
    loss = zeros(iter+1)
    loss_pdf = zeros(iter+1)
    loss[1] = CFD(cc, x; d=D)
    loss_pdf[1] = mean(logpdf(cc.root, x))
    @info 0, "cfd", loss[1]
    @info 0, "pdf", loss_pdf[1]
    for i in 1:iter
        # set lr
        @assert lr1 >= lr2
        lr_cur =  (- (lr1 -lr2) * i + (lr1*iter-lr2))/(iter-1)

        myg = Zygote.gradient(Params([θl; θs])) do
            CFD(cc, x; d=D, n=100)
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
        loss[i+1] = CFD(cc, x; d=D)
        loss_pdf[i+1] = mean(logpdf(cc.root, x))
        if mod(i,20)==0
            @info i, "cfd", loss[i+1]
            @info i, "pdf", loss_pdf[i+1]
        end
    end
    return cc, loss, loss_pdf
end

function inv_softmax(x; c=1.0, ϵ=ξ)
    return log.(x.+ϵ) .+ c
end

# load toy dataset 1
@info "**** MM ****"
experiment = generate_mm()

# data
x_train = experiment.train
x_test = experiment.test

# baseline: structure learning
cc_learn = deepcopy(experiment.cc_p)
@info "SL: from structure learning"
@info "cfd = " CFD(cc_learn, x_train; d=2, n=100)
@info "pdf = " mean(logpdf(cc_learn.root, x_train))
@info "pdf = " mean(logpdf(cc_learn.root, x_test))

# optim over structure learnerd model
cc1 = deepcopy(experiment.cc_p)
Random.seed!(111)
ξ=1e-5 # to avoid 0 bounds
lnodes = filter(n -> isa(n, UnivariateNode), values(cc1))
for (jl, n) in enumerate(lnodes)
    if typeof(n.dist) == Categorical{Float64, Vector{Float64}}
        n.opt_params = inv_softmax(Distributions.params(n.dist)[1])
    elseif typeof(n.dist) == Normal{Float64}
        n.opt_params = [Distributions.params(n.dist)[1], log(Distributions.params(n.dist)[2])]
    elseif typeof(n.dist) == AlphaStable{Float64}
        x_1 = -log(2/Distributions.params(n.dist)[1] - 1 + ξ)
        x_2 = -log(2/(Distributions.params(n.dist)[2]+1+ ξ) - 1 + ξ)
        n.opt_params = [x_1, x_2, log(Distributions.params(n.dist)[3]+ξ), Distributions.params(n.dist)[4]]
    end
    n.dist_params = Float64[]
end
cc1, loss1_cfd, loss1_pdf = train_cc(cc1, x_train; iter=40, lr1=0.5, lr2=0.05)
@info "SL & CFD: optim on structure learned model"
@info "cfd = " CFD(cc1, x_train; d=2, n=100)
@info "pdf = " mean(logpdf(cc1.root, x_train))
@info "pdf = " mean(logpdf(cc1.root, x_test))

lineplot(loss1_cfd)
lineplot(loss1_pdf)

# optim over structure learnerd model with random weights
cc2 = deepcopy(experiment.cc_p)
Random.seed!(111)
lnodes = filter(n -> isa(n, UnivariateNode), values(cc2))
for (jl, n) in enumerate(lnodes)
    if typeof(n.dist) == Categorical{Float64, Vector{Float64}}
        p_categorical = rand(3)
        n.opt_params = p_categorical./sum(p_categorical)
    elseif typeof(n.dist) == Normal{Float64}
        n.opt_params = [randn()+2.5, rand()]
    elseif typeof(n.dist) == AlphaStable{Float64}
        @error "notImplemented"
    end
    n.dist_params = Float64[]
end
cc2, loss2_cfd, loss2_pdf = train_cc(cc2, x_train; iter=300, lr1=0.5, lr2=0.005)

@info "SL (random w) & CFD"
@info "cfd = " CFD(cc2, x_train; d=2, n=100)
@info "pdf = " mean(logpdf(cc2.root, x_train))
@info "pdf = " mean(logpdf(cc2.root, x_test))

lineplot(loss2_cfd)
lineplot(loss2_pdf)

# optim over random structure
Random.seed!(111)
cc3 = generate_cc(x_train, :random, distributions = [Normal, Categorical])
updatescope!(cc3)
cc3, loss3_cfd, loss3_pdf = train_cc(cc3, x_train; iter=300, lr1=0.5, lr2=0.01)

@info "RS & CFD: optim on random structure"
@info "cfd = " CFD(cc3, x_train; d=2, n=100)
@info "pdf = " mean(logpdf(cc3.root, x_train))
@info "pdf = " mean(logpdf(cc3.root, x_test))

lineplot(loss3_cfd)
lineplot(loss3_pdf)

####################################
# load toy dataset 2
@info "**** BN ****"
experiment_bn = generate_bn()

# data
x2_train = experiment_bn.train
x2_test = experiment_bn.test

# baseline: structure learning
cc2_learn = deepcopy(experiment_bn.cc_p)
@info "SL: from structure learning"
@info "cfd = " CFD(cc2_learn, x2_train; d=5)
@info "pdf = " mean(logpdf(cc2_learn.root, x2_train))
@info "pdf = " mean(logpdf(cc2_learn.root, x2_test))

# optim over structure learnerd model
cc1 = deepcopy(experiment_bn.cc_p)
Random.seed!(111)
ξ=1e-5 # to avoid 0 bounds
lnodes = filter(n -> isa(n, UnivariateNode), values(cc1))
for (jl, n) in enumerate(lnodes)
    if typeof(n.dist) == Categorical{Float64, Vector{Float64}}
        n.opt_params = inv_softmax(Distributions.params(n.dist)[1])
    elseif typeof(n.dist) == Normal{Float64}
        n.opt_params = [Distributions.params(n.dist)[1], log(Distributions.params(n.dist)[2])]
    elseif typeof(n.dist) == AlphaStable{Float64}
        x_1 = -log(2/Distributions.params(n.dist)[1] - 1 + ξ)
        x_2 = -log(2/(Distributions.params(n.dist)[2]+1+ ξ) - 1 + ξ)
        n.opt_params = [x_1, x_2, log(Distributions.params(n.dist)[3]+ξ), Distributions.params(n.dist)[4]]
    end
    n.dist_params = Float64[]
end
cc1, loss1_cfd, loss1_pdf = train_cc(cc1, x2_train; iter=200, lr1=0.5, lr2=0.01)
@info "SL & CFD: optim on structure learned model"
@info "cfd = " CFD(cc1, x2_train; d=5)
@info "pdf = " mean(logpdf(cc1.root, x2_train))
@info "pdf = " mean(logpdf(cc1.root, x2_test))

lineplot(loss1_cfd)
lineplot(loss1_pdf)

# optim over structure learnerd model with random weights
cc2 = deepcopy(experiment_bn.cc_p)
Random.seed!(111)
lnodes = filter(n -> isa(n, UnivariateNode), values(cc2))
for (jl, n) in enumerate(lnodes)
    if typeof(n.dist) == Categorical{Float64, Vector{Float64}}
        p_categorical = rand(2)
        n.opt_params = p_categorical./sum(p_categorical)
    elseif typeof(n.dist) == Normal{Float64}
        n.opt_params = [randn()+4.5, rand()]
    elseif typeof(n.dist) == AlphaStable{Float64}
        @error "notImplemented"
    end
    n.dist_params = Float64[]
end
cc2, loss2_cfd, loss2_pdf = train_cc(cc2, x2_train; iter=40, lr1=1, lr2=0.05)
@info "SL (random w) & CFD"
@info "cfd = " CFD(cc2, x2_train; d=5)
@info "pdf = " mean(logpdf(cc2.root, x2_train))
@info "pdf = " mean(logpdf(cc2.root, x2_test))

lineplot(loss2_cfd)
lineplot(loss2_pdf)

# optim over random structure
Random.seed!(111)
cc3 = generate_cc(x2_train, :random, distributions = [Categorical, Categorical, Categorical, Categorical, Normal])
updatescope!(cc3)
cc3, loss3_cfd, loss3_pdf = train_cc(cc3, x2_train; iter=40, lr1=1, lr2=0.05)
@info "RS & CFD: optim on random structure"
@info "cfd = " CFD(cc3, x2_train; d=5)
@info "pdf = " mean(logpdf(cc3.root, x2_train))
@info "pdf = " mean(logpdf(cc3.root, x2_test))

lineplot(loss3_cfd)
lineplot(loss3_pdf)
