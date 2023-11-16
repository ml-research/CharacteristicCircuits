using ForwardDiff
using Optimisers
using Distributions
using JLD
using Plots
include("../src/CharacteristicCircuits.jl")
using .CharacteristicCircuits

# Load the MM data and models generated from the following script
include("exp_helper.jl")
experiment = generate_mm()

function test_optim(cc; σ₀ = 1.0, iterations = 1000)
    # Find the σ that maximizes the CFD(gt, cc; σ)
    gt = experiment.gt

    f(σ) = CFD(gt, cc; σ=exp(σ[]), d=2, n=100)
    ℓ = zeros(iterations)
    x = [log(σ₀)]

    state = Optimisers.setup(Optimisers.Descent(1.0), x)

    for i in 1:iterations
        g = ForwardDiff.gradient(f, x)[]

        state, x = Optimisers.update(state, x, -g)
        
        ℓ[i] = f(x)
    end

    return (σ = exp(x[]), loss = ℓ)
end

function calc_cfd(σ_list; n_repeat = 5, k = 1000)
    gt = experiment.gt
    cc_n = experiment.cc_n
    cc_e = experiment.cc_e
    cc_p = experiment.cc_p
    
    d = size(experiment.train)[2]
    cfd_list = zeros(length(σ_list),4,n_repeat)
    for nn in 1:n_repeat
        println("********")
        for j in eachindex(σ_list)
            σ = exp(σ_list[j])
            println("σ = exp ", σ_list[j] )
            
            cfd1 = CFD(gt, experiment.train; σ=σ, d=d, n=k)
            cfd2 = CFD(gt, cc_e; σ=σ, d=d, n=k)
            cfd3 = CFD(gt, cc_p; σ=σ, d=d, n=k)
            cfd4 = CFD(gt, cc_n; σ=σ, d=d, n=k)

            cfd_list[j, 1, nn] = cfd1
            cfd_list[j, 2, nn] = cfd2
            cfd_list[j, 3, nn] = cfd3
            cfd_list[j, 4, nn] = cfd4
        end
    end
    cfd_mean = reshape(mean(cfd_list, dims=[3]), size(cfd_list)[1], size(cfd_list)[2])
    cfd_std = reshape(std(cfd_list, dims=[3]), size(cfd_list)[1], size(cfd_list)[2])
    return cfd_mean, cfd_std
end

function calc_optim(n_repeat)
    cc_n = experiment.cc_n
    cc_p = experiment.cc_p
    cc_e = experiment.cc_e
    cfd_list = zeros(n_repeat, 3)
    σ_list = zeros(n_repeat, 3)
    for i in 1:n_repeat
        println("***", i, "***")
        r_n = test_optim(cc_n; iterations = 100)
        r_p = test_optim(cc_p; iterations = 100)
        r_e = test_optim(cc_e; iterations = 100)
        cfd_list[i,:] = [r_n.loss[100], r_p.loss[100], r_e.loss[100]]
        σ_list[i,:] = [r_n.σ, r_p.σ, r_e.σ]
    end
    # Return the average of CFD, and std of CFD and the average of the optimized σ
    return mean(cfd_list, dims=[1]), std(cfd_list, dims=[1]), mean(σ_list, dims=[1])
end

function plot_mm(σ_list, cfd_mean, cfd_std, optim_mean, optim_σ)
    plot(σ_list, cfd_mean[:,1], label="ECF", linewidth=3, 
    title="CFD with varying σ", xlabel = "log(σ)", ylabel = "CFD", 
    ribbon=(cfd_std[:,1], cfd_std[:,1]), fillalpha=.4, 
    xtickfontsize=12,ytickfontsize=12, xguidefontsize=17, 
    yguidefontsize=17,legendfontsize=12, titlefontsize=18, color="blue")
    plot!(σ_list, cfd_mean[:,2], label="CC-E", linewidth=3,  
    ribbon=(cfd_std[:,2], cfd_std[:,2]), fillalpha=.4, 
    xtickfontsize=12,ytickfontsize=12, xguidefontsize=17, 
    yguidefontsize=17,legendfontsize=12, titlefontsize=18, color="red")
    plot!(σ_list, cfd_mean[:,3], label="CC-P", linewidth=3, 
    ribbon=(cfd_std[:,3], cfd_std[:,3]), fillalpha=.4, 
    xtickfontsize=12,ytickfontsize=12, xguidefontsize=17, 
    yguidefontsize=17,legendfontsize=12, titlefontsize=18, linecolor="purple")
    scatter!([log.(optim_σ)[3]], [optim_mean[3]], seriestype=:scatter, 
    label="max", markersize = 5, color="red")
    scatter!([log.(optim_σ)[2]], [optim_mean[2]], seriestype=:scatter, 
    label="max", markersize = 5, color="purple")
end

function main_mm()
    # Optim std 
    n_repeat = 5
    @time m_cfd_list, std_cfd_list, optim_σ = calc_optim(n_repeat)
    # CFD with changing σ + optim σ
    σ_list = Vector(-5:0.5:6)
    σ_list_new = sort(vcat(σ_list, log.(optim_σ[:])))
    cfd_mean, cfd_std = calc_cfd(σ_list_new; n_repeat = 5, k = 1000)
    optim_mean = [maximum(cfd_mean[:,4]), maximum(cfd_mean[:,3]), maximum(cfd_mean[:,2])]
    # Save with JLD
    save("mm.jld", "σ_list_new", σ_list_new, "cfd_mean", cfd_mean, "cfd_std", cfd_std, "optim_mean", optim_mean, "optim_σ", optim_σ)
    #results_saved = load("mm.jld")
    plot_ref = plot_mm(σ_list_new, cfd_mean, cfd_std, optim_mean, optim_σ[1:3])
    Plots.savefig(plot_ref, "mm_cfd.pdf") 
end

main_mm()

