export generate_cc

"""
    generate_cc(X::Matrix, algo::Symbol; ...)

Generate a CC structure using structure learning.

Arguments:
* `X`: Data matrix.
* `algo`: Algorithm, one out of [:learncc, :learnecf, :random].
"""
function generate_cc(X::Matrix, algo::Symbol; params...)
    if algo == :learncc
        return learncc(X; params...)
    elseif algo == :learnecf
        return learnecf(X; params...)
    elseif algo == :random
        return randomcc(X; params...)
    else
        @error("Unknown structure learning algorithm: ", algo)
    end
end

"""
    learncc(X; distribution=Normal(), minclustersize=100)

Return CC learned by a simplified stracture learning algorithm.
"""
function learncc(X; distributions::Vector = map(d -> Normal, 1:size(X,2)), minclustersize=100, ϵ=1e-5, use_RDC=false, threshold=0.3)
    q = Queue{Tuple}()
    root = FiniteSumNode()
    instances = collect(1:size(X)[1])
    variables = collect(1:size(X)[2])
    enqueue!(q, (root, variables, instances))

    while length(q) > 0
        node, variables, instances = dequeue!(q)

        # stopping condition, one variable left, estimate distribution
        if length(variables) == 1
            v = variables[1]
            if distributions[v] == Categorical # should add other discrete types
                # get support and initialize a random vector
                slice = Int.(X[instances, v])#.+1 # fix data value starting at 0
                d_leaf = fit(distributions[v], Int(maximum(X[:,v])), slice)
                add!(node, UnivariateNode(d_leaf, v; dist_params=d_leaf.p))
            else
                slice = X[instances, v]
                if std(slice) > ϵ
                    try # fit distribution
                        add!(node, UnivariateNode(fit(distributions[v], slice), v))
                    catch e # if cannot fit alpha stable distribution, use its Normal case
                        add!(node, UnivariateNode(AlphaStable(α=2, β=0.0, scale=std(slice)/sqrt(2), location=mean(slice)), v))
                    end
                else # if all instances are (almost) the same 
                    if distributions[v] == Normal
                        add!(node, UnivariateNode(Normal(mean(slice), ϵ), v))
                    elseif distributions[v] == AlphaStable
                        add!(node, UnivariateNode(AlphaStable(α=2, β=0.0, scale=ϵ/sqrt(2), location=mean(slice)), v))
                    end
                end
            end
            continue
        end

        # stopping condition: too small cluster, factorize variables
        if length(instances) <= minclustersize
            # root doesn't have enough instances for clustering
            if typeof(node) <: SumNode
                prod = FiniteProductNode()
                add!(node, prod, log(1.0))
                node = prod
            end
            for v in variables
                if distributions[v] == Categorical # should add other discrete types
                    slice = Int.(X[instances, v])#.+1
                    d_leaf = fit(distributions[v], Int(maximum(X[:,v])), slice)
                    add!(node, UnivariateNode(d_leaf, v; dist_params=d_leaf.p))
                else
                    slice = X[instances, v]
                    if var(slice) > ϵ #std
                        try
                            add!(node, UnivariateNode(fit(distributions[v], slice), v))
                        catch e
                            add!(node, UnivariateNode(AlphaStable(α=2, β=0.0, scale=std(slice)/sqrt(2), location=mean(slice)), v))
                        end
                    else # if all instances are the same 
                        if distributions[v] == Normal
                            add!(node, UnivariateNode(Normal(mean(slice), ϵ), v))
                        elseif distributions[v] == AlphaStable
                            add!(node, UnivariateNode(AlphaStable(α=2, β=0.0, scale=ϵ/sqrt(2), location=mean(slice)), v))
                        end
                    end
                end
            end
            continue
        end

        # divide and conquer
        if isa(node, SumNode)
            clusterweights = cluster_instances(X, variables, instances)
            for (cluster, weight) in clusterweights
                prod = FiniteProductNode()
                add!(node, prod, log(weight))
                enqueue!(q, (prod, variables, cluster))
            end
        else  # isa(node, ProductNode)
            if use_RDC
                splits = split_variables_RDC(X, variables, instances, distributions; threshold)
            else
                splits = split_variables(X, variables, instances)
            end
            for split in splits
                if length(split) == 1
                    enqueue!(q, (node, split, instances))
                    continue
                end
                sum = FiniteSumNode()
                add!(node, sum)
                enqueue!(q, (sum, split, instances))
            end
        end
    end

    return CharacteristicCircuit(root)
end

"""
    learnecf(X; minclustersize=100)

Return CC learned by structure learning with ECF leaves.
"""
function learnecf(X; minclustersize=100)
    q = Queue{Tuple}()
    root = FiniteSumNode()
    instances = collect(1:size(X)[1])
    variables = collect(1:size(X)[2])
    enqueue!(q, (root, variables, instances))

    while length(q) > 0
        node, variables, instances = dequeue!(q)

        # stopping condition, one variable left, estimate distribution
        if length(variables) == 1
            v = variables[1]
            slice = X[instances, v]
            n = length(instances)
            # Create an ECF leaf node for the slice of data
            add!(node, UnivariateECF(slice, v))
            continue
        end

        # stopping condition: too small cluster, factorize variables
        if length(instances) <= minclustersize
            # root doesn't have enough instances for clustering
            if typeof(node) <: SumNode
                prod = FiniteProductNode()
                add!(node, prod, log(1.0))
                node = prod
            end
            for v in variables
                slice = X[instances, v]
                add!(node, UnivariateECF(slice, v))
            end
            continue
        end

        # divide and conquer
        if isa(node, SumNode)
            clusterweights = cluster_instances(X, variables, instances)
            for (cluster, weight) in clusterweights
                prod = FiniteProductNode()
                add!(node, prod, log(weight))
                enqueue!(q, (prod, variables, cluster))
            end
        else  # isa(node, ProductNode)
            splits = split_variables(X, variables, instances)
            for split in splits
                if length(split) == 1
                    enqueue!(q, (node, split, instances))
                    continue
                end
                sum = FiniteSumNode()
                add!(node, sum)
                enqueue!(q, (sum, split, instances))
            end
        end
    end

    return CharacteristicCircuit(root)
end

"""
    split_variables(X, variables, instances)

Split variables into two groups by a G-test based method.
"""
function split_variables(X, variables, instances)
    function binarize(x)
        binary_x = zeros(Int, size(x))
        binary_x[x .> mean(x)] .= 1
        return binary_x
    end
    @assert length(variables) > 1
    slice = X[instances, variables]
    distances = zeros(length(variables))
    p = sum(binarize(slice[:, rand(1:length(variables))]))/length(instances)
    for i in 1:length(variables)
        q = sum(binarize(slice[:, i]))/length(instances)
        e = (p + q)/2
        d = evaluate(KLDivergence(), [p, (1 - p), q, (1 - q)], [e, (1 - e), e, (1 - e)])
        distances[i] = d
    end
    dependentindex = partialsortperm(distances, 1:floor(Integer, length(variables)/2))
    splitone = variables[dependentindex]
    splittwo = setdiff(variables, splitone)

    return (splitone, splittwo)
end

"""
    cluster_instances(X, variables, instances)

Cluster instances into two groups by k-means clustering.
"""
function cluster_instances(X, variables, instances)
    slice = X[instances, variables]
    results = kmeans(transpose(slice), 2)
    clusterone = instances[results.assignments .== 1]
    clustertwo = setdiff(instances, clusterone)
    weight = length(clusterone)/length(instances)

    if length(clustertwo) == 0
        return ((clusterone, weight),)
    end
    return ((clusterone, weight), (clustertwo, 1 - weight))
end

"""
    randomcc(X; distribution=Normal())

Return CC structure generated by random splitting.
"""
function randomcc(X; distributions::Vector = map(d -> Normal, 1:size(X,2)))
    q = Queue{Tuple}()
    root = FiniteSumNode()
    instances = collect(1:size(X)[1])
    variables = collect(1:size(X)[2])
    enqueue!(q, (root, variables, instances))
    Ks = 2
    # Kp is fixed to 2 in this function
    X_ave = mean(X, dims=1)

    while length(q) > 0
        node, variables, instances = dequeue!(q)

        # stopping condition, one variable left, initialize distribution
        if length(variables) == 1
            v = variables[1]
            if distributions[v] == Categorical 
                # get support and initialize a random vector
                N_max = Int(maximum(X[:,v]))
                p_categorical = rand(N_max)
                θ_leaf = p_categorical./sum(p_categorical)
                add!(node, UnivariateNode(Categorical(θ_leaf), v; opt_params=θ_leaf))
            elseif distributions[v] == Normal 
                mu_l, sigma_l = randn() + X_ave[v], rand(); # initialize the Normal leaf with average of Data
                θ_leaf = [mu_l, sigma_l]
                add!(node, UnivariateNode(Normal(mu_l[], sigma_l[]), v; opt_params=θ_leaf))
            elseif distributions[v] == AlphaStable 
                α=rand()+1
                β=rand()
                scale=rand()
                location=rand() + X_ave[v]  # initialize the location with average of Data
                θ_leaf = [α, β, scale, location]
                add!(node, UnivariateNode(AlphaStable(2/(1 + exp(-α)), 2/(1 + exp(-β)) - 1, exp(scale), location), v; opt_params=θ_leaf))
            else
                @error("NotImplementedError")
            end
            continue
        end

        # divide and conquer
        if isa(node, SumNode)
            # random normalized weights
            p_sum = rand(Ks)
            p_sum = p_sum./sum(p_sum)
            for id_sum in 1:Ks
                prod = FiniteProductNode()
                add!(node, prod, log(p_sum[id_sum]))
                enqueue!(q, (prod, variables, instances))
            end
        else  # isa(node, ProductNode)
            # random splits of variables
            variables = shuffle!(variables)
            split_point = Int(round(length(variables)/2))
            splits = (variables[1:split_point], variables[split_point+1:length(variables)])
            for split in splits
                if length(split) == 1
                    enqueue!(q, (node, split, instances))
                    continue
                end
                sum = FiniteSumNode()
                add!(node, sum)
                enqueue!(q, (sum, split, instances))
            end
        end
    end

    return CharacteristicCircuit(root)
end

