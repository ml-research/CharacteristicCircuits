export split_variables_RDC

sklearn = pyimport_conda("sklearn.cross_decomposition", "scikit-learn") # where to put this line?

# This script is developed based on 
# https://github.com/SPFlow/SPFlow/blob/master/src/spn/algorithms/splitting/RDC.py
"
    split_variables_RDC(X, variables, instances)

RDC implementation in julia
"
function split_variables_RDC(X, variables, instances, dist_list; threshold = 0.3)
    slice = X[instances, variables]
    domain = maximum(X, dims=1)
    dist = dist_list[variables]

    # get adjancency matrix
    rdc_adjacency_matrix = rdc_test(slice, domain[variables], dist)

    # thresholding
    rdc_adjacency_matrix[rdc_adjacency_matrix .< threshold] .= 0
    rdc_adjacency_matrix[rdc_adjacency_matrix .>= threshold] .= 1
    g=SimpleGraph(rdc_adjacency_matrix)
    
    # get connected components
    result = zeros(size(rdc_adjacency_matrix)[1])
    c_c = connected_components(g)
    for (i,c) in enumerate(c_c)
        result[c] .= i
    end

    # create the splits with Tuple, need to improve the code
    splits = ()
    for i in 1:length(c_c)
        splits = (variables[result.==i], splits...)
    end
    @info splits

    return splits
end


function rdc_test(slice, domain, dist)
    D = size(slice)[2] # n_variables
    pairwise_comparisons = collect(combinations(1:D,2))
    
    rdc_features = rdc_transformer(slice, domain, dist)
    rdc_adjacency_matrix = zeros(D, D)
    
    for (i,j) in pairwise_comparisons
        rdc_val = rdc_cca(i,j,rdc_features)
        # fill rdc values to matrix
        rdc_adjacency_matrix[i, j] = rdc_val[]
        rdc_adjacency_matrix[j, i] = rdc_val[]
    end
    # set diagonal to 1
    rdc_adjacency_matrix[diagind(rdc_adjacency_matrix)] .= 1.0
    
    return rdc_adjacency_matrix
end


"
rdc_cca(i,j,rdc_features)
Canonical Correlation Analysis
"
function rdc_cca(i,j, rdc_features; CCA_MAX_ITER=100)
    pycca = sklearn.CCA(n_components=1, max_iter=CCA_MAX_ITER)
    pycca.fit(rdc_features[i], rdc_features[j])
    X_cca, Y_cca = pycca.transform(rdc_features[i], rdc_features[j])
    rdc = cor(X_cca, Y_cca)
    # if cor() returns NaN (in the case X or Y is constant), set it to 0.0, meaning no correlation
    if isnan(rdc[1])
        return 0.0
    else
        return rdc
    end
end

"
Given a data_slice,
return a transformation of the features data in it according to the rdc
pipeline:
1 - empirical copula transformation
2 - random projection into a k-dimensional gaussian space
3 - pointwise  non-linear transform
"
function rdc_transformer(slice, domain, dist)
    N, D = size(slice)
    list_output = Any[]
    k = 10
    for f in 1:D
        if dist[f] == Categorical
            features = ohe_data(slice[:, f], Int(domain[f]))
        else
            features = reshape(slice[:,f], :,1) # forcing two columness
        end

        # 1. 
        # transform through the empirical copula
        features = empirical_copula_transformation(features, dist[f])

        # 2.
        # random projection through a gaussian
        random_gaussians = rand(Normal(0,1), (size(features)[2], k))
        s = 1.0/6.0 # default?
        rand_proj_features = s / size(features)[2] * (features * random_gaussians)

        # 3.
        # non_linearity
        nl_rand_proj_features = sin.(rand_proj_features)
        # append to a list, c.f. SPFlow code
        push!(list_output, hcat(nl_rand_proj_features, ones(size(nl_rand_proj_features)[1], 1)))
    end
    return list_output
end

function ohe_data(data, d_max)
    # not efficient but works?
    domain = 1:d_max
    dataenc = Float64.(Int.(repeat(data, 1,d_max)) .== transpose(repeat(domain, 1, size(data)[1])))

    return dataenc
end

function empirical_copula_transformation(features, dist)
    ones_matrix = ones((size(features)[1], size(features)[2]+1))
    for i in 1:size(features)[2]
        ones_matrix[:,i] = ecdf(features[:,i], dist)
    end
    return ones_matrix
end

"
    ecdf(X)
Empirical cumulative distribution function
for data X (one dimensional, if not it is linearized first)    
"
function ecdf(X, dist)
    N = size(X)[1]
    den = denserank(X)
    if dist == Categorical
    # if discrete
        # cumulative counts of each unique value
        c = length(unique(X))
        count = zeros(c)
        x = Int.(sort(X))
        for i in 1:N
            count[x[i]+1:end].+=1
        end
        result = count[den] ./ N
    else
    # if continuous 
    result = den ./ N
    end
    
    return result
end

