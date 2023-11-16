using Random
using Revise # <- Revise keeps track of changes and updates the inluded libs if needed

function generate_mm(; seed = 111, D = 2, Ntrain = 800, Ntest = 200, mink = 100)

    Random.seed!(seed)
    # generate data from MM, using a hand crafted CC
    rootnode = FiniteSumNode();
    # Add two product nodes to the root.
    add!(rootnode, FiniteProductNode(), log(0.3));
    add!(rootnode, FiniteProductNode(), log(0.7));
    # add leaf as child of 1. Prod node
    prod_1 = CharacteristicCircuits.children(rootnode)[1];
    add!(prod_1, UnivariateNode(Normal(0,1), 1));
    add!(prod_1, UnivariateNode(Categorical([0.6, 0.4, 0.0]), 2));
    # add leaf as child of 2. Prod node
    prod_2 = CharacteristicCircuits.children(rootnode)[2];
    add!(prod_2, UnivariateNode(Normal(5,1), 1));
    add!(prod_2, UnivariateNode(Categorical([0.1, 0.2, 0.7]), 2));

    cc = CharacteristicCircuit(rootnode);
    updatescope!(cc)

    # sample data from cc
    X = zeros(Ntrain+Ntest, D)
    for i in 1:Ntrain+Ntest
        X[i,:] = rand(cc)
    end
    x_train = X[1:Ntrain,:]
    x_test = X[(Ntrain+1):(Ntrain+Ntest),:]

    ## learn the CCs
    # learn CC-E
    cc_e = generate_cc(x_train, :learnecf, minclustersize=mink)
    updatescope!(cc_e)

    # learn CC-N
    cc_n = generate_cc(x_train, :learncc, minclustersize=mink)
    updatescope!(cc_n)

    # learn CC-P
    cc_p = generate_cc(x_train, :learncc, distributions = [Normal, Categorical], minclustersize=mink)
    updatescope!(cc_p)

    return (gt = cc, cc_e = cc_e, cc_n = cc_n, cc_p = cc_p, train = x_train, test = x_test)
end

function generate_bn(; seed = 111, D = 5, Ntrain = 800, Ntest = 200, mink = 100)

    Random.seed!(seed)

    bn = BayesNet()
    Pa_1 = 0.3; Pa_2 = 0.7
    push!(bn, StaticCPD(:a, Categorical([Pa_1,Pa_2])))
    Pb_1 = 0.8; Pb_2 = 0.2
    push!(bn, StaticCPD(:b, Categorical([Pb_1,Pb_2])))
    Pc_11 = 1.0; Pc_12 = 0.9; Pc_21 = 0.3; Pc_22 = 0.1; 
    push!(bn, CategoricalCPD(:c, [:a, :b], [2, 2], [Categorical([Pc_11,1-Pc_11]), Categorical([Pc_12,1-Pc_12]), Categorical([Pc_21, 1-Pc_21]), Categorical([Pc_22, 1-Pc_22])]))
    # P(x|parents(x)) = Normal(μ=a×parents(x) + b, σ)
    a1 = 1.0; b1 = 3.0; σ1 = 1.0
    a2 = 1.0; b1 = 3.0; σ1 = 1.0
    push!(bn, ConditionalLinearGaussianCPD(:d, [:c], [:c], [2], [LinearGaussianCPD(:d, [:c], [1.0], 3.0, 1.0), LinearGaussianCPD(:d, [:c], [1.0], 3.0, 1.0)]))
    push!(bn, CategoricalCPD(:e, [:c], [2], [Categorical([0.98,0.02]), Categorical([0.05, 0.95])]))

    # sample data from BN
    X = Matrix(rand(bn, Ntrain+Ntest))
    x_train = X[1:Ntrain,:]
    x_test = X[(Ntrain+1):(Ntrain+Ntest),:]

    ## learn the CCs
    # learn CC-E
    cc_e = generate_cc(x_train, :learnecf, minclustersize=mink)
    updatescope!(cc_e)

    # learn CC-N
    cc_n = generate_cc(x_train, :learncc, minclustersize=mink)
    updatescope!(cc_n)

    # learn CC-P
    cc_p = generate_cc(x_train, :learncc, distributions = [Categorical, Categorical, Categorical, Categorical, Normal], minclustersize=mink)
    updatescope!(cc_p)

    ## create CC for groundtruth, c.f. Darwiche
    rootnode = FiniteSumNode();
    # Add 4 product nodes to the root.
    add!(rootnode, FiniteProductNode(), log(0.3*0.8)); 
    add!(rootnode, FiniteProductNode(), log(0.3*0.2)); 
    add!(rootnode, FiniteProductNode(), log(0.7*0.8)); 
    add!(rootnode, FiniteProductNode(), log(0.7*0.2)); 
    # add leaf and sum nodes as child of 1. Prod node
    prod_1 = CharacteristicCircuits.children(rootnode)[1];
    add!(prod_1, UnivariateNode(Categorical([1,0]), 1));
    add!(prod_1, UnivariateNode(Categorical([1,0]), 2));
    add!(prod_1, FiniteSumNode());
    sumnode_1 = CharacteristicCircuits.children(prod_1)[3];
    add!(sumnode_1, FiniteProductNode(), log(1.0));
    add!(sumnode_1, FiniteProductNode(), log(0.0));
    prod_1_1 = CharacteristicCircuits.children(sumnode_1)[1];
    add!(prod_1_1, UnivariateNode(Categorical([1,0]), 3));
    add!(prod_1_1, UnivariateNode(Categorical([0.98,0.02]), 4));
    add!(prod_1_1, UnivariateNode(Normal(4,1), 5));
    prod_1_2 = CharacteristicCircuits.children(sumnode_1)[2];
    add!(prod_1_2, UnivariateNode(Categorical([0,1]), 3));
    add!(prod_1_2, UnivariateNode(Categorical([0.05, 0.95]), 4));
    add!(prod_1_2, UnivariateNode(Normal(5,1), 5));
    # add leaf and sum nodes as child of 2. Prod node
    prod_2 = CharacteristicCircuits.children(rootnode)[2];
    add!(prod_2, UnivariateNode(Categorical([1,0]), 1));
    add!(prod_2, UnivariateNode(Categorical([0,1]), 2));
    add!(prod_2, FiniteSumNode());
    sumnode_2 = CharacteristicCircuits.children(prod_2)[3];
    add!(sumnode_2, FiniteProductNode(), log(0.9));
    add!(sumnode_2, FiniteProductNode(), log(0.1));
    prod_2_1 = CharacteristicCircuits.children(sumnode_2)[1];
    add!(prod_2_1, UnivariateNode(Categorical([1,0]), 3));
    add!(prod_2_1, UnivariateNode(Categorical([0.98,0.02]), 4));
    add!(prod_2_1, UnivariateNode(Normal(4,1), 5));
    prod_2_2 = CharacteristicCircuits.children(sumnode_2)[2];
    add!(prod_2_2, UnivariateNode(Categorical([0,1]), 3));
    add!(prod_2_2, UnivariateNode(Categorical([0.05, 0.95]), 4));
    add!(prod_2_2, UnivariateNode(Normal(5,1), 5));
    # add leaf and sum nodes as child of 3. Prod node
    prod_3 = CharacteristicCircuits.children(rootnode)[3];
    add!(prod_3, UnivariateNode(Categorical([0,1]), 1));
    add!(prod_3, UnivariateNode(Categorical([1,0]), 2));
    add!(prod_3, FiniteSumNode());
    sumnode_3 = CharacteristicCircuits.children(prod_3)[3];
    add!(sumnode_3, FiniteProductNode(), log(0.3));
    add!(sumnode_3, FiniteProductNode(), log(0.7));
    prod_3_1 = CharacteristicCircuits.children(sumnode_3)[1];
    add!(prod_3_1, UnivariateNode(Categorical([1,0]), 3));
    add!(prod_3_1, UnivariateNode(Categorical([0.98,0.02]), 4));
    add!(prod_3_1, UnivariateNode(Normal(4,1), 5));
    prod_3_2 = CharacteristicCircuits.children(sumnode_3)[2];
    add!(prod_3_2, UnivariateNode(Categorical([0,1]), 3));
    add!(prod_3_2, UnivariateNode(Categorical([0.05, 0.95]), 4));
    add!(prod_3_2, UnivariateNode(Normal(5,1), 5));
    # add leaf and sum nodes as child of 4. Prod node
    prod_4 = CharacteristicCircuits.children(rootnode)[4];
    add!(prod_4, UnivariateNode(Categorical([0,1]), 1));
    add!(prod_4, UnivariateNode(Categorical([0,1]), 2));
    add!(prod_4, FiniteSumNode());
    sumnode_4 = CharacteristicCircuits.children(prod_4)[3];
    add!(sumnode_4, FiniteProductNode(), log(0.1));
    add!(sumnode_4, FiniteProductNode(), log(0.9));
    prod_4_1 = CharacteristicCircuits.children(sumnode_4)[1];
    add!(prod_4_1, UnivariateNode(Categorical([1,0]), 3));
    add!(prod_4_1, UnivariateNode(Categorical([0.98,0.02]), 4));
    add!(prod_4_1, UnivariateNode(Normal(4,1), 5));
    prod_4_2 = CharacteristicCircuits.children(sumnode_4)[2];
    add!(prod_4_2, UnivariateNode(Categorical([0,1]), 3));
    add!(prod_4_2, UnivariateNode(Categorical([0.05, 0.95]), 4));
    add!(prod_4_2, UnivariateNode(Normal(5,1), 5));

    cc_gt = CharacteristicCircuit(rootnode);
    updatescope!(cc_gt)

    return (gt = cc_gt, cc_e = cc_e, cc_n = cc_n, cc_p = cc_p, train = x_train, test = x_test)
end