using RCall
using Distributions

function test_RCall()
    kek = MvTDist(1, [1.0,2],[2.0 1; 1 3])
    p = rand(kek,100)'
    @rput p
    @rlibrary fitHeavyTail
    R"scale_matrix = fit_Cauchy(p)$scatter"
    @rget scale_matrix
    return scale_matrix
end
