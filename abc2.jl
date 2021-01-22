using Distributions
using LinearAlgebra
using CovarianceEstimation

# Some sort of initialise function
function init(models, np, rho)
    d = Inf
    count = 1
    # sample a number from 1 to Number of models
    m = sample(range(1, stop=length(models)))
    # Take a random sample from model m
    pars = rand.(models[m])
    # Calculate the distance of model m with that random sample
    d = rho[m](pars)
    # Return a column vector whose elements are m, pars, max(np)-np[m] zeros, d and 1
    return vcat(m, pars, fill(0, maximum(np) - np[m]), d, count)
end

# SMC sampler (for subsequent iterations)
function cont(models, pts, wts, np, i, ker, rho)
    d = Inf
    count = 1
    # sample a number from 1 to Number of models
    m = sample(1:length(models))
    # While the size (row length) of the m-th particle at iteration i-1 is zero, keep sampling m
    while size(pts[m,i - 1])[2] == 0
        m = sample(1:length(models))
    end
    # Sample a particle
    params = pts[m,i - 1][:,sample(1:size(pts[m,i - 1])[2], wts[m,i - 1])]
    # Perturb the particle with kernel m
    params = params + rand(ker[m])
    ### Keep repeating the sampling and perturbation step if the pdf of the model is zero.
    while prod(pdf.(models[m], params)) == 0
        m = sample(1:length(models))
        while size(pts[m,i - 1])[2] == 0
            m = sample(1:length(models))
        end
        count = count + 1
        params = pts[m,i - 1][:,sample(1:size(pts[m,i - 1])[2], wts[m,i - 1])]
        params = params + rand(ker[m])
    end
    ###
    # Calculate the distance
    d = rho[m](params)
    # Return a column vector whose elements are m, pars, max(np)-np[m] zeros, d and 1
    return vcat(m, params, fill(0, maximum(np) - np[m]), d, count)
end


function APMC(N, models, rho,;names=Vector[[string("parameter", i) for i in 1:length(models[m])] for m in 1:length(models)],prop=0.5,paccmin=0.02,n=2,covar="LinearShrinkage(DiagonalUnequalVariance(), :lw)",perturb="Normal")
    i = 1
    lm = length(models)
    # N_alpha
    s = round(Int, N * prop)
  # array for number of parameters in each model
    np = Array{Int64}(undef, length(models))
    for j in 1:lm
        np[j] = length(models[j])
    end
  # array for SMC kernel used in weights
    ker = Array{Any}(undef, lm)
    template = Array{Any}(undef, lm, 1)
  # particles array
    pts = similar(template)
  # covariance matrix array
    sig = similar(template)
  # weights array
    wts = similar(template)
  # model probability at each iteration array
    p = zeros(lm, 1)
    temp = @distributed hcat for j in 1:N
        init(models, np, rho)
    end
    its = [sum(temp[size(temp)[1],:])]
    # Calculate epsilon using alpha quantile
    epsilon = [quantile(collect(temp[maximum(np) + 2,:]), prop)]
    # pacc is 1 for each model
    pacc = ones(lm, 1)
    #println(round.([epsilon[i];its[i]], digits=3))
    # temp is reduced to the ones which are less than or equal to epsilon
    temp = temp[:,temp[maximum(np) + 2,:] .<= epsilon[i]]
    # accept the first N_alpha
    temp = temp[:,1:s]
    # Loop through per model and reset? the particles and weights
    for j in 1:lm
        pts[j,i] = temp[2:(np[j] + 1),temp[1,:] .== j]
        wts[j,i] = StatsBase.weights(fill(1.0, sum(temp[1,:] .== j)))
    end
    dists = transpose(temp[(maximum(np) + 2),:])
    for j in 1:lm
        p[j] = sum(wts[j,1])
    end
    # Calculate the covariance for the first iteration
    for j in 1:lm
        params = zeros(N, np[j])
        for num = 1:N
            params[num,:] = pts[j,i][:,sample(1:size(pts[j,i])[2], wts[j,i])]
        end
        sig[j,i] = CovarianceEstimation.cov(eval(Meta.parse("$covar")), params)
    end
    # making sure the elements of p sum to 1
    p = p ./ sum(p)
    nbs = Array{Integer}(undef, length(models))
    for j in 1:lm
        nbs[j] = length(wts[j,i])
        #println(round.(hcat(mean(diag(sig[j,i])[1:(np[j])]), pacc[j,i], nbs[j], p[j,i]), digits=3))
    end
    # The main while loop for the second iteration and beyond
    while maximum(pacc[:,i]) > paccmin && i < 500
        pts = reshape(pts, i * length(models))
        sig = reshape(sig, i * length(models))
        wts = reshape(wts, i * length(models))
        # Append a size 1 array with undef element
        for j in 1:length(models)
            push!(pts, Array{Any}(undef, 1))
            push!(sig, Array{Any}(undef, 1))
            push!(wts, Array{Any}(undef, 1))
        end
        pts = reshape(pts, length(models), i + 1)
        sig = reshape(sig, length(models), i + 1)
        wts = reshape(wts, length(models), i + 1)
        # Increase the iteration count
        i = i + 1
        # Calculate perturbation kernel for each model
        for j in 1:lm
            if perturb == "Cauchy"
                ker[j] = MvTDist(1, fill(0.0, np[j]), float.(n * sig[j,i - 1]))
            elseif perturb == "Normal"
                ker[j] = MvNormal(fill(0.0, np[j]), n * sig[j,i - 1])
            end
        end
        # SMC Sampler step
        temp2 = @distributed hcat for j in (1:(N - s))
            cont(models, pts, wts, np, i, ker, rho)
        end
        # vcat and hcat used to concatenate elments/vectors
        its = vcat(its, sum(temp2[size(temp2)[1],:]))
        temp = hcat(temp, temp2)
        inds = sortperm(reshape(temp[maximum(np) + 2,:], N))[1:s]
        temp = temp[:,inds]
        dists = hcat(dists, transpose(temp[(maximum(np) + 2),:]))
        epsilon = vcat(epsilon, temp[(maximum(np) + 2),s])
        pacc = hcat(pacc, zeros(lm))
        for j in 1:lm
            if sum(temp2[1,:] .== j) > 0
                pacc[j,i] = sum(temp[1,inds .> s] .== j) / sum(temp2[1,:] .== j)
            else pacc[j,i] == 0
            end
        end
        # println(round.(vcat(log10(epsilon[i]), its[i]), digits=3))
        for j in 1:lm
            pts[j,i] = temp[2:(np[j] + 1),temp[1,:] .== j]
            if size(pts[j,i])[2] > 0
                # weight calculation
                keep = inds[reshape(temp[1,:] .== j, s)] .<= s
                wts[j,i] = @distributed vcat for k in range(1, stop=length(keep))
                    if !keep[k]
                        prod(pdf.(models[j], (pts[j,i][:,k]))) / (1 / (sum(wts[j,i - 1])) * dot(convert(Vector, wts[j,i - 1]), pdf(ker[j], broadcast(-, pts[j,i - 1], pts[j,i][:,k]))))
                    else
                        0.0
                    end
                end
                if length(wts[j,i]) == 1
                    wts[j,i] = fill(wts[j,i], 1)
                end
                l = 1
                for k in 1:length(keep)
                    if keep[k]
                        wts[j,i][k] = wts[j,i - 1][l]
                        l = l + 1
                    end
                end
                if length(wts[j,i]) > 1
                    wts[j,i] = StatsBase.weights(wts[j,i])
                end
            else
                wts[j,i] = zeros(0)
            end
        end
        p = hcat(p, zeros(length(models)))
        for j in 1:lm
            wts[j,i] = weights(wts[j,i])
            p[j,i] = sum(wts[j,i])
        end
        # Calculate covariance (to use in kernel)
        for j in 1:lm
            if (size(pts[j,i])[2] > np[j])
                params = zeros(N, np[j])
                for num = 1:N
                    params[num,:] = pts[j,i][:,sample(1:size(pts[j,i])[2], wts[j,i])]
                end
                sig[j,i] = CovarianceEstimation.cov(eval(Meta.parse("$covar")), params)
                if isposdef(sig[j,i])
                    if perturb == "Cauchy"
                        dker = MvTDist(1, pts[j,i - 1][:,1], float.(n * sig[j,i]))
                    elseif perturb == "Normal"
                        dker = MvNormal(pts[j,i - 1][:,1], n * sig[j,i - 1])
                    end
                    if pdf(dker, pts[j,i][:,1]) == Inf
                        sig[j,i] = sig[j,i - 1]
                    end
                else
                    sig[j,i] = sig[j,i - 1]
                end
            else
                sig[j,i] = sig[j,i - 1]
            end
        end
        # making sure p are probabilities
        p[:,i] = p[:,i] ./ sum(p[:,i])
        for j in 1:lm
            nbs[j] = length(wts[j,i])
            #println(round.(hcat(mean(diag(sig[j,i]) ./ diag(sig[j,1])), pacc[j,i], nbs[j], p[j,i]), digits=3))
        end
    end
    samp = ABCfit(pts, sig, wts, p, its, dists, epsilon, temp, pacc, names, models)
    return(samp)
end
