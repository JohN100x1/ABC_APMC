using Distributed #- can parallelise code here by adding extra cores...
using Plots
using StatsBase, Random
using Distributions
using CovarianceEstimation

include("./types.jl")
#include("functions.jl")
include("./abc2.jl") #John you may have different file name here
dims = 2 #change dimension of problem here

# Set random seed
Random.seed!(285)

# prepare data
X = rand(1000, dims)   # feature matrix
params = randn(dims)    # ground truths
sigma = 0.2
y = X * params[1:dims] + sigma * randn(1000)  # generate response
function rho_lens(d2)
    pars = copy(d2)
    errors = rand(Normal(0,sigma), 1000)
    yp = X * pars[1:dims] + errors
    rmse = sqrt(mean(abs2.((y .- yp)./sigma)))
    return rmse
end

A = Normal(0,1) #define priors
model_lens = repeat([A];outer=dims)
apmc_output = APMC(1000,[model_lens],[rho_lens],paccmin=0.01,n=0.5)

T = size(apmc_output.pts)[2]
println("t=",T)
println(apmc_output.wts[1,T])


plot(apmc_output.pts[1,T][1,:],apmc_output.pts[1,T][2,:],linetype=:scatter)
plot!([params[1]],[params[2]],linetype=:scatter)

println(mean(apmc_output.pts[1,T][1,:],weights(apmc_output.wts[1,T])))
println(mean(apmc_output.pts[1,T][2,:],weights(apmc_output.wts[1,T])))
