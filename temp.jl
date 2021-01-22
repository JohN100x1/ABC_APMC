using Distributed #- can parallelise code here by adding extra cores...
using Plots
using StatsBase, Random
using Distributions
using CovarianceEstimation
using LinearAlgebra
using DelimitedFiles

include("./types.jl")
include("./abc2.jl")

# Set random seed
seed = 100

### Model
# Number of data points
Ny = 100
# Model parameters
tau = 50; sigma = 0.1; k = 2
xmin = 0; xmax = 10
# APMC timing parameters
t_len = 1
n = 10 .^ (-3)
ker = "Normal"

# Set random seed
Random.seed!(seed)
# Generate Normal Theta prior
prior_d() = Normal(0,tau)
th_true = rand(prior_d(), k, 1)
# Generate design matrix
X = rand(Uniform(xmin,xmax), Ny, k)
X[:,1] .= 1
# Generate Guassian errors
err = rand(Normal(0,sigma), Ny)
# Compute data set
y = X*th_true + err
# Distance function
function rho_lens(d2)
    pars = copy(d2)
    errors = rand(Normal(0,sigma), Ny)
    yp = X * pars[1:k] + errors
    rmse = sqrt(mean(abs2.(y .- yp)))
    return rmse
end
model_lens = repeat([prior_d()];outer=k)
nt = length(n)
mean_time = zeros(nt)
mean_err = zeros(nt)
eps_123 = 0
for i in 1:nt
    time = zeros(t_len)
    err = zeros(t_len)
    for j in 1:t_len
        time[j] = @elapsed apmc_output = APMC(10000,[model_lens],[rho_lens],paccmin=0.01,n=n[i],perturb=ker)
        err[j] = apmc_output.epsilon[end]
        eps_123 = apmc_output.epsilon
    end
    mean_time[i] = mean(time)
    mean_err[i] = mean(err)
    println(i," ",mean_time[i]," ",mean_err[i])
end
