using Distributed #- can parallelise code here by adding extra cores...
using Plots
using StatsBase, Random
using Distributions
using CovarianceEstimation

include("./types.jl")
include("./abc2.jl")

function rosenbrock(x)
    n = length(x)
    return sum(100*(view(x, 2:n) .- view(x, 1:(n-1)).^2).^2 .+ (view(x, 1:(n-1)) .- 1).^2)
end
# Set random seed
Random.seed!(100)

### Model
# Number of data points
Ny = 100
# Dimension
k = 3
# Generate Normal Theta prior
prior_d() = Normal(0,1)
th_true = rand(prior_d(), k, 1)
# Compute data set
y = rosenbrock(th_true)
# Distance function
function rho_lens(d2)
    return rosenbrock(d2)
end

model_lens = repeat([prior_d()];outer=k)
apmc_output = APMC(1000,[model_lens],[rho_lens],paccmin=0.01,n=2)

T = size(apmc_output.pts)[2]
wts = apmc_output.wts[1,T]
ths = apmc_output.pts[1,T]
println("t=",T)
println(wts)

x = (minimum(ths[1,:])-0.2):0.1:(maximum(ths[1,:])+0.2)
y = x .^ 2
plot(x,y,label="y=x^2",legend=:bottomright)
plot!(ths[1,:],ths[2,:],label="Simulated theta",linetype=:scatter)

println(mean(ths[1,:],weights(wts)))
println(mean(ths[2,:],weights(wts)))

using Plots;
x=range(-2,stop=2,length=100)
y=range(sqrt(2),stop=2,length=100)
f(x,y) = x*y-x-y+1
plot(ths[1,:],ths[2,:],ths[3,:],st=:scatter3d,camera=(-0,0))
