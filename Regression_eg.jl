using Distributed #- can parallelise code here by adding extra cores...
using Plots
using StatsBase, Random
using Distributions
using CovarianceEstimation
using DelimitedFiles
using LinearAlgebra

include("./types.jl")
include("./abc2.jl")

### Model
# Number of data points
Ny = 100
# Model parameters
tau = 20; sigma = 0.1; k = 2
xmin = 0; xmax = 10
# Generate Normal Theta prior
prior_d() = Normal(0,tau)
th_true = rand(prior_d(), k, 1)
# Generate design matrix
X = rand(Uniform(xmin,xmax), Ny,k)
X[:,1] .= 1
# Generate Guassian errors
err = rand(Normal(0,sigma), Ny)
# Compute data set
y = X*th_true + err

# Set random seed
seed = abs.(rand(Int, 2))
Random.seed!(seed[1])


# Distance function
function rho_lens(d2)
    pars = copy(d2)
    errors = rand(Normal(0,sigma), Ny)
    yp = X * pars[1:k] + errors
    rmse = sqrt(mean(abs2.(y .- yp)))
    return rmse
end

model_lens = repeat([prior_d()];outer=k)
apmc_output = APMC(1000,[model_lens],[rho_lens],paccmin=0.001,n=2,perturb="Cauchy")

T = size(apmc_output.pts)[2]
wts = apmc_output.wts[1,T]
ths = apmc_output.pts[1,T]
println("t=",T)
#println(wts)


### Least squares estimator
println("Computing Least Squares Estimator...")
th_ls = (X'*X)\(X'*y)

### Ridge regression estimator
println("Computing Ridge Regression Estimator...")
XtX_sigtau = X'*X + (sigma^2/tau^2)*I
XtX_sigtau_inv = inv(XtX_sigtau)
th_ridge = XtX_sigtau_inv*X'*y
bigsigma = sigma^2*XtX_sigtau_inv


# Define Multi-variate Normal density
function MVN_density(x,y)
    return pdf(MvNormal([th_ridge[1],th_ridge[2]],bigsigma),[x,y])
end

### Plotting (First two dimensions of theta)
println("Plotting...")
cNx = 50; cNy = 50
th_x_min = min(minimum(ths[1,:]),th_true[1,1],th_ls[1,1],th_ridge[1,1])
th_x_max = max(maximum(ths[1,:]),th_true[1,1],th_ls[1,1],th_ridge[1,1])
th_y_min = min(minimum(ths[2,:]),th_true[2,1],th_ls[2,1],th_ridge[2,1])
th_y_max = max(maximum(ths[2,:]),th_true[2,1],th_ls[2,1],th_ridge[2,1])
cx = th_x_min:(th_x_max-th_x_min)/(cNx-1):th_x_max
cy = th_y_min:(th_y_max-th_y_min)/(cNy-1):th_y_max
cX = repeat(reshape(cx, 1, :), length(cy), 1)
cY = repeat(cy, 1, length(cx))
cZ = map(MVN_density, cX, cY)
println("Plotting Contour...")

plot_contour = contour(cx, cy, cZ)
plot(plot_contour)

sampled_th = zeros(2,500)
for i in 1:500
    sampled_th[:,i] = ths[:,sample(1:500,weights(wts))]
end
plot!(sampled_th[1,:],sampled_th[2,:],linetype=:scatter)


# Save figure(s)
savefig("C:\\Users\\JohN100x1\\Documents\\_Programming\\Julia\\images\\plots_apmc\\seed="*string(seed[1]))

#--------------------------------------------
















plot!(ths[1,:],ths[2,:],linetype=:scatter)
plot!([th_true[1,1]],[th_true[2,1]],linetype=:scatter)
plot!([th_ls[1,1]],[th_ls[2,1]],linetype=:scatter)
plot!([th_ridge[1,1]],[th_ridge[2,1]],linetype=:scatter)

println(mean(ths[1,:],weights(wts)))
println(mean(ths[2,:],weights(wts)))

println(th_ridge[1,1])
println(th_ridge[2,1])
