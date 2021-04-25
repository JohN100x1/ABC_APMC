using Plots
using Distributions
using LinearAlgebra
using Random
# Set random seed
Random.seed!(256)
# Initialise prior and observed data
tau = 1; sigma = 5; k = 2; Nx = 10
# Generate Normal Theta prior
prior_d() = Normal(0,tau)
th_true = rand(prior_d(), k)
# Generate design matrix
Xhat = rand(Normal(-4,4), Nx, k)
Xhat[:,1] .= 1
# Generate Guassian errors
err = rand(Normal(0,sigma), Nx)
# Compute data set
x = Xhat*th_true + err
# ABC Rejection parameters
N = 500; epsilon = 4
# Distance measure
rho(x, y) = sqrt(mean((x - y).^2))
# ABC Rejection
thetas = Array{Float64}(undef,k,N)
for i = 1:N
    theta = rand(prior_d(), k)
    errors = rand(Normal(0,sigma), Nx)
    y = Xhat*theta + errors
    while rho(x, y) > epsilon
        theta = rand(prior_d(), k)
        errors = rand(Normal(0,sigma), Nx)
        y = Xhat*theta + errors
    end
    thetas[:,i] = theta
end
# Plot
H = inv(Xhat'*Xhat + (sigma^2/tau^2)*I)
th_rd = H*Xhat'*x
SIG = sigma^2*H
NLM = MvNormal(th_rd, 0.5*(SIG+SIG'))
maxth1 = maximum(thetas[1,:]); maxth2 = maximum(thetas[2,:])
minth1 = minimum(thetas[1,:]); minth2 = minimum(thetas[2,:])
cx = minth1:(maxth1-minth1)/(50-1):maxth1
cy = minth2:(maxth2-minth2)/(50-1):maxth2
NLMpdf(x,y) = pdf(NLM,[x,y])
cX = repeat(reshape(cx, 1, :), length(cy), 1)
cY = repeat(cy, 1, length(cx))
cZ = map(NLMpdf, cX, cY)
plot_contour = contour(cx, cy, cZ)
plot(plot_contour, legend=:bottomright)
xlabel!("θ1"); ylabel!("θ2"); title!("Generated θ vs Posterior")
scatter!(thetas[1,:],thetas[2,:], label="Accepted Thetas")
