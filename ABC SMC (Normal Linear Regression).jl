### Loading Modules
println("Loading modules...")
using Plots
using Random
using StatsBase
using Distributions
using DelimitedFiles
using LinearAlgebra
# Set random seed
Random.seed!(285)

### Normal linear model
println("Initialising Normal Linear Model...")
# Number of data points
Ny = 20
# Model parameters
tau = 1; sigma = 1; k = 2
xmin = 0; xmax = 10
c1 = (2*pi)^(0.5 - 0.5*k)/tau^k
c2 = (2*pi)^(0.5 - 0.5*k)
# Generate Normal Theta prior
prior_sample(n) = rand(Normal(0,tau), k, n)
prior_density(th) = c1*pdf(Normal(0,1),norm(th./tau))
th_true = prior_sample(1)
# Generate design matrix
X = rand(Uniform(xmin,xmax), Ny,k)
X[:,1] .= 1
# Generate Guassian errors
err = rand(Normal(0,sigma), Ny)
# Compute data set
y = X*th_true + err

### ABC SMC
println("Initialising ABC SMC parameters...")
# Define distance measure
d(y,z) = (mean(y) - mean(z))^2 + (std(y) - std(z))^2
# Define unit vector function
normalise(v) = v/norm(v)
# Define kernel function (Component-wise Normal)
K_sample(mu,sig) = rand.(Normal.(mu, sig))
K_density(th,mu,sig) = c2*pdf(Normal(0,1),norm((th-mu)./sig))/prod(sig)
# Epsilon tolerance schedule (Non-adaptive)
T = 5; U = 1; L = 0.1
eps = [U:(L-U)/(T-1):L;]
# Number of Particles
Nth = 50
# Particle index
th_is = [1:1:Nth;]
# Theta, weight, normalised weight, and kernel sigma arrays
ths = zeros(Nth, T, k)
ws = zeros(Nth, T)
ws_til = zeros(Nth, T)
K_sig = zeros(T, k)
# Begin ABC SMC loop
t = 0
# Tolerance layer
println("Beginning ABC SMC loop...")
while t < T
    global t += 1
    i = 1
    if t != 1
        # Calculating K_sig
        prev_th = ths[:,t-1,:]
        prev_w = ws[:,t-1]
        prev_w_til = ws_til[:,t-1]
        sum_w = sum(prev_w)
        sum_w_til = sum(prev_w_til)
        for k_i in 1:k
            a1 = sum_w*sum(prev_w_til.*prev_th[:,k_i].^2)
            a2 = -2*sum(prev_w.*prev_th[:,k_i])*sum(prev_w_til.*prev_th[:,k_i])
            a3 = sum_w_til*sum(prev_w.*prev_th[:,k_i].^2)
            K_sig[t,k_i] = sqrt(a1 + a2 + a3)
        end
        curr_K_sig = K_sig[t,:]
    end
    # Inner loop simulates theta (t=1) and then perturbs them (t>1)
    while i <= Nth
        if t == 1
            # Simulate theta from prior
            th_til = prior_sample(1)
        else
            # Sample from previous theta with weights
            th_i = sample(th_is, Weights(prev_w_til))
            th = prev_th[th_i,:]
            # Sample th_til from Kernel with positive prior density
            th_til = K_sample(th, curr_K_sig)
            while prior_density(th_til) <= 0
                th_til = K_sample(th, curr_K_sig)
            end
        end
        # Generate fake data within tolerance
        errors = rand(Normal(0,sigma), Ny)
        z = X*th_til + errors
        if d(y,z) <= eps[t]
            ths[i,t,:] = th_til
            i += 1
        end
    end
    # Calculate weights for each particle
    for j in th_is
        if t != 1
            curr_th = ths[j,t,:]
            numer = prior_density(curr_th)
            denom = sum(prev_w.*K_density(curr_th,prev_th[j,:],curr_K_sig))
            ws[j,t] = numer/denom
        else
            ws[j,1] = 1
        end
    end
    # Normalising weights
    ws_til[:,t] = normalise(ws[:,t])
end

### Least squares estimator
println("Computing Least Squares Estimator...")
th_ls = (X'*X)\(X'*y)

### Ridge regression estimator
println("Computing Ridge Regression Estimator...")
XtX_sigtau = X'*X + (sigma^2/tau^2)*I
XtX_sigtau_inv = inv(XtX_sigtau)
th_ridge = XtX_sigtau_inv*X'*y
bigsigma = sigma^2*XtX_sigtau_inv
bigsigma_det = det(bigsigma)
bigsigma_inv = XtX_sigtau/sigma^2
c3 = (2*pi)^(-0.5*k)*bigsigma_det^(-0.5)
# Define Multi-variate Normal density
v = zeros(2,1)
function MVN_density(x,y)
    global v[1,1] = x
    global v[2,1] = y
    return c3*exp((-0.5*(v-th_ridge)'*bigsigma_inv*(v-th_ridge))[1])
end

### Plotting (First two dimensions of theta)
println("Plotting...")
cNx = 50; cNy = 50
th_x_min = min(minimum(ths[:,T,1]),th_true[1,1],th_ls[1,1],th_ridge[1,1])
th_x_max = max(maximum(ths[:,T,1]),th_true[1,1],th_ls[1,1],th_ridge[1,1])
th_y_min = min(minimum(ths[:,T,2]),th_true[2,1],th_ls[2,1],th_ridge[2,1])
th_y_max = max(maximum(ths[:,T,2]),th_true[2,1],th_ls[2,1],th_ridge[2,1])
cx = th_x_min:(th_x_max-th_x_min)/(cNx-1):th_x_max
cy = th_y_min:(th_y_max-th_y_min)/(cNy-1):th_y_max
cX = repeat(reshape(cx, 1, :), length(cy), 1)
cY = repeat(cy, 1, length(cx))
cZ = map(MVN_density, cX, cY)
println("Plotting Contour...")
plot_contour = contour(cx, cy, cZ)
plot(plot_contour)
plot!(ths[:,T,1],ths[:,T,2],linetype=:scatter)
plot!([th_true[1,1]],[th_true[2,1]],linetype=:scatter)
plot!([th_ls[1,1]],[th_ls[2,1]],linetype=:scatter)
plot!([th_ridge[1,1]],[th_ridge[2,1]],linetype=:scatter)

### Keep the Plots alive
gui()
readline(stdin)
