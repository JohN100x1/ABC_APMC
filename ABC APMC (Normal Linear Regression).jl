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

### Model
println("Initialising Normal Linear Model...")
# Number of data points
Ny = 20
# Model parameters
tau = 1; sigma = 1; k = 2
xmin = 0; xmax = 10
c1 = (2*pi)^(0.5 - 0.5*k)/tau^k
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

### APMC parameters
println("Initialising APMC parameters...")
# Number of particles
N = 1000
# Quantile to keep per iteration
alpha = 0.5
Nalpha = floor(Int,N*alpha)
# Max iterations
T = 100
# Minimum acceptance rate
paccmin = 0.02
# Define distance measure
d(y,z) = sum((y - z).^2)

### Initialise ths, wts, rho
ths = zeros(N, T, k)
wts = zeros(N, T)
rhos = zeros(N, T)
eps = zeros(T)
sig = zeros(T, k, k)

### Initial step (t = 1)
println("Beginning APMC loop...")
for i in 1:N
    # Simulate theta from prior
    th_til = prior_sample(1)
    ths[i,1,:] = th_til
    # Generate fake data
    errors = rand(Normal(0,sigma), Ny)
    z = X*th_til + errors
    rhos[i,1] = d(y,z)
    wts[i,1] = 1
end
eps[1] = quantile(rhos[:,1],alpha)
acc = rhos[:,1] .<= eps[1]

ths_acc = ths[acc,1,:]
wts_acc = wts[acc,1]
rhos_acc = rhos[acc,1]
# Cov. matrix function
function covmat(ths_acc, wts_acc, Nalpha)
    # Cov. matrix
    part1 = zeros(k, k)
    part2 = zeros(k, k)
    # Calculating first part of cov. matrix
    for i in 1:Nalpha
        part1 +=  wts_acc[i] * ths_acc[i,:] * ths_acc[i,:]'
    end
    part1 *= 2/sum(wts_acc)
    # Calculating second part of cov. matrix
    for i in 1:Nalpha, j in 1:Nalpha
        part2 += wts_acc[i] * wts_acc[j] * ths_acc[i,:] * ths_acc[j,:]'
    end
    part2 *= -2/sum(wts_acc)^2
    cov = part1 + part2
    #######################
    # Symmetric correction
    cov[2,1] = cov[1,2]
    #######################
    return cov
end
# Cov. matrix
sig[1,:,:] = 2*covmat(ths_acc,wts_acc,Nalpha)
pacc = 1
t = 2
while pacc > paccmin && t <= T
    sum_wts = sum(wts_acc)
    writedlm(stdout, sig[t-1,:,:])
    for i in findall(.!acc)
        # Sample from previous theta with weights
        th_i = sample(1:Nalpha, Weights(wts_acc))
        th_star = ths_acc[th_i,:]
        # Generate from Kernel
        th_til = rand(MvNormal(th_star,sig[t-1,:,:]), 1)
        ths[i,t-1,:] = th_til
        # Generate fake data
        errors = rand(Normal(0,sigma), Ny)
        z = X*th_til + errors
        # Set distance
        rhos[i,t-1] = d(y,z)
        # Set weight
        numer = prior_density(th_til)*sum_wts
        denom = sum(wts_acc .* pdf(MvNormal(sig[t-1,:,:]),repeat(th_til,1,Nalpha)-ths[acc,t-1,:]'))
        wts[i,t-1] = numer/denom
    end
    global pacc = sum(rhos[.!acc,t-1] .< eps[t-1])/(N-Nalpha)
    eps[t] = quantile(rhos[:,t-1],alpha)
    global acc = rhos[:,t-1] .<= eps[t]
    # Accepted ths, wts, rhos
    global ths_acc = ths[acc,t,:]
    global wts_acc = wts[acc,t]
    global rhos_acc = rhos[acc,t]
    # Cov. matrix
    sig[t,:,:] = 2*covmat(ths_acc,wts_acc,Nalpha)
    global t += 1
end



T = t-2
println("t=",T)
println(wts[wts[:,T] .>= 0.1,T])
writedlm(stdout, ths[wts[:,T] .>= 0.1,T,:])




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
plot!(ths[:,T,1],ths[wts[:,T] .>= 0.1,T,2],linetype=:scatter)
plot!([th_true[1,1]],[th_true[2,1]],linetype=:scatter)
plot!([th_ls[1,1]],[th_ls[2,1]],linetype=:scatter)
plot!([th_ridge[1,1]],[th_ridge[2,1]],linetype=:scatter)


println(mean(ths[:,T,1],weights(wts[:,T])))
println(mean(ths[:,T,2],weights(wts[:,T])))

println(th_ridge[1,1])
println(th_ridge[2,1])

### Keep the Plots alive
gui()
readline(stdin)






































#
