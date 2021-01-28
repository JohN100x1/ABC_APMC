using Distributed #- can parallelise code here by adding extra cores...
using Plots
using StatsBase, Random
using Distributions
using CovarianceEstimation
using DelimitedFiles
using LinearAlgebra

include("./types.jl")
include("./abc2.jl")
include("./abc_unweighted.jl")

# Generate data from Normal Linear Model
function get_NLM(Ny, tau, sigma, k, xlim)
    # Generate Normal Theta prior
    prior_d() = Normal(0,tau)
    th_true = rand(prior_d(), k, 1)
    # Generate design matrix
    X = rand(Uniform(xlim[1],xlim[2]), Ny, k)
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
    ### Least squares estimator
    th_ls = (X'*X)\(X'*y)
    ### Ridge regression estimator
    H = inv(X'*X + (sigma^2/tau^2)*I)
    th_rd = H*X'*y
    SIG = sigma^2*H
    ### Multivariate Normal Density
    function p(th)
        return pdf(MvNormal(vec(th_rd),SIG),th)
    end
    return y, prior_d, rho_lens, th_true, th_ls, th_rd, SIG, p
end
# Number of data points
Ny = 100
# Model parameters
tau = 10; sigma = 0.1
xlim = [0, 10]
K = [2]
# APMC parameters
n = 2
# Set random seed
seed = 100
Random.seed!(seed[1])
### Start KLD calculation
avg_len = 1
avg_KLD1 = zeros(length(K))
avg_KLD2 = zeros(length(K))

### Edited later
ths1 = 0
wts1 = 0
ths2 = 0
SIG = 0
th_rd = 0
th_true = 0
th_ls = 0

for (j,k) in enumerate(K)
    KLD1 = zeros(avg_len)
    KLD2 = zeros(avg_len)
    # Data, Prior, and Rho
    y, prior_d, rho_lens, th_true, th_ls, th_rd, SIG, p = get_NLM(Ny, tau, sigma, k, xlim)
    mu = th_rd
    model_lens = repeat([prior_d()];outer=k)
    for l in 1:avg_len
        # APMC Output
        apmc_out1 = APMC(5000,[model_lens,model_lens],[rho_lens,rho_lens],paccmin=0.001,n=n,perturb="Normal")
        apmc_out2 = APMC_unweighted(5000,[model_lens,model_lens],[rho_lens,rho_lens],paccmin=0.001,n=n,perturb="Normal")
        # AMPC Output using weights
        T1 = size(apmc_out1.pts)[2]
        wts1 = apmc_out1.wts[1,T1]
        Nth1 = length(wts1)
        ths1 = zeros(k,Nth1)
        for i in 1:Nth1
            ths1[:,i] = apmc_out1.pts[1,T1][:,sample(1:Nth1,weights(wts1))]
        end
        # AMPC Output no weights
        T2 = size(apmc_out2.pts)[2]
        ths2 = apmc_out2.pts[1,T2]
        ### Calculate Kullback-leibler Divergence
        mu1 = mean(ths1, dims=2)
        mu2 = mean(ths2, dims=2)
        SIG1 = cov(SimpleCovariance(corrected=true), ths1')
        SIG2 = cov(SimpleCovariance(corrected=true), ths2')
        ### KLD for weighted APMC vs True Posterior
        iSIG1 = inv(SIG1)
        iSIG2 = inv(SIG2)
        KLD1[l] = 0.5*(log(det(SIG1)/det(SIG))-k+tr(iSIG1*SIG)+dot((mu1-mu),iSIG1*(mu1-mu)))
        KLD2[l] = 0.5*(log(det(SIG2)/det(SIG))-k+tr(iSIG2*SIG)+dot((mu2-mu),iSIG2*(mu2-mu)))
    end
    avg_KLD1[j] = mean(KLD1)
    avg_KLD2[j] = mean(KLD2)
    println("k=",string(k), ": KLD1=",string(avg_KLD1[j]), ", KLD2=",string(avg_KLD2[j]))
    writedlm("data\\KLD_VARY_k\\mean_KLD1_n"*string(n)*".txt", avg_KLD1)
    writedlm("data\\KLD_VARY_k\\mean_KLD2_n"*string(n)*".txt", avg_KLD2)
end


### Multivariate Normal Density 2nd case
function p2(x,y)
    return pdf(MvNormal(vec(th_rd),SIG),[x,y])
end





#-------------------------------------------

## Plotting (First two dimensions of theta)
println("Plotting...")
cNx = 50; cNy = 50
th_x_min = min(minimum(ths1[1,:]),minimum(ths2[1,:]),th_true[1,1],th_ls[1,1],th_rd[1,1])
th_x_max = max(maximum(ths1[1,:]),maximum(ths2[1,:]),th_true[1,1],th_ls[1,1],th_rd[1,1])
th_y_min = min(minimum(ths1[2,:]),minimum(ths2[2,:]),th_true[2,1],th_ls[2,1],th_rd[2,1])
th_y_max = max(maximum(ths1[2,:]),maximum(ths2[2,:]),th_true[2,1],th_ls[2,1],th_rd[2,1])
cx = th_x_min:(th_x_max-th_x_min)/(cNx-1):th_x_max
cy = th_y_min:(th_y_max-th_y_min)/(cNy-1):th_y_max
cX = repeat(reshape(cx, 1, :), length(cy), 1)
cY = repeat(cy, 1, length(cx))
cZ = map(p2, cX, cY)
println("Plotting Contour...")
plot_contour = contour(cx, cy, cZ)
plot(plot_contour)
plot!(ths1[1,:], ths1[2,:], linetype=:scatter,label="Weighted")
plot!(ths2[1,:], ths2[2,:], linetype=:scatter,label="Unweighted")

# Save figure(s)
savefig("C:\\Users\\JohN100x1\\Documents\\_Programming\\Julia\\images\\plots_apmc\\seed="*string(seed)*"true")

#--------------------------------------------

# Kullback-leibler implement
# Implement constant weights
# Keep dimensions/n constant
# Sparseness of covariance
