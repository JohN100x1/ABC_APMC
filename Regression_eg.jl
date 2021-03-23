using Distributed #- can parallelise code here by adding extra cores...
using Plots
using StatsPlots
using StatsBase, Random
using Distributions
using CovarianceEstimation
using DelimitedFiles
using LinearAlgebra

include("./types.jl")
include("./abc2.jl")

# Set random seed
seed = 200
Random.seed!(seed[1])

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
function run_apmc(N, np, mu, SIG, k, apmc_repeats, model_lens, rho_lens, ker, df)
    KLD1s = zeros(apmc_repeats, length(N))
    eps1s = zeros(apmc_repeats, length(N))
    for (j,n) in enumerate(N)
        KLD1 = zeros(apmc_repeats)
        eps1 = zeros(apmc_repeats)
        for l in 1:apmc_repeats
            # APMC Output
            apmc_out1 = APMC(np,[model_lens,model_lens],[rho_lens,rho_lens],paccmin=0.001,n=n,perturb=ker,covar="LinearShrinkage(DiagonalUnequalVariance(), :lw)", df=df)
            eps1[l] = apmc_out1.epsilon[end]
            # AMPC Output using weights
            T1 = size(apmc_out1.pts)[2]
            wts1 = apmc_out1.wts[1,T1]
            Nth1 = length(wts1)
            ths1 = apmc_out1.pts[1,T1][:,wsample(1:Nth1, weights(wts1), Nth1)]
            ### Calculate Kullback-leibler Divergence
            mu1 = mean(ths1, dims=2)
            SIG1 = cov(SimpleCovariance(corrected=true), ths1')
            ### KLD for weighted APMC vs True Posterior
            iSIG1 = inv(SIG1)
            KLD1[l] = 0.5*(log(det(SIG1)/det(SIG))-k+tr(iSIG1*SIG)+dot((mu1-mu),iSIG1*(mu1-mu)))
        end
        KLD1s[:,j] = KLD1
        eps1s[:,j] = eps1
        println("n=",string(n), ": avg. KLD1=",string(mean(KLD1)),": avg. eps1=",string(mean(eps1)))
        writedlm("data\\KLD_k"*string(k)*"_VARY_n\\KLD1_np"*string(np)*"_"*ker*"_df"*string(df)*".txt", KLD1s)
        writedlm("data\\KLD_k"*string(k)*"_VARY_n\\eps1_np"*string(np)*"_"*ker*"_df"*string(df)*".txt", eps1s)
    end
end
# Number of data points
Ny = 1000
# Model parameters
tau = 10; sigma = 0.1
xlim = [0, 10]
k = 5
# APMC parameters
N = [0.1, 0.5, 1, 2, 5, 10, 15]
np = 3000
ker = "Cauchy"
df = 10
### Start KLD calculation
apmc_repeats = 1

# re-run t-dist
# vary n

# Data, Prior, and Rho
y, prior_d, rho_lens, th_true, th_ls, th_rd, SIG, p = get_NLM(Ny, tau, sigma, k, xlim)
mu = th_rd
model_lens = repeat([prior_d()];outer=k)
run_apmc(N, np, mu, SIG, k, apmc_repeats, model_lens, rho_lens, ker, df)

#------------------------------

folder = "KLD_k5_VARY_n"
#################### Box plots here
y0 = readdlm("data\\"*string(folder)*"\\eps1_np3000_Normal_df10.txt",'\t',Float64,'\n')
y1 = readdlm("data\\"*string(folder)*"\\eps1_np3000_Cauchy_df1.txt",'\t',Float64,'\n')
y2 = readdlm("data\\"*string(folder)*"\\eps1_np3000_Cauchy_df5.txt",'\t',Float64,'\n')
y3 = readdlm("data\\"*string(folder)*"\\eps1_np3000_Cauchy_df10.txt",'\t',Float64,'\n')

plot(N[1:5],mean(y0,dims=1)'[1:5],xlabel = "n",ylabel = "err", fillalpha=0.75, linewidth=2, title="err, np="*string(np)*", k="*string(k),label="Normal",legend=:bottomright)
plot!(N[1:6],mean(y1,dims=1)'[1:6],label="Cauchy")
plot!(N[1:6],mean(y2,dims=1)'[1:6],label="T-dist df=5")
plot!(N[1:6],mean(y3,dims=1)'[1:6],label="T-dist df=10")
# check error with KLD
# compare with cauchy kernel
# box plots (i.e. record all repeats)

# scale matrix
# vary t-distribution degree

############## create truncated cauchy distribution wrapper (and sampler)
############## fix issue and investigate lower n

# Save figure(s)
savefig("C:\\Users\\JohN100x1\\Documents\\_Programming\\Julia\\images\\plots_apmc\\plot_compare_err_np3000_k5_upton10")




# #-------------------------------------------
# ### Multivariate Normal Density 2nd case
# function p2(x,y)
#     return pdf(MvNormal(vec(th_rd),SIG),[x,y])
# end
# ## Plotting (First two dimensions of theta)
# println("Plotting...")
# cNx = 50; cNy = 50
# th_x_min = min(minimum(ths1[1,:]),minimum(ths2[1,:]),th_true[1,1],th_ls[1,1],th_rd[1,1])
# th_x_max = max(maximum(ths1[1,:]),maximum(ths2[1,:]),th_true[1,1],th_ls[1,1],th_rd[1,1])
# th_y_min = min(minimum(ths1[2,:]),minimum(ths2[2,:]),th_true[2,1],th_ls[2,1],th_rd[2,1])
# th_y_max = max(maximum(ths1[2,:]),maximum(ths2[2,:]),th_true[2,1],th_ls[2,1],th_rd[2,1])
# cx = th_x_min:(th_x_max-th_x_min)/(cNx-1):th_x_max
# cy = th_y_min:(th_y_max-th_y_min)/(cNy-1):th_y_max
# cX = repeat(reshape(cx, 1, :), length(cy), 1)
# cY = repeat(cy, 1, length(cx))
# cZ = map(p2, cX, cY)
# println("Plotting Contour...")
# plot_contour = contour(cx, cy, cZ)
# marginalkde(ths1[1,:],ths1[2,:])
# plot!(plot_contour)
# #plot!(ths1[1,:], ths1[2,:], linetype=:scatter,label="n=1")
# #plot!(ths2[1,:], ths2[2,:], linetype=:scatter,label="n=2")
