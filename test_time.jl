using Distributed #- can parallelise code here by adding extra cores...
using Plots
using StatsBase, Random
using Distributions
using CovarianceEstimation
using LinearAlgebra
using DelimitedFiles

include("./types.jl")
include("./abc2.jl")

function time_apmc(N_th,Ny,tau,sigma,k,t_len,n,ker,seed,folder)
    # Set random seed
    Random.seed!(seed)
    # Generate Normal Theta prior
    prior_d() = Normal(0, tau)
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
    mean_err_len = zeros(nt)
    for i in 1:nt
        time = zeros(t_len)
        err = zeros(t_len)
        err_len = zeros(t_len)
        for j in 1:t_len
            time[j] = @elapsed apmc_output = APMC(N_th,[model_lens],[rho_lens],paccmin=0.01,n=n[i],perturb=ker)
            err[j] = apmc_output.epsilon[end]
            err_len[j] = length(apmc_output.epsilon)
        end
        mean_time[i] = mean(time)
        mean_err[i] = mean(err)
        mean_err_len[i] = mean(err_len)
        write_results(mean_time,mean_err,mean_err_len,tau,sigma,k,ker,folder)
        println(n[i]," ",mean_time[i]," ",mean_err[i]," ",mean_err_len[i])
    end
end
function write_results(mean_time,mean_err,mean_err_len,tau,sigma,k,ker,folder)
    writedlm("data\\"*string(folder)*"\\"*string(ker)*"_mean_time_tau"*string(tau)*"_sig"*string(sigma)*"_k"*string(k)*".txt",mean_time)
    writedlm("data\\"*string(folder)*"\\"*string(ker)*"_mean_err_tau"*string(tau)*"_sig"*string(sigma)*"_k"*string(k)*".txt",mean_err)
    writedlm("data\\"*string(folder)*"\\"*string(ker)*"_mean_errlen_tau"*string(tau)*"_sig"*string(sigma)*"_k"*string(k)*".txt",mean_err_len)
end


# Set random seed
seed = 100

### Model
# Number of data points
Ny = 100
# Number of particles
N_th = 1000
# Model parameters
tau = 10; sigma = 0.1; k = 30
xmin = 0; xmax = 10
# APMC timing parameters
t_len = 20
n = 10 .^ (-3:0.2:2.8)
ker = "Cauchy"
folder = "Normal_Linear_Model"
time_apmc(N_th,Ny,tau,sigma,k,t_len,n,ker,seed,folder)



#-----------------------------------------------------------------

cutoff = 20

# Multi plot

y0 = readdlm("data\\"*string(folder)*"\\Normal_mean_errlen_tau10_sig0.1_k5.txt",'\t',Float64,'\n')
y1 = readdlm("data\\"*string(folder)*"\\Normal_mean_errlen_tau10_sig0.1_k15.txt",'\t',Float64,'\n')
y2 = readdlm("data\\"*string(folder)*"\\Normal_mean_errlen_tau10_sig0.1_k30.txt",'\t',Float64,'\n')
y3 = readdlm("data\\"*string(folder)*"\\Cauchy_mean_errlen_tau10_sig0.1_k5.txt",'\t',Float64,'\n')
y4 = readdlm("data\\"*string(folder)*"\\Cauchy_mean_errlen_tau10_sig0.1_k15.txt",'\t',Float64,'\n')
y5 = readdlm("data\\"*string(folder)*"\\Cauchy_mean_errlen_tau10_sig0.1_k30.txt",'\t',Float64,'\n')

plot( n[1:30],y0[1:30],xaxis=:log,yaxis=:log,title="Normal(0, "*string(tau)*")",label="k=5, ker=Normal",legend=:bottomleft)
plot!(n[1:30],y1[1:30],label="k=15 ker=Normal")
plot!(n[1:27],y2[1:27],label="k=30 ker=Normal")
plot!(n[1:30],y3[1:30],label="k=5 ker=Cauchy",style=:dash)
plot!(n[1:30],y4[1:30],label="k=15 ker=Cauchy",style=:dash)
plot!(n[1:30],y5[1:30],label="k=30 ker=Cauchy",style=:dash)


xlabel!("n")
ylabel!("Mean errlen ("*string(t_len)*" iterations per n)")

# Save figure(s)
savefig("C:\\Users\\JohN100x1\\Documents\\_Programming\\Julia\\NLM_errlen")

# test for k=15,  and higher k
# uniform(0.5%, 99.5%) prior
# test rosenbrock (limit 500 iters)
