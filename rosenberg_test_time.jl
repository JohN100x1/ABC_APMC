using Distributed #- can parallelise code here by adding extra cores...
using Plots
using StatsBase, Random
using Distributions
using CovarianceEstimation
using LinearAlgebra
using DelimitedFiles

include("./types.jl")
include("./abc2.jl")

function rosenbrock(x)
    n = length(x)
    return sum(100*(view(x, 2:n) .- view(x, 1:(n-1)).^2).^2 .+ (view(x, 1:(n-1)) .- 1).^2)
end

function time_apmc(N_th,k,t_len,n,ker,seed,folder)
    # Set random seed
    Random.seed!(seed)
    # Generate Normal Theta prior
    prior_d() = Normal(0,1)
    th_true = rand(prior_d(), k, 1)
    # Compute rosenberg
    y = rosenbrock(th_true)
    # Distance function
    function rho_lens(d2)
        return rosenbrock(d2)
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
function write_results(mean_time,mean_err,mean_err_len,tau,sigma,k,ker)
    writedlm("data\\"*string(folder)*"\\"*string(ker)*"_mean_time_k"*string(k)*".txt",mean_time)
    writedlm("data\\"*string(folder)*"\\"*string(ker)*"_mean_err_k"*string(k)*".txt",mean_err)
    writedlm("data\\"*string(folder)*"\\"*string(ker)*"_mean_errlen_k"*string(k)*".txt",mean_err_len)
end


# Set random seed
seed = 100
# Number of particles
N_th = 1000
# Dimension
k = 30
# APMC timing parameters
t_len = 10
n = 10 .^ (-1:0.2:1)
ker = "Cauchy"
folder = "Rosenbrock"
time_apmc(N_th,k,t_len,n,ker,seed,folder)

# Increase dimensions k=10, 30
# Effect of no of particles


#-----------------------------------------------------------------

cutoff = 11

# Multi plot

y0 = readdlm("data\\"*string(folder)*"\\Normal_mean_time_k10.txt",'\t',Float64,'\n')
y1 = readdlm("data\\"*string(folder)*"\\Normal_mean_time_k30.txt",'\t',Float64,'\n')
y2 = readdlm("data\\"*string(folder)*"\\Cauchy_mean_time_k10.txt",'\t',Float64,'\n')
y3 = readdlm("data\\"*string(folder)*"\\Cauchy_mean_time_k30.txt",'\t',Float64,'\n')

plot( n[1:cutoff],y0[1:cutoff],xaxis=:log,yaxis=:log,title="Rosenbrock",label="k=10, ker=Normal",legend=:bottomleft)
plot!(n[1:cutoff],y1[1:cutoff],label="k=30 ker=Normal")
plot!(n[1:cutoff],y2[1:cutoff],label="k=10 ker=Cauchy",style=:dash)
plot!(n[1:cutoff],y3[1:cutoff],label="k=30 ker=Cauchy",style=:dash)

xlabel!("n")
ylabel!("Mean time ("*string(t_len)*" iterations per n)")

# Save figure(s)
savefig("C:\\Users\\JohN100x1\\Documents\\_Programming\\Julia\\Rosenbrock_time")
