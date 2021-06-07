using Distributed
using Distributions
using StatsBase
using StatsPlots
using Plots
using DelimitedFiles

include("./types.jl")
include("./abc2.jl")

# Set random seed
seed = 201
Random.seed!(seed[1])

function two_moons_model()
    prior_d() = Uniform(-1,1)
    th_true = rand(prior_d(), 2, 1)
    a = rand(Uniform(-pi/2,pi/2), data_size, 1)
    r = rand(Normal(0.1,0.01), data_size, 1)
    c1 = r.*cos.(a).+0.25.-abs(th_true[1]+th_true[2])/sqrt(2)
    c2 = r.*sin.(a).+(-th_true[1]+th_true[2])/sqrt(2)
    target_data = [c1 c2]
    function rho_lens(d2)
        a = rand(Uniform(-pi/2,pi/2), data_size, 1)
        r = rand(Normal(0.1,0.01), data_size, 1)
        c1 = r.*cos.(a).+0.25.-abs(d2[1]+d2[2])/sqrt(2)
        c2 = r.*sin.(a).+(-d2[1]+d2[2])/sqrt(2)
        sim = [c1 c2]
        error = norm(sim .- target_data)
        return error
    end
    return prior_d, th_true, target_data, rho_lens
end
function run_apmc(N, np, apmc_repeats, model_lens, rho_lens, ker, df)
    epss = zeros(apmc_repeats, length(N))
    times = zeros(apmc_repeats, length(N))
    for (j,n) in enumerate(N)
        eps = zeros(apmc_repeats)
        time = zeros(apmc_repeats)
        for l in 1:apmc_repeats
            # APMC Output
            time[l] = @elapsed apmc_output = APMC(np,[model_lens],[rho_lens],prop=0.5,paccmin=0.01,n=n,covar="LinearShrinkage(DiagonalUnequalVariance(), :lw)",perturb=ker,df=df)
            eps[l] = apmc_output.epsilon[end]
            # AMPC Output using weights
            wts = apmc_output.wts[1,end]
            Nth = length(wts)
            ths = apmc_output.pts[1,end][:,wsample(1:Nth, weights(wts), Nth)]
            # Record 1st set of particles
            if l == 1
                writedlm("data\\TWO_MOONS\\ths\\ths_np"*string(np)*"_"*ker*"_df"*string(df)*"_n"*string(n)*"_it"*string(l)*".txt", ths)
            end
        end
        epss[:,j] = eps
        times[:,j] = time
        println("n=",string(n), ": avg. time=",string(mean(time)),": avg. eps=",string(mean(eps)))
        if ker == "Normal"
            writedlm("data\\TWO_MOONS\\eps_np"*string(np)*"_"*ker*".txt", epss)
            writedlm("data\\TWO_MOONS\\time_np"*string(np)*"_"*ker*".txt", times)
        else
            writedlm("data\\TWO_MOONS\\eps_np"*string(np)*"_"*ker*"_df"*string(df)*".txt", epss)
            writedlm("data\\TWO_MOONS\\time_np"*string(np)*"_"*ker*"_df"*string(df)*".txt", times)
        end
    end
end
data_size = 1000
np = 2000
N = 2 .^ (-15.0:1.0:5.0)
ker = "TDist"
df = 1
apmc_repeats = 20
prior_d, th_true, target_data, rho_lens = two_moons_model()

model_lens = [prior_d(), prior_d()]
run_apmc(N, np, apmc_repeats, model_lens, rho_lens, ker, df)

# Plot final eps
y0 = readdlm("data\\TWO_MOONS\\eps_np2000_Normal.txt",'\t',Float64,'\n')
y1 = readdlm("data\\TWO_MOONS\\eps_np2000_TDist_df1.txt",'\t',Float64,'\n')
plot(N,mean(y0,dims=1)', xaxis=:log,xlabel = "n",ylabel = "err (average over 20 iterations)", fillalpha=0.75, linewidth=2, title="err, np="*string(np)*" with scale estimation",label="Normal",legend=:top)
plot!(N,mean(y1,dims=1)', label="TDist df=1")

# Specific n run
apmc_output = APMC(np,[model_lens],[rho_lens],prop=0.5,paccmin=0.01,n=2^(-20),covar="LinearShrinkage(DiagonalUnequalVariance(), :lw)",perturb=ker,df=df)
print(apmc_output.epsilon[end])
