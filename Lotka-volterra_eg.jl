using Distributed
using Distributions
using UrlDownload
using StatsBase
using StatsPlots
using Plots
using DelimitedFiles
using DifferentialEquations

include("./types.jl")
include("./abc2.jl")

# Set random seed
seed = 201
Random.seed!(seed[1])
# Target data
url = "https://raw.githubusercontent.com/sbi-benchmark/sbibm/main/sbibm/tasks/lotka_volterra/files/num_observation_1/observation.csv"
df = urldownload(url)
targetdata = [df[1][i] for i = 1:20]
# True parameters
url2 = "https://raw.githubusercontent.com/sbi-benchmark/sbibm/main/sbibm/tasks/lotka_volterra/files/num_observation_1/true_parameters.csv"
df2 = urldownload(url2)
true_params = [df2[1][i] for i = 1:4]

# define Lotka Voltera ODE
function f(du,u,p,t)
  x, y = u
  alpha, beta, gamma, delta = p
  du[1] = alpha*x -beta*x*y
  du[2] = -gamma*y +delta*x*y
end

function rho_lens(d2)
  x0 = [30.0; 1.0]
  tspan = (0.0, 20.0)
  prob = ODEProblem(f, x0, tspan, d2)
  sol = try solve(prob,Tsit5(),maxiters=1e8,reltol=1e-08,abstol=1e-08,saveat=0.1)
  catch
    1000 .* ones(2,402)
  end
  X = sol[1,1:21:end]
  Y = sol[2,1:21:end]
  sim = try rand.(LogNormal.([log.(X);log.(Y)],0.1))
  catch
    1000 .* ones(20)
  end
  error = norm(sim .- targetdata)
  return error
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
          # Record particles
          writedlm("data\\LOTKA_VOLTERRA\\ths\\ths_np"*string(np)*"_"*ker*"_df"*string(df)*"_n"*string(n)*"_it"*string(l)*".txt", ths)
      end
      epss[:,j] = eps
      times[:,j] = time
      println("n=",string(n), ": avg. time=",string(mean(time)),": avg. eps=",string(mean(eps)))
      if ker == "Normal"
        writedlm("data\\LOTKA_VOLTERRA\\eps_np"*string(np)*"_"*ker*".txt", epss)
        writedlm("data\\LOTKA_VOLTERRA\\time_np"*string(np)*"_"*ker*".txt", times)
      else
        writedlm("data\\LOTKA_VOLTERRA\\eps_np"*string(np)*"_"*ker*"_df"*string(df)*".txt", epss)
        writedlm("data\\LOTKA_VOLTERRA\\time_np"*string(np)*"_"*ker*"_df"*string(df)*".txt", times)
      end
  end
end


# APMC Parameters

np = 2000
N = 2 .^ (-10.0:1.0:5.0)
ker = "TDist"
df = 5
apmc_repeats = 20

model_lens = [LogNormal(-0.125, 0.5), LogNormal(-3.0, 0.5),LogNormal(-0.125, 0.5), LogNormal(-3.0, 0.5)]
run_apmc(N, np, apmc_repeats, model_lens, rho_lens, ker, df)

# Plot final eps
y0 = readdlm("data\\LOTKA_VOLTERRA\\eps_np2000_Normal.txt",'\t',Float64,'\n')
y1 = readdlm("data\\LOTKA_VOLTERRA\\eps_np2000_TDist_df1.txt",'\t',Float64,'\n')
y2 = readdlm("data\\LOTKA_VOLTERRA\\eps_np2000_TDist_df5.txt",'\t',Float64,'\n')
plot(N,mean(y0,dims=1)'[1:16], xaxis=:log,xlabel = "n",ylabel = "err (average over 20 iterations)", fillalpha=0.75, linewidth=2, title="err, np="*string(np)*" with scale estimation",label="Normal",legend=:top)
plot!(N,mean(y1,dims=1)', label="TDist df=1")
plot!(N,mean(y2,dims=1)', label="TDist df=5")


# Load Posterior
ths_normal = readdlm("data\\LOTKA_VOLTERRA\\ths\\ths_np2000_Normal_n0.03125_it1.txt",'\t',Float64,'\n')
ths_cauchy = readdlm("data\\LOTKA_VOLTERRA\\ths\\ths_np2000_TDist_df1_n0.03125_it1.txt",'\t',Float64,'\n')
ths_tdist = readdlm("data\\LOTKA_VOLTERRA\\ths\\ths_np2000_TDist_df5_n0.03125_it1.txt",'\t',Float64,'\n')

# Plot Prior and Posterior
pars = ["\$\\alpha\$", "\$\\beta\$", "\$\\gamma\$", "\$\\delta\$"]
plots = []
for i in 1:4
  xvals = range(0,stop=6*std(model_lens[i]),length=501)
  if i % 2 == 1
    p = plot(xvals,pdf.(model_lens[i],xvals),fill = (0, 0.2),linecolor=1,title=pars[i],label="Prior",legend=:best)
  else
    p = plot(xvals,pdf.(model_lens[i],xvals),fill = (0, 0.2),linecolor=1,title=pars[i],label="Prior",legend=:topleft)
  end
  plot!([true_params[i]],seriestype=:vline,label="True value")
  density!(ths_normal[i,:],fill = (0, 0.2),label="Posterior (Normal kernel)")
  density!(ths_cauchy[i,:],fill = (0, 0.2),label="Posterior (T-Dist df=1 kernel)")
  density!(ths_tdist[i,:],fill = (0, 0.2),label="Posterior (T-Dist df=5 kernel)")
  push!(plots,p)
end
plot(plots...,size=(1000,500))

# Save figure
savefig("C:\\Users\\JohN100x1\\Documents\\_Programming\\Julia\\M4R Project\\images\\plots_final\\lotka-volterra_posteriors_maxiters")
