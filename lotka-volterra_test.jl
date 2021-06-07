using Distributed
using Distributions
using UrlDownload
using StatsBase
using Plots
using DelimitedFiles
using DifferentialEquations

include("./types.jl")
include("./abc2.jl")

# Set random seed
seed = 200
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
  sol = try solve(prob,Tsit5(),reltol=1e-08,abstol=1e-08,saveat=0.1)
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

# APMC Parameters

np = 2000
n = 2.0
ker = "TDist"
df = 1

model_lens = [LogNormal(-0.125, 0.5), LogNormal(-3.0, 0.5),LogNormal(-0.125, 0.5), LogNormal(-3.0, 0.5)]
TIME = @elapsed apmc_output = APMC(np,[model_lens],[rho_lens],prop=0.5,paccmin=0.01,n=n,covar="LinearShrinkage(DiagonalUnequalVariance(), :lw)",perturb=ker,df=df)
