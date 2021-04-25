using Plots
using Distributions
# Initialise prior and observed data
N = 10000; x = 4
prior = Beta(1, 1)
thetas = Array{Float64}(undef,N)
ys = Array{Int64}(undef,N)
# Get N samples of (θ,y) where y is sampled based on θ
for i = 1:N
    theta = rand(prior)
    y = rand(Binomial(10, theta))
    thetas[i] = theta
    ys[i] = y
end
# Scatter plot of θ vs y
ys4 = ys[ys.==4]; ths4 = thetas[ys.==4]; n4 = length(ys4); xv = 0:0.01:1
scatter(ys[ys.!=4],thetas[ys.!=4],label="y != 4",legend=:bottomright)
xlabel!("y"); ylabel!("θ"); title!("Plot of θ vs y")
scatter!(ys[ys.==4],thetas[ys.==4],label="y = 4")
# Frequency density vs actual posterior for y=x=4
histogram(ys4, ths4, bin=50, label="Frequency density",normalize=:pdf)
tstr = "Frenquency density vs Posterior for y=4, N="*string(n4)
xlabel!("θ"); ylabel!("f(θ)"); title!(tstr)
plot!(xv,pdf.(Beta(5,7),xv), label="True posterior",color="red", linewidth=3)
