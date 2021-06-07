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
epsilon = 1
K(u) = exp((0.5)*((u-ys4)/epsilon).^2)/(sqrt(2*pi)*epsilon)
