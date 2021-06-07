using Plots
u = -1:0.01:1
K_uniform = 0*u .+ 0.5
K_triangular = 1 .- abs.(u)
K_epanechnikov = 0.75*(1 .- u.^2)
K_biweight = (15/16)*(1 .- u.^2).^2
K_guassian = exp.(-0.5*u.^2)/sqrt(2*pi)

plot(u, K_uniform,label="Uniform")
xlabel!("u"); ylabel!("K(u)"); title!("Kernel functions")
plot!(u, K_triangular,label="Triangular")
plot!(u, K_epanechnikov,label="Epanechnikov")
plot!(u, K_biweight,label="Biweight")
plot!(u, K_guassian,label="Guassian")
