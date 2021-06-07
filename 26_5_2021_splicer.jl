# Splicer

g = readdlm("data\\LOTKA_VOLTERRA\\eps_np2000_TDist_df1.txt",'\t',Float64,'\n')
h = readdlm("data\\LOTKA_VOLTERRA\\21_eps_np2000_TDist_df1.txt",'\t',Float64,'\n')
g[:,12:16] = h[:,1:5]
writedlm("data\\LOTKA_VOLTERRA\\eps_np2000_TDist_df1.txt",g)
