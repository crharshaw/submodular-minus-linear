# a_optimality_sweep_tests.jl
#
# This performs a sweep test 
#

# import libraries and my code
using DelimitedFiles
using Printf
using Random
using Statistics
using LinearAlgebra
using PyPlot

@printf("Including my code...\n")
include("a_optimality_funs.jl")

# plotting parameters
figsize = (38, 4.8) # in inches, not that it super matters
large_fontsize = 22 # text size
fontsize = 18 # text size 
small_fontsize = 13
lw = 3 # line width
small_lw = 1.5 # smaller linewidth
markersize = 5

Random.seed!(1)

num_trials = 20 # number of times to run the algorithm (small for now)


# use real data matrix X from housing data
@printf("Loading data...\n")
X = readdlm("./data/housing_data.txt")
X = X' # take transpose
d, n = size(X)
# normalize each of the rows to be zero mean with std 1
for i=1:d
    X[i,:] = X[i,:] - ones(n)*mean(X[i,:])
    X[i,:] = X[i,:] /  std(X[i,:])
end

# create theta and noise var
C = randn(d,d)
lambda = (Array(1:d) / d).^2
D = Diagonal(lambda)
theta_cov = C' * D * C
noise_var =1/d

# report on the gamma lower bound
s = maximum([norm(X[:,i]) for i=1:n])
em = eigmax(theta_cov)
gamma_lb = 1. / (1. + ( (s^2 * em) / noise_var ))
@printf("The gamma lower bound is %f\n", gamma_lb)
@printf("\tMax Norm: %f\n\tMax Eig: %f\n\tSigma^2: %f\n\n", s, em, noise_var)

# create costs - proportional to initial marginal gain
p = 0.7 # cost proportionality
marginal_gains = zeros(n)
for e=1:n 
    x_e = X[:,e]
    z_e = theta_cov * x_e
    marginal_gains[e] = (z_e' * z_e) / (noise_var + z_e' * x_e)
end
cost = p * marginal_gains # costs linearly proportional to marginal gains

delta = 0.05
eps = 0.1

# run and plot the sweep values
figure(figsize=figsize)
k_vals = [5, 10, 20]
num_k = length(k_vals)
for (i,k) in enumerate(k_vals)

    @printf("\tWorking on k=%d (%d of %d)\n", k, i, num_k)

    # do distorted greedy sweep
    dg_vals, _, gamma_array, _ = sweep_dg(X, theta_cov, noise_var, cost, k, delta)
    
    sdg_vals = zeros(length(dg_vals), num_trials)
    # many trials for stochastic distorted greedy
    for j=1:num_trials 
        vals, _, _, _ = sweep_sdg(X, theta_cov, noise_var, cost, k, delta, eps=eps)
        sdg_vals[:,j] = vals 
    end

    # compute statistics
    sdg_val_mean = mean(sdg_vals, dims=2)
    sdg_val_std = std(sdg_vals, dims=2)

    # plot the results
    subplot(1, num_k, i)
    label_str = @sprintf("SDG (eps=%.1f)", eps)
    errorbar(gamma_array, sdg_val_mean, yerr=sdg_val_std, lw=lw, marker="D", markersize=markersize, elinewidth=small_lw, label=label_str, c="b")
    semilogx(gamma_array, dg_vals, lw=lw, marker="s", markersize=markersize, label="DG", c="m")
    title_str = @sprintf("k=%d", k)
    title(title_str, fontsize=large_fontsize)
    xlabel("gamma", fontsize=large_fontsize)
    if i==1
        ylabel("Objective Value", fontsize=large_fontsize)
        
    elseif i==2
        legend(loc="lower left", fontsize=fontsize)
    end
    xticks(fontsize=small_fontsize)
    yticks(fontsize=small_fontsize)
end
show()