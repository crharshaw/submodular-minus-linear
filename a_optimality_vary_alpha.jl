# a_optimality_vary_alpha.jl
#
# This is a test to see the effects of varying alpha
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
figsize = (10.0, 4.8) # in inches, not that it super matters
large_fontsize = 22 # text size
fontsize = 18 # text size 
small_fontsize = 13
lw = 3 # line width
small_lw = 1.5 # smaller linewidth
markersize = 10

Random.seed!(1)

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
s_max = maximum([norm(X[:,i]) for i=1:n])
s_min = minimum([norm(X[:,i]) for i=1:n])
em = eigmax(theta_cov)
gamma_lb = 1. / (1. + ( (s_max^2 * em) / noise_var ))
@printf("The gamma lower bound is %f\n", gamma_lb)
@printf("\tMax Norm: %f\n\tMin Norm: %f\n\tMax Eig: %f\n\tSigma^2: %f\n\n", s_max, s_min, em, noise_var)

# create costs - proportional to initial marginal gain
marginal_gains = zeros(n)
for e=1:n 
    x_e = X[:,e]
    z_e = theta_cov * x_e
    marginal_gains[e] = (z_e' * z_e) / (noise_var + z_e' * x_e)
end

# setting the cost search parameters
a_min = 0.0
a_max = 1.0
num_a = 15
a_vals = Array(range(a_min, stop=a_max, length=num_a))

k = 15 # this is the cardinality to test on

# stochastic params
eps = 0.05
delta = 0.05
num_trials = 20 # number of times to run the algorithm 

# initialize arrays
greedy_vals = zeros(num_a)
dg_vals = zeros(num_a)
sdg_vals = zeros(num_a, num_trials)

# run the algorithm for each cost
for (i,a) in enumerate(a_vals)

    @printf("Working on a=%f\t(%d of %d)\n", a, i, num_a)

    cost = a * marginal_gains

    # run the greedy algorithm
    @printf("\tRunning greedy\n")
    greedy_gain, _, _  = greedy(X, theta_cov, noise_var, cost, k)
    greedy_vals[i] = sum(greedy_gain)

    # run the distorted greedy sweep
    @printf("\tRunning distorted greedy\n")
    val_array, _, _, _ = sweep_dg(X, theta_cov, noise_var, cost, k, delta)
    dg_vals[i] = maximum(val_array)

    # run the stochastic distorted greedy sweep
    @printf("\tRunning stochastic distorted greedy\n")
    for j=1:num_trials
        val_array, _, _, _ = sweep_sdg(X, theta_cov, noise_var, cost, k, delta, eps=eps)
        sdg_vals[i,j] = maximum(val_array)
    end
end

# compute statistics
mean_sdg_vals = mean(sdg_vals, dims=2)
std_sdg_vals  = std(sdg_vals, dims=2)

# plot the results
figure(figsize=figsize)
plot(a_vals, greedy_vals, lw=lw, marker="o", markersize=markersize, label="Greedy")
plot(a_vals, dg_vals, lw=lw, marker="s", markersize=markersize, label="DG")
label_str = @sprintf("SDG (err=%.1f)", 2*eps)
errorbar(a_vals, mean_sdg_vals, yerr=std_sdg_vals, c="tab:red", lw=lw, marker="D", markersize=markersize, elinewidth=small_lw, label=label_str)
xlabel("Cost Penalty", fontsize=large_fontsize)
ylabel("Objective Value", fontsize=large_fontsize)
legend(fontsize=fontsize)
title("Varying Cost Penalty", fontsize=large_fontsize)
xticks(fontsize=small_fontsize)
show()



