# a_optimality_test_unconstrained.jl
#
# This file runs experiments for unconstrained Bayesian A-optimal design with costs.
# Comparisons are between our unconstrained distorted greedy and the vanilla greedy algorithm.
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

Random.seed!(1)

# problem + algorithm parameters
n = 200 # number of ground elements
d = 50 # number of observations in Bayesian
num_trials = 30 # number of times to run the algorithm (small for now)
delta_vals = [0.2, 0.1, 0.01]
# p_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95] # cost proportionality
p_vals = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
rep_p = 5 # the proportionality used for reporting timing

num_p = length(p_vals)
num_delta = length(delta_vals)
greedy_vals = zeros(num_p) # initialize arrays for measuring diagnostics
greedy_evals = zeros(num_p)
greedy_time = zeros(num_p)
sweep_vals = zeros(num_delta, num_p, num_trials)
sweep_evals = zeros(num_delta, num_p, num_trials)
sweep_time = zeros(num_delta, num_p, num_trials)

# create Bayesian a-optimality instance
#   1. X            uniformly random unit vectors 
#   2. theta Cov    identitiy matrix
#   3. noise        1 
# @printf("Creating data...\n")
# X = randn(d,n)
# for i=1:n
#     X[:,i] /= norm(X[:,i])
# end

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

# create a covariance which is pretty skewed
C = randn(d,d)
lambda = (Array(1:d) / d).^4
D = Diagonal(lambda)
theta_cov = C' * D * C
noise_var =(1/d)^2

# record initial marginal gains
marginal_gains = zeros(n)
for e=1:n 
    x_e = X[:,e]
    z_e = theta_cov * x_e
    marginal_gains[e] = (z_e' * z_e) / (noise_var + z_e' * x_e)
end

# run two timed instances to burn
cost = 0.5 * marginal_gains
res = @timed greedy(X, theta_cov, noise_var, cost, n)
res = @timed sweep_udg(X, theta_cov, noise_var, cost, delta_vals[1])

# create costs, run algorithm
for (i, p) in enumerate(p_vals)

    cost = p * marginal_gains # costs linearly proportional to marginal gains
    @printf("\nRunning on costs proportions p=%f\t(%d of %d)\n", p, i, num_p)

    # vanilla greedy algorithm
    @printf("\trunning greedy\n")
    res = @timed greedy(X, theta_cov, noise_var, cost, n)
    greedy_gain, _, greedy_num_evals = res[1]
    greedy_vals[i] = cumsum(greedy_gain)[end]
    greedy_evals[i] = cumsum(greedy_num_evals)[end]
    greedy_time[i] = res[2]

    # distorted greedy algorithm
    for (r, delta) in enumerate(delta_vals)
        @printf("\trunning stochastic, (delta=%.3f)\n", delta)
        for j=1:num_trials
            res = @timed sweep_udg(X, theta_cov, noise_var, cost, delta)
            val_array, _, _, num_evals = res[1]
            sweep_vals[r,i,j] = maximum(val_array)
            sweep_evals[r,i,j] = num_evals
            sweep_time[r,i,j] = res[2]
        end
    end
end

# generate statistics
@printf("\nGenerating statistics and plotting...\n")
mean_sweep_vals = mean(sweep_vals, dims=3)
std_sweep_vals  = std(sweep_vals, dims=3)
mean_sweep_evals = mean(sweep_evals, dims=3)
std_sweep_evals = std(sweep_evals, dims=3)
mean_sweep_time = mean(sweep_time, dims=3)
std_sweep_time = std(sweep_time, dims=3)

# make plot labels
alg_labels = ["Greedy"]
for delta in delta_vals
    push!(alg_labels, @sprintf("Stochastic, (delta=%.2f)", delta))
end
num_algs = length(alg_labels)

# Figure 1 -- Solution quality as costs increase
figure(1)
plot(p_vals, greedy_vals, label=alg_labels[1])
for r=1:num_delta
    errorbar(p_vals, mean_sweep_vals[r,:], yerr=std_sweep_vals[r,:], label=alg_labels[r+1])
end
title("Comparison of Solution Quality with Increasing Costs")
xlabel("Cost Proportion")
ylabel("g(S) + l(S)")
legend()

# Figure 2 -- # of function evalutions
ticks = 1:num_algs
figure(2)
mean_evals = vcat(greedy_evals[rep_p], mean_sweep_evals[:,rep_p])
bar(ticks, mean_evals)
xticks(ticks, alg_labels)
ylabel("# Function Evaluations")
xlabel("Algorithms")
title("Comparison of Function Evalutions")

# Figure 3 -- CPU timing
figure(3)
mean_times = vcat(greedy_time[rep_p], mean_sweep_time[:,rep_p])
bar(ticks, mean_times)
xticks(ticks, alg_labels)
ylabel("Run Time (sec)")
xlabel("Algorithms")
title("Comparison of CPU Time")
show()



