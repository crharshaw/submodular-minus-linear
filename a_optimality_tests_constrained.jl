# a_optimality_tests_constrained.jl
#
# This file runs experiments for Bayesian A-optimal design - costs with cardinality constraints.
# Comparisons are between our stochastic distorted greedy and the vanilla greedy algorithm
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

# problem + algorithm parameters
k_vals =Array(1:20) # [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50] # cardinality constraint values
err_values = [0.2, 0.1] # epsilon = delta = err / 2 ==> error in approx is err and sweep running time is O( 1/err log^2(1/err) )
num_trials = 20 # number of times to run the algorithm (small for now)
p = 0.8 # cost proportionality

num_k = length(k_vals) # derived problem parameters, mostly bookkeeping
k_max = maximum(k_vals)
num_err = length(err_values)
sweep_vals = zeros(num_err, num_k, num_trials) # initialize arrays for measuring diagnostics
sweep_evals = zeros(num_err, num_k, num_trials)
sweep_time = zeros(num_err, num_k, num_trials)
sdg_vals = zeros(num_k)
sdg_evals = zeros(num_k)
sdg_time = zeros(num_k)

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

# theta_cov = 20.5 * Matrix{Float64}(I, d, d)
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
marginal_gains = zeros(n)
for e=1:n 
    x_e = X[:,e]
    z_e = theta_cov * x_e
    marginal_gains[e] = (z_e' * z_e) / (noise_var + z_e' * x_e)
end

cost = p * marginal_gains # costs linearly proportional to marginal gains
# max_mg = maximum(marginal_gains)
# cost = [p*(mg/max_mg)^2 * mg for mg in marginal_gains] # cost proportion increasing as marginal gain increases

delta_dg = 0.1

# burn runs for timing -- print stuff here so it doesn't affect timing
res = @timed greedy(X, theta_cov, noise_var, cost, k_max)
res = @timed sweep_dg(X, theta_cov, noise_var, cost, k_max, delta_dg)
res = @timed sweep_sdg(X, theta_cov, noise_var, cost, 5, err_values[1]/2, eps=err_values[1]/2) # burn run for timing

# run the greedy algorithm 
@printf("\nRunning the greedy algorithm\n")
res = @timed greedy(X, theta_cov, noise_var, cost, k_max)
greedy_gain, _, greedy_num_evals = res[1]
greedy_vals = cumsum(greedy_gain)
greedy_evals = cumsum(greedy_num_evals)
greedy_time = res[2]
@printf("\tDone. Completed in %f seconds\n", greedy_time)


# run for each cardinality k constraint
for (i,k) in enumerate(k_vals)
    @printf("\nk=%d \t(%d of %d)\n",k, i, num_k)

    @printf("\t\tRunning Sweep Distorted Greedy\n")

    # run the sweep greedy
    res = @timed sweep_dg(X, theta_cov, noise_var, cost, k, delta_dg)
    val_array, _, _, num_evals = res[1]
            
    # record measurements
    sdg_vals[i] = maximum(val_array)
    sdg_evals[i] = num_evals
    sdg_time[i] = res[2]

    # run the sweep SDG algorithm -- different err values
    for (r, err) in enumerate(err_values)
        @printf("\t\tRunning Sweep SDG (err=%.3f)\n", err)
        epsilon = err / 2.
        delta = err / 2.

        # many trials (to report means + std, etc)
        for j in 1:num_trials
            # run algorithm
            res = @timed sweep_sdg(X, theta_cov, noise_var, cost, k, delta, eps=epsilon)
            val_array, _, _, num_evals = res[1]
            
            # record measurements
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

# TODO: save all these statistics somewhere?

# plot the function values
figure(figsize=figsize)
plot(1:k_max, greedy_vals, lw=lw, marker="o", markersize=markersize, label="Greedy")
plot(k_vals, sdg_vals, lw=lw, marker="s", markersize=markersize, label="DG")
label_str = @sprintf("SDG (err=%.1f)", err_values[1])
errorbar(k_vals, mean_sweep_vals[1,:], yerr=std_sweep_vals[1,:], lw=lw, marker="D", markersize=markersize, elinewidth=small_lw, label=label_str)
label_str = @sprintf("SDG (err=%.1f)", err_values[2])
errorbar(k_vals, mean_sweep_vals[2,:], yerr=std_sweep_vals[2,:], lw=lw, marker="X", markersize=markersize, elinewidth=small_lw, label=label_str)

title_str = @sprintf("Comparison of Solution Quality")
title(title_str, fontsize=large_fontsize)
xlabel("Cardinality Constraint k", fontsize=large_fontsize)
ylabel("Objective Value", fontsize=large_fontsize)
legend(fontsize=fontsize)
ticks = vcat(1, Array(5:5:k_max))
tick_labels = [string(t) for t in ticks]
xticks(ticks, tick_labels, fontsize=small_fontsize)
ticks = Array(5:5:20)
tick_labels = [string(t) for t in ticks]
yticks(ticks, tick_labels)


# set up ticks
num_algs = length(err_values) + 2
ticks = 1:num_algs
tick_labels = ["Greedy", "DG"]
for err in err_values
    push!(tick_labels, @sprintf("SDG, (err=%.2f)", err))
end

# this is basically the same as timing so don't plot it anymore
# # plot the # of function evaluations -- as a bar plot
# figure(figsize=figsize)
# mean_evals = vcat([greedy_evals[end], sdg_evals[end]], mean_sweep_evals[:,end])
# bar(ticks, mean_evals)
# xticks(ticks, tick_labels)
# ylabel("# Function Evaluations", fontsize=fontsize)
# xlabel("Algorithms", fontsize=fontsize)
# title("Comparison of Function Evalutions", fontsize=fontsize)

# plot the timing as a bar plot
figure(figsize=figsize)
mean_times = vcat([greedy_time, sdg_time[end]], mean_sweep_time[:,end])
bar(1:num_algs, mean_times)
xticks(ticks, tick_labels, fontsize=fontsize)
ylabel("Run Time (sec)", fontsize=large_fontsize)
xlabel("Algorithms", fontsize=large_fontsize)
title("Comparison of CPU Time", fontsize=large_fontsize)
yticks(fontsize=fontsize)
show()
