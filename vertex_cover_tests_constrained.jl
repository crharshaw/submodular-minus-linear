# vertex_cover_tests_constrained.jl
#
# This file runs experiments for vertex cover with cardinality constraints.
# Comparisons are between our stochastic distorted greedy and the vanilla greedy algorithm.
#

# import libraries and my code
using DelimitedFiles
using Printf
using Random
using Statistics
using LinearAlgebra
using PyPlot

@printf("Including my code...\n")
include("vertex_cover_funs.jl")

# plotting parameters
figsize = (10.0, 4.8) # in inches, not that it super matters
large_fontsize = 22 # text size
fontsize = 18 # text size 
small_fontsize = 13
lw = 3 # line width
small_lw = 1.5 # smaller linewidth
markersize = 10

Random.seed!(1)

# Step 1: Load the EU Core Graph
@printf("Loading the EU Email Core Graph....")
n, nb_list, deg = get_email_eu_core()
# n, nb_list, deg = get_email_eu_full()
@printf("Done.\n\t%d Vertices\n\tMedian Degree: %d\n\tMax Degree: %d\n\tMean Degree: %d\n", n, median(deg), maximum(deg), mean(deg))

# decide number of cardinality constraints
k_vals = vcat([1,3,5], Array(10:10:130)) # cardinality constraint values
num_k = length(k_vals) # derived problem parameters, mostly bookkeeping
k_max = maximum(k_vals)

eps_values = [0.1, 0.2] # error in approx is err
num_eps = length(eps_values)
num_trials = 20 # number of times to run the algorithm (small for now)
q = 6 # cost parameter
cost = q_cost(deg, q) # get the cost

# create the arrays
dg_vals = zeros(num_k)
sdg_vals = zeros(num_eps, num_k, num_trials)
dg_times = zeros(num_k)
sdg_times = zeros(num_eps, num_k, num_trials)
dg_evals = zeros(num_k)
sdg_evals = zeros(num_eps, num_k, num_trials)

# run all algorithms once for a burn run
@timed greedy(nb_list, cost, k_vals[2])
@timed distorted_greedy(nb_list, cost, k_vals[2])
@timed stochastic_distorted_greedy(nb_list, cost, k_vals[2], eps=0.1)

# run the greedy algorithm 
@printf("\nRunning the greedy algorithm\n")
res = @timed greedy(nb_list, cost, k_max)
greedy_gain, _, evals = res[1]
greedy_vals = cumsum(greedy_gain)
greedy_evals = sum(evals)
greedy_time = res[2]
@printf("\tDone.\n\n")

# for each value of k, run our algorithms
for (i,k) in enumerate(k_vals)

    @printf("\nWorking on k=%d (%d of %d)\n", k, i, num_k)

    # distorted greedy
    @printf("\tRunning distorted greedy\n")
    res = @timed distorted_greedy(nb_list, cost, k)
    dg_gain, _, evals = res[1]
    dg_vals[i] = sum(dg_gain)
    dg_evals[i] = evals
    dg_times[i] = res[2]
    
    # stochastic distorted greedy
    for (r,eps) in enumerate(eps_values)
        @printf("\tRunning stochastic (eps=%.2f)\n",eps)
        for j=1:num_trials
            res = @timed stochastic_distorted_greedy(nb_list, cost, k, eps=eps)
            sdg_gain, _, evals = res[1]
            sdg_vals[r,i,j] = sum(sdg_gain)
            sdg_evals[r,i,j] = evals
            sdg_times[r,i,j] = res[2]
        end
    end
end

# generate statistics
@printf("\nGenerating statistics and plotting...\n")
mean_sdg_vals = mean(sdg_vals, dims=3)
std_sdg_vals  = std(sdg_vals, dims=3)
mean_sdg_times = mean(sdg_times, dims=3)
std_sdg_times = mean(sdg_times, dims=3)
mean_sdg_evals = mean(sdg_evals, dims=3)
std_sdg_evals = mean(sdg_evals, dims=3)

# plot the function values
figure(figsize=figsize)
plot(k_vals, greedy_vals[k_vals], lw=lw, marker="o", markersize=markersize, label="Greedy")
plot(k_vals, dg_vals, lw=lw, marker="s", markersize=markersize, label="DG")
label_str = @sprintf("SDG (eps=%.1f)", eps_values[1])
errorbar(k_vals, mean_sdg_vals[1,:], yerr=std_sdg_vals[1,:], lw=lw, marker="D", markersize=markersize, elinewidth=small_lw, label=label_str)
label_str = @sprintf("SDG (eps=%.1f)", eps_values[2])
errorbar(k_vals, mean_sdg_vals[2,:], yerr=std_sdg_vals[2,:], lw=lw, marker="X", markersize=markersize, elinewidth=small_lw, label=label_str)
title("Comparison of Solution Quality", fontsize=large_fontsize)
xlabel("Cardinality Constraint k", fontsize=large_fontsize)
ylabel("Objective Value", fontsize=large_fontsize)
legend(fontsize=fontsize)
xticks(fontsize=small_fontsize)
yticks(fontsize=small_fontsize)
# ticks = vcat(1, Array(10:20:k_max))
# tick_labels = [string(t) for t in ticks]
# xticks(ticks, tick_labels, fontsize=small_fontsize)
# ticks = Array(5:5:20)
# tick_labels = [string(t) for t in ticks]
# yticks(ticks, tick_labels)


# set up ticks
num_algs = length(eps_values) + 2
ticks = 1:num_algs
tick_labels = ["Greedy", "DG"]
for eps in eps_values
    push!(tick_labels, @sprintf("SDG, (eps=%.1f)", eps))
end

# plot the # of function evaluations -- as a bar plot
figure(figsize=figsize)
mean_evals = vcat([greedy_evals, dg_evals[end]], mean_sdg_evals[:,end])
bar(ticks, mean_evals)
xticks(ticks, tick_labels, fontsize=fontsize)
ylabel("# Function Evaluations", fontsize=large_fontsize)
xlabel("Algorithms", fontsize=large_fontsize)
title("Comparison of Function Evalutions", fontsize=large_fontsize)
yticks(fontsize=small_fontsize)

# plot the timing as a bar plot
figure(figsize=figsize)
mean_times = vcat([greedy_time, dg_times[end]], mean_sdg_times[:,end])
bar(ticks, mean_times)
xticks(ticks, tick_labels, fontsize=fontsize)
ylabel("Run Time (sec)", fontsize=large_fontsize)
xlabel("Algorithms", fontsize=large_fontsize)
title("Comparison of CPU Time", fontsize=large_fontsize)
show()