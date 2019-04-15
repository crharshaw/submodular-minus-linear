# vertex_cover_unconstrained.jl
#
# This is a test of the unconstrained vertex cover
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


# num of trials
q_vals = [1,2,3,4,5,6,7,8,9,10,11,12,13]
num_q = length(q_vals)
num_trials = 20

# initialize arrays
greedy_vals = zeros(num_q)
udg_vals = zeros(num_q, num_trials)

for (i,q) in enumerate(q_vals)

    # get costs
    cost = q_cost(deg, q) # get the cost

    # run the greedy algorithm 
    greedy_gain, _, _ = greedy(nb_list, cost, n)
    greedy_vals[i] = sum(greedy_gain)

    # run the unconstrained distorted greedy algorithm
    for j=1:num_trials
        udg_gain, _, _ = unconstrained_distorted_greedy(nb_list, cost)
        udg_vals[i,j] = sum(udg_gain)
    end
end

# generate statistics
@printf("\nGenerating statistics and plotting...\n")
mean_udg_vals = mean(udg_vals, dims=2)
std_udg_vals  = std(udg_vals, dims=2)

# plot
figure(figsize=figsize)
plot(q_vals, greedy_vals, lw=lw, marker="o", markersize=markersize, label="Greedy")
errorbar(q_vals, mean_udg_vals, yerr=std_udg_vals, c="tab:purple", lw=lw, marker="P", markersize=markersize, elinewidth=small_lw, label="UDG")
xlabel("Cost Parameter q", fontsize=large_fontsize)
ylabel("Objective Value", fontsize=large_fontsize)
title("Comparison of Solution Quality", fontsize=large_fontsize)
legend(fontsize=fontsize)
xticks(fontsize=small_fontsize) 
yticks(fontsize=small_fontsize)
show()
