# Submodular Minus Linear

This repository contains code used for the experiments in our paper "Submodular Maximization beyond Non-negativity: Guarantees, Fast Algorithms, and Applications". The authors are Christopher Harshaw (me), Moran Feldman, Justin Ward, and Amin Karbasi. 

In that paper, we study the optimization problem
$$ \max_{|S| \leq k} g(S) - c(S) $$
where $g$ is a non-negative $\gamma$-weakly monotone submodular function and $c$ is a non-negative linear function. If one interprets $g(S)$ as a revenue obtained by producing a set items $S$ and $c(S)$ to be the cost of those items, then $f(S) = g(S) - c(S)$ may naturally interpreted as a profit. This formulation may also be useful for adding regularizing penalty term to existing submodular maximization techniques for machine learning tasks. However, existing algorithms cannot solve this formulation as the objective $f$ may take negative values and is itself not $\gamma$-weakly submodular.

In that paper, we presented a suite of algorithms for the problem of interest. The main idea behind each of these algorithms is to optimize a surrogate objective, which changes throughout the algorithm, preventing us from getting stuck in poor local optima. A summarization of the results is presented below (see the paper for more details)
- **Distorted Greedy** A deterministic algorithm which produces a set $S$ such that $g(S) - c(S) \geq (1 - e^{\gamma})g(S^*) - c(S^*)$ using $O(nk)$ function evaluations. If $g$ is regarded as revenue and $c$ as a cost, then this guarantee intuitively states that the algorithm will return a solution whose total profit is at least as much as would be obtained by paying the same cost as the optimal solution while gaining at least $(1 ? e^{-\gamma} )$ as much revenue.
- **Stochastic Random Greedy** A randomized variant which produces a random set $S$ with an expected approximation guarantee of $\mathbb{E} \left[ g(S) - c(S) \right] \geq (1 - e^{\gamma} - \epsilon) g(S^*) - c(S^*)$, requiring only $O \left( n \log \frac{1}{\epsilon} \right)$ function evaluations. Based on sampling techniques of Mirzasoleiman et al., 2015.
- **Unconstrained Distorted Greedy**  A randomized algorithm for solving the unconstrained problem ($k=n$) which achieves the $(1 - e^{-\gamma})$ approximation guarantee but using only $O(n)$ function evaluations.
- **Gamma Sweep** The above algorithms assume knowledge of $\gamma$, the submodularity ratio. When $g$ is submodular, $\gamma = 1$ is known; in practice, one can usually only lower bound $\gamma$. This is a meta-algorithm which guessing $\gamma$ to produce a good solution when $\gamma$ is unknown. The guessing step in the meta-algorithm loses a $\delta$ additive factor in the approximation and increases run time by a multiplicative $O \left( \frac{1}{\delta} \log \frac{1}{\delta} \right)$ factor.

## The Code

We are not actively maintaining this code base; rather, we have made it available so that
our experimental results are more easily reproducible. 
Please email Chris Harshaw with any questions regarding these algorithms.
The optimization algorithms are written in the Julia programming language. 

A few notes about the code:
1. The main functions can be found in `a_optimality_funs.jl` and `vertex_cover_funs.jl`. Other files contain experiments.
2. Our code uses standard Julia libraries, but the dependencies must manually be installed (we haven't provided an install file)
3. Code was written for the Julia Language version 1.0, using earlier versions will throw errors
4. Certain experiments make reference to a `/data` folder, which is not included here due to size limitations. One needs to manually populate the data folder with the publicly available data to reproduce these experiments.

