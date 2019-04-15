# a_optimality_code.jl
#
# This file contains code to optimize Bayesian A-optimal design objectives with costs.
#

using StatsBase
using LinearAlgebra

function max_marginal_search(X, M_inv, noise_var, cost, search_elm; df=1.0)
    """
    # max_marginal_search
    # This implements the search for an element of maximal marginal gain
    # The most important part of this algorithm is that
    #
    #   f( e | S ) = < z_e , z_e > / ( sigma^2 + < z_e , x_e > )
    # 
    # where z_e = inv(M_S) * x_e
    # and M_S = Cov^-1 + (1/sigma^2) * X_S X_S^T
    #
    # This function only requires access to inv(M_S) and updating happens
    # elsehere.
    #
    # INPUTS
    #   X           d x n data matrix, columns are possible stimuli x_1, ... x_n
    #   M_inv       the inverse of M_S = Cov^-1 + (1/sigma^2) * X_S X_S^T
    #   noise_var   the noise variance, denoted sigma^2 in notes
    #   cost        a n vector of element costs (positive)
    #   search_elm  the set of elements to search over
    #   df          a distortion factor, only relevant for distorted algorithms
    #
    # OUTPUTS
    #   max_gain    maximal gain
    #   best_e      element with maximium gain in available
    #   best_e_mg   the g marginal gain of the maximal element, i.e. g( best_e | S )
    #
    """

    # search over all elements
    max_gain = -Inf 
    best_e = nothing
    best_e_mg = 0.0
    for e in search_elm

        # compute gain
        x_e = X[:,e]
        z_e = M_inv * x_e
        g_gain = (z_e' * z_e) / (noise_var + z_e' * x_e)
        gain = df*g_gain - cost[e]

        # check if its maximal
        if gain > max_gain
            best_e = e 
            max_gain = gain 
            best_e_mg = g_gain
        end
    end
    return max_gain, best_e, best_e_mg
end

function greedy(X, theta_cov, noise_var, cost, k; verbose=false)
    """
    # greedy
    # This function is an implementation of the vanilla greedy algorithm.
    # The most important part of this algorithm is that
    #
    #   f( e | S ) = < z_e , z_e > / ( sigma^2 + < z_e , x_e > )
    # 
    # where z_e = inv(M_S) * x_e
    # and M_S = Cov^-1 + (1/sigma^2) * X_S X_S^T
    #
    # crucially, the algorithm maintains the inverse of M_S which is 
    # updated via the Woodbury matrix identity. More precisely, 
    #
    # inv(M_(S+e)) = inv(M_S) - (z_e * z_e') / (sigma^2 + < x_e ,  z_e > )
    #
    # where, again, z_e = inv(M_S) * x_e
    # In this way, inv(M_S) is updated as elements are added.
    #
    # INPUTS
    #   X           d x n data matrix, columns are possible stimuli x_1, ... x_n
    #   theta_cov   d x d covariance matrix for theta (this is the bayesian prior)
    #   noise_var   the noise variance, denoted sigma^2 in notes
    #   cost        a n vector of element costs (positive)
    #   k           cardinality constraints
    #   verbose     if true, prints the condition number of inv(M_S) throughout
    #
    # OUTPUTS
    #   gain_array  a length k array of the gains at each iter, i.e. f( e | S ) = g( e | S) + ell( e | S)
    #   elm_array   a length k array of elements added at each iter
    #   evals_array a length k array of number of evals at each iter
    #
    """

    # get dimensions 
    d, n = size(X)
    available_e = Set(1:n)

    # initialize the solution and record arrays
    sol = Set()
    gain_array = zeros(k)
    elm_array = zeros(Int64, k)
    evals_array = zeros(Int64, k)

    # initialize M_inv
    M_inv = theta_cov

    # greedy updates
    for iter=1:k

        if verbose
            @printf("\tIter %d \n\t\tCondition Number: %f\n\t\tMax Eig: %f\n", iter, cond(M_inv), eigmax(M_inv))
        end

        # search for the maximal element -- faster now that it has its own function
        max_gain, best_e, best_e_mg = max_marginal_search(X, M_inv, noise_var, cost, available_e)
        
        # update number of function evals
        evals_array[iter] = length(available_e)

        if max_gain > 0
            # update solution
            union!(sol, best_e)
            setdiff!(available_e, best_e)

            # update M_inv - via Woodbury matrix identity
            x_e = X[:,best_e]
            z_e = M_inv * x_e
            M_inv -= (z_e * z_e') / (noise_var + x_e' * z_e)

            # update record arrays
            gain_array[iter] = best_e_mg - cost[best_e]
            elm_array[iter] = best_e
        else 
            break # the algorithm can't make progress, so quit
        end
    end
    return gain_array, elm_array, evals_array
end


function distorted_greedy(X, theta_cov, noise_var, cost, k; gamma=1.0)
    """
    # distorted_greedy
    # This function is an implementation of the distorted greedy algorithm.
    # The most important part of this algorithm is that
    #
    #   f( e | S ) = < z_e , z_e > / ( sigma^2 + < z_e , x_e > )
    # 
    # where z_e = inv(M_S) * x_e
    # and M_S = Cov^-1 + (1/sigma^2) * X_S X_S^T
    #
    # crucially, the algorithm maintains the inverse of M_S which is 
    # updated via the Woodbury matrix identity. More precisely, 
    #
    # inv(M_(S+e)) = inv(M_S) - (z_e * z_e') / (sigma^2 + < x_e ,  z_e > )
    #
    # where, again, z_e = inv(M_S) * x_e
    # In this way, inv(M_S) is updated as elements are added.
    #
    # INPUTS
    #   X           d x n data matrix, columns are possible stimuli x_1, ... x_n
    #   theta_cov   d x d covariance matrix for theta (this is the bayesian prior)
    #   noise_var   the noise variance, denoted sigma^2 in notes
    #   cost        a n vector of element costs (positive)
    #   k           cardinality constraints
    #
    # OUTPUTS
    #   gain_array  a length k array of the gains at each iter, i.e. f( e | S ) = g( e | S) + ell( e | S)
    #   elm_array   a length k array of elements added at each iter
    #   num_evals   the number of function evaluations
    #
    """

    # get dimensions 
    d, n = size(X)
    available_e = Set(1:n)

    # initialize the solution and record arrays
    sol = Set()
    gain_array = zeros(k)
    elm_array = zeros(Int64, k)
    num_evals = 0

    # initialize M_inv
    M_inv = theta_cov

    # greedy updates
    for i=0:k-1

        # compute distortion factor
        df = (1 - (gamma / k))^(k - (i+1)) 

        # search for the maximal element -- faster now that it has its own function
        max_gain, best_e, best_e_mg = max_marginal_search(X, M_inv, noise_var, cost, available_e, df=df)
        
        # update number of function evals
        num_evals += length(available_e)

        if max_gain > 0
            # update solution
            union!(sol, best_e)
            setdiff!(available_e, best_e)

            # update M_inv - via Woodbury matrix identity
            x_e = X[:,best_e]
            z_e = M_inv * x_e
            M_inv -= (z_e * z_e') / (noise_var + x_e' * z_e)

            # update record arrays
            gain_array[i+1] = best_e_mg - cost[best_e]
            elm_array[i+1] = best_e
        end
    end
    return gain_array, elm_array, num_evals
end


function sweep_dg(X, theta_cov, noise_var, cost, k, delta; lb=0.0)
    """
    # sweep_sdg
    # This function is an implementation of sweeping with the
    # stochastic  distorted greedy algorithm
    #
    # INPUTS
    #   X           d x n data matrix, columns are possible stimuli x_1, ... x_n
    #   theta_cov   d x d covariance matrix for theta (this is the bayesian prior)
    #   noise_var   the noise variance, denoted sigma^2 in notes
    #   cost        a n vector of element costs (positive)
    #   k           cardinality constraint
    #   delta       sweep parameter / approximation parameter
    #   lb          optional lower bound on gamma (default = 0)
    #
    # OUTPUTS
    #   val_array   an array of function values at each iter, i.e. f( S_t )
    #   sol_array   an array of the solution sets at each iter, i.e. f( S_t )
    #   gamma_array an array of the gamma values at each iter
    #   num_evals   the number of function evaluations
    #
    """
    
    # calculate number of sweep iterations
    B = max(lb, delta)
    num_iter = convert(Int64, ceil(log(1. / B) / delta ))

    # initialize arrays
    val_array = zeros(num_iter)
    gamma_array = zeros(num_iter)
    sol_array = []
    num_evals = 0

    # sweep through guesses of gamma
    for t=0:num_iter-1

        # gamma guess
        gamma_t = (1 - delta) ^ t

        # run the stochastic distorted greedy subroutine
        gain_array_t, elm_array_t, num_evals_t = distorted_greedy(X, theta_cov, noise_var, cost, k, gamma=gamma_t)
        
        # update the records
        val_array[t+1] = sum(gain_array_t) # summing marginal gains == total function value
        gamma_array[t+1] = gamma_t
        push!(sol_array, setdiff!(Set(elm_array_t),0))
        num_evals += num_evals_t
    end

    return val_array, sol_array, gamma_array, num_evals
end


function stochastic_distorted_greedy(X, theta_cov, noise_var, cost, k; gamma=1.0, eps=0.2)
    """
    # stochastic_distorted_greedy
    # This function is an implementation of the stochastic distorted greedy algorithm
    # The most important part of this algorithm is that
    #
    #   f( e | S ) = < z_e , z_e > / ( sigma^2 + < z_e , x_e > )
    # 
    # where z_e = inv(M_S) * x_e
    # and M_S = Cov^-1 + (1/sigma^2) * X_S X_S^T
    #
    # crucially, the algorithm maintains the inverse of M_S which is 
    # updated via the Woodbury matrix identity. More precisely, 
    #
    # inv(M_(S+e)) = inv(M_S) - (z_e * z_e') / (sigma^2 + < x_e ,  z_e > )
    #
    # where, again, z_e = inv(M_S) * x_e
    # In this way, inv(M_S) is updated as elements are added.
    #
    # INPUTS
    #   X           d x n data matrix, columns are possible stimuli x_1, ... x_n
    #   theta_cov   d x d covariance matrix for theta (this is the bayesian prior)
    #   noise_var   the noise variance, denoted sigma^2 in notes
    #   cost        a n vector of element costs (positive)
    #   k           cardinality constraint
    #   gamma       weak submodularity parameter
    #   eps         epsilon, sampling parameter
    #
    # OUTPUTS
    #   gain_array  a length k array of the gains at each iter, i.e. f( e | S ) = g( e | S) + ell( e | S)
    #   elm_array   a length k array of elements added at each iter
    #   num_evals   the number of function evaluations
    #
    """

    # initialize set and record arrays
    sol = Set()
    gain_array = zeros(k)
    elm_array = zeros(Int64, k)
    num_evals = 0

    # set the sample size to n/k log(1/eps)
    sample_size = min(convert(Int64, ceil(n/k * log(1/eps))), n)
    available_e = Set(1:n)

    # initialize M_inv
    M_inv = theta_cov

    for i=0:k-1 

        # compute distortion factor
        df = (1 - (gamma / k))^(k - (i+1))  

        # sample a subset, find maximal element inside -- faster now that it has its own function 
        sample_size = min(length(available_e), sample_size)
        search_elm = sample(collect(available_e), sample_size, replace=false)
        max_gain, best_e, best_e_mg = max_marginal_search(X, M_inv, noise_var, cost, search_elm, df=df)
        
        # update number of function evals
        num_evals += sample_size

        # add element if the distorted gain is > 0
        if max_gain > 0

            # update solution
            union!(sol, best_e)
            setdiff!(available_e, best_e)

            # update M_inv - via Woodbury matrix identity
            x_e = X[:,best_e]
            z_e = M_inv * x_e
            M_inv -= (z_e * z_e') / (noise_var + x_e' * z_e)

            # update record arrays
            gain_array[i+1] = best_e_mg - cost[best_e]
            elm_array[i+1] = best_e
        end
    end
    return gain_array, elm_array, num_evals
end


function sweep_sdg(X, theta_cov, noise_var, cost, k, delta; lb=0.0, eps=0.2)
    """
    # sweep_sdg
    # This function is an implementation of sweeping with the
    # stochastic  distorted greedy algorithm
    #
    # INPUTS
    #   X           d x n data matrix, columns are possible stimuli x_1, ... x_n
    #   theta_cov   d x d covariance matrix for theta (this is the bayesian prior)
    #   noise_var   the noise variance, denoted sigma^2 in notes
    #   cost        a n vector of element costs (positive)
    #   k           cardinality constraint
    #   delta       sweep parameter / approximation parameter
    #   lb          optional lower bound on gamma (default = 0)
    #   eps         epsilon, sampling parameter  
    #
    # OUTPUTS
    #   val_array   an array of function values at each iter, i.e. f( S_t )
    #   sol_array   an array of the solution sets at each iter, i.e. f( S_t )
    #   gamma_array an array of the gamma values at each iter
    #   num_evals   the number of function evaluations
    #
    """
    
    # calculate number of sweep iterations
    B = max(lb, delta)
    num_iter = convert(Int64, ceil(log(1. / B) / delta ))

    # initialize arrays
    val_array = zeros(num_iter)
    gamma_array = zeros(num_iter)
    sol_array = []
    num_evals = 0

    # sweep through guesses of gamma
    for t=0:num_iter-1

        # gamma guess
        gamma_t = (1 - delta) ^ t

        # run the stochastic distorted greedy subroutine
        gain_array_t, elm_array_t, num_evals_t = stochastic_distorted_greedy(X, theta_cov, noise_var, cost, k; gamma=gamma_t, eps=eps)
        
        # update the records
        val_array[t+1] = sum(gain_array_t) # summing marginal gains == total function value
        gamma_array[t+1] = gamma_t
        push!(sol_array, setdiff!(Set(elm_array_t),0))
        num_evals += num_evals_t
    end

    return val_array, sol_array, gamma_array, num_evals
end

function unconstrained_distorted_greedy(X, theta_cov, noise_var, cost; gamma=1.0)
    """
    # unconstrained_distorted_greedy
    # This function is an implementation of the unconstrained distorted greedy algorithm
    # The most important part of this algorithm is that
    #
    #   f( e | S ) = < z_e , z_e > / ( sigma^2 + < z_e , x_e > )
    # 
    # where z_e = inv(M_S) * x_e
    # and M_S = Cov^-1 + (1/sigma^2) * X_S X_S^T
    #
    # crucially, the algorithm maintains the inverse of M_S which is 
    # updated via the Woodbury matrix identity. More precisely, 
    #
    # inv(M_(S+e)) = inv(M_S) - (z_e * z_e') / (sigma^2 + < x_e ,  z_e > )
    #
    # where, again, z_e = inv(M_S) * x_e
    # In this way, inv(M_S) is updated as elements are added.
    #
    # INPUTS
    #   X           d x n data matrix, columns are possible stimuli x_1, ... x_n
    #   theta_cov   d x d covariance matrix for theta (this is the bayesian prior)
    #   noise_var   the noise variance, denoted sigma^2 in notes
    #   cost        a n vector of element costs (positive)
    #   gamma       weak submodularity parameter 
    #
    # OUTPUTS
    #   gain_array  a length k array of the gains at each iter, i.e. f( e | S ) = g( e | S) + ell( e | S)
    #   elm_array   a length k array of elements added at each iter
    #   num_evals   the number of function evaluations
    #
    """

    # get dimensions
    n,d = size(X)

    # initialize set and record arrays
    k = 0
    sol = Set()
    gain_array = zeros(n)
    elm_array = zeros(Int64, n)
    num_evals = 0

    # initialize M_inv
    M_inv = theta_cov

    # initialize set of available elements
    available_e = Set(1:n)

    for i=0:n-1

        # sample a single element -- forbid choosing e from S, although we allow in analysis
        e = sample(collect(available_e))
        if e in sol # we should never get here - but just in case...
            continue 
        end

        # compute distortion factor
        df = (1. - (gamma / n))^(n - (i+1)) 

        # compute gain
        x_e = X[:,e]
        z_e = M_inv * x_e
        g_gain = (z_e' * z_e) / (noise_var + z_e' * x_e)
        gain = df*g_gain - cost[e]

        # update number of evaluations
        num_evals += 1

        # add element if distorted gain is large enough
        if gain > 0.

            # udpate solution
            union!(sol, e)
            setdiff!(available_e, e)

            # update M_inv - via Woodbury matrix identity
            x_e = X[:,e]
            z_e = M_inv * x_e
            M_inv -= (z_e * z_e') / (noise_var + x_e' * z_e)

            # update record arrays
            gain_array[i+1] = g_gain - cost[e]
            elm_array[i+1] = e
        end
    end
    return gain_array, elm_array, num_evals
end


function sweep_udg(X, theta_cov, noise_var, cost, delta; lb=0.0)
    """
    # sweep_udg
    # This function is an implementation of sweeping with the
    # unconstrained distorted greedy algorithm
    #
    # INPUTS
    #   X           d x n data matrix, columns are possible stimuli x_1, ... x_n
    #   theta_cov   d x d covariance matrix for theta (this is the bayesian prior)
    #   noise_var   the noise variance, denoted sigma^2 in notes
    #   cost        a n vector of element costs (positive)
    #   delta       sweep parameter / approximation parameter
    #   lb          optional lower bound on gamma (default = 0) 
    #
    # OUTPUTS
    #   val_array   an array of function values at each iter, i.e. f( S_t )
    #   sol_array   an array of the solution sets at each iter, i.e. f( S_t )
    #   gamma_array an array of the gamma values at each iter
    #   num_evals   the number of function evaluations
    #
    """
    
    # calculate number of sweep iterations
    B = max(lb, delta)
    num_iter = convert(Int64, ceil(log(1. / B) / delta ))

    # initialize arrays
    val_array = zeros(num_iter)
    gamma_array = zeros(num_iter)
    sol_array = []
    num_evals = 0

    # sweep through guesses of gamma
    for t=0:num_iter-1

        # gamma guess
        gamma_t = (1 - delta) ^ t

        # run the stochastic distorted greedy subroutine
        gain_array_t, elm_array_t, num_evals_t = unconstrained_distorted_greedy(X, theta_cov, noise_var, cost, gamma=gamma_t)
        
        # update the records
        val_array[t+1] = sum(gain_array_t) # summing marginal gains == total function value
        gamma_array[t+1] = gamma_t
        push!(sol_array, setdiff!(Set(elm_array_t),0))
        num_evals += num_evals_t
    end
    return val_array, sol_array, gamma_array, num_evals
end