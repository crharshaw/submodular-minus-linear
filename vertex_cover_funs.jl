# vertex_cover_funs.jl
#
# This file contains code to optimize vertex cover objectives with costs.
#


using Printf
using StatsBase
using DataStructures
using DelimitedFiles


function initialize_pq(cost)
    """
    # initialize_pq
    # Initializes a priority queue for lazy evaluations
    #   NOTE: this isn't actually a priority queue
    #
    # This priority queue is just a matrix with size (n,5) where each row is
    #       ( e , g_e , c_e , v) 
    # where e is the element, g_e is most recently computed marginal gain, c_e is cost, 
    # and v = a*g_e + c_e . So, we update v at every iteration and re-sort.
    #
    # Initially, all e are not yet queried to g_e = Inf and then v = Inf
    #
    # INPUTS
    #   cost        a n vector of element costs (positive)
    #
    # OUTPUTS
    #   pq          the priority queue, a matrix of size n by 4
    """    

    # get size of ground set
    n = length(cost)

    # initialize the priority queue -- typed to float, so if we use e we have to convert back to Int64
    pq = zeros(n,4)

    # populate the priority queue 
    pq[:,1] = collect(1:n) # add elements
    pq[:,2] = Inf * ones(n) # set all marginal gains to Inf (until queried)
    pq[:,3] = cost # cost
    pq[:,4] = Inf * ones(n) # the value is Inf (until queried)
    return pq
end


function update_pq_with_new_df(pq, df)
    """
    # update_pq_with_new_df
    # This update the value in the priority queue by using a new distortion factor
    #   NOTE: this isn't an optimal update, but it'll work for now.
    #
    # INPUT
    #   pq      the priority queue, (n,5) array
    #   df      new distortion factor
    # 
    # OUTPUT
    #   no output -- priority queue is modified in place
    """
    # update w/ new distortion factor
    pq[:,4] = df * pq[:,2] - pq[:,3] # compute the value

    # sort by this value
    sorted_ind = sortperm(pq[:,4], rev=true)
    pq[:,:] = pq[sorted_ind,:]
    return 
end

function lazy_max_marginal_search(nb_list, covered_v, pq, search_elm; df=1.0)
    """
    # lazy_max_marginal_search
    # This implements the lazy search for an element of maximal marginal gain
    #
    # INPUTS
    #   nb_list     a neighborhood list (dictionary of type Int64 -> Sets(Int64) )
    #   covered_v   a set of vertices which are covered by the current set S
    #   pq          priority queue
    #   search_elm  elements to search over 
    #   df          a distortion factor, only relevant for distorted algorithms
    #
    # OUTPUTS
    #   max_gain    maximal gain
    #   best_e      element with maximium gain in available
    #   best_e_mg   the g marginal gain of the maximal element, i.e. g( best_e | S )
    #   num_evals   the number of times the prioity queue was updated
    #
    """
    # initialize variables to track the lazy evals
    max_gain = -Inf 
    best_e = -1
    best_e_mg = -Inf
    num_evals = 0

    # walk down the priority queue
    for i=1:n

        # we only look at elements e in the search set
        e = convert(Int64, pq[i,1])
        if e in search_elm 

            # pq row is ( e , g_e , c_e , v) 
            # if the stale v_e is worse than the max gain so far, break since v_e will only go down
            v_e = pq[i,4]
            if v_e < max_gain 
                break 
            else # else, update to get current v_e, update the best_e (if necessary), and keep going

                # compute its marginal gain of g and new distorted gain, we call v_e
                g_e = length(setdiff(union(nb_list[e], e), covered_v)) # this is # of nodes adding e would add
                v_e = df * g_e - pq[i,3]

                # update pq and number of evaluations
                pq[i,2] = g_e
                pq[i,4] = v_e 
                num_evals += 1

                # update best seen so far
                if v_e > max_gain 
                    max_gain = v_e 
                    best_e = e 
                    best_e_mg = g_e 
                end
            end
        end
    end
    return max_gain, best_e, best_e_mg, num_evals
end

function greedy(nb_list, cost, k; verbose=false, disp_iter=false)
    """
    # greedy
    # This function is an implementation of the vanilla greedy algorithm.
    #
    # INPUTS
    #   nb_list     a neighborhood list (dictionary of type Int64 -> Sets(Int64) )
    #   cost        a n vector of element costs (positive)
    #   k           cardinality constraints
    #
    # OUTPUTS
    #   gain_array  a length k array of the gains at each iter, i.e. f( e | S ) = g( e | S) + ell( e | S)
    #   elm_array   a length k array of elements added at each iter
    #   evals_array a length k array of number of evals at each iter
    #
    """

    # get dimensions 
    n = length(cost)

    # initialize the solution and record arrays
    sol = Set{Int64}()
    gain_array = zeros(k)
    elm_array = zeros(Int64, k)
    evals_array = zeros(Int64, k)
    available_e = Set(1:n)

    # initialize the number of vertices covered
    covered_v = Set{Int64}()

    # initialize priority queue
    pq = initialize_pq(cost)

    # greedy updates
    for iter=1:k

        # update priority queue with new df (re-sort by value v)
        df = 1.0
        update_pq_with_new_df(pq, df)

        # print priority queue
        if verbose 
            @printf("\tIteration %d current priority queue\n", iter)
            show(stdout, "text/plain", pq)
            println()
        end

        if disp_iter
            @printf("\tIteration %d of %d\n", iter, k)
        end

        # lazy evaluation to find best element
        max_gain, best_e, best_e_mg, num_evals = lazy_max_marginal_search(nb_list, covered_v, pq, available_e)
        evals_array[iter] = num_evals

        # print at each iteration
        if verbose
            @printf("\tIteration %d\tBest Element %d\tGain: %.3f\n", iter, best_e, max_gain)
        end
        
        if max_gain > 0
            # update solution
            union!(sol, best_e)
            setdiff!(available_e, best_e)

            # update the covered vertices
            union!(covered_v, nb_list[best_e])
            union!(covered_v, best_e)

            # update record arrays
            gain_array[iter] = best_e_mg - cost[best_e]
            elm_array[iter] = best_e
        else 
            break # the algorithm can't make progress, so quit
        end
    end
    return gain_array, elm_array, evals_array
end


function distorted_greedy(nb_list, cost, k; verbose=false)
    """
    # distorted_greedy
    # This function is an implementation of the distorted greedy algorithm.
    #
    # INPUTS
    #   nb_list     a neighborhood list (dictionary of type Int64 -> Sets(Int64) )
    #   cost        a n vector of element costs (positive)
    #   k           cardinality constraints
    #
    # OUTPUTS
    #   gain_array  a length k array of the gains at each iter, i.e. f( e | S ) = g( e | S) + ell( e | S)
    #   elm_array   a length k array of elements added at each iter
    #   num_evals      number of evaluations
    #
    """

    # get dimensions 
    n = length(cost)

    # initialize the solution and record arrays
    sol = Set{Int64}()
    gain_array = zeros(k)
    elm_array = zeros(Int64, k)
    num_evals = 0
    available_e = Set(1:n)

    # initialize the number of vertices covered
    covered_v = Set{Int64}()

    # initialize priority queue
    pq = initialize_pq(cost)

    # greedy updates
    for i=0:k-1

        # update priority queue with new df (re-sort by value v)
        df = (1 - (1. / k))^(k - (i+1))
        update_pq_with_new_df(pq, df)

        # print priority queue
        if verbose 
            @printf("\tIteration %d current priority queue\n", iter)
            show(stdout, "text/plain", pq)
            println()
        end

        # lazy evaluation to find best element
        max_gain, best_e, best_e_mg, nevals = lazy_max_marginal_search(nb_list, covered_v, pq, available_e, df=df)
        num_evals += nevals

        # print at each iteration
        if verbose
            @printf("\tIteration %d\tBest Element %d\tGain: %.3f\n", iter, best_e, max_gain)
        end
        
        if max_gain > 0
            # update solution
            union!(sol, best_e)
            setdiff!(available_e, best_e)

            # update the covered vertices
            union!(covered_v, nb_list[best_e])
            union!(covered_v, best_e)

            # update record arrays
            gain_array[i+1] = best_e_mg - cost[best_e]
            elm_array[i+1] = best_e
        end
    end
    return gain_array, elm_array, num_evals
end


function stochastic_distorted_greedy(nb_list, cost, k; eps=0.01)
    """
    # stochastic_distorted_greedy
    # This function is an implementation of the stochastic distorted greedy algorithm.
    # Note: no weakly submodular parameter, as vertex cover is submodular
    #
    # INPUTS
    #   nb_list     a neighborhood list (dictionary of type Int64 -> Sets(Int64) )
    #   cost        a n vector of element costs (positive)
    #   k           cardinality constraint
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

    # initialize the number of vertices covered
    covered_v = Set{Int64}()

    # initialize priority queue
    pq = initialize_pq(cost)

    for i=0:k-1 

        # compute distortion factor, update priority queue with new df
        df = (1 - (1. / k))^(k - (i+1))
        update_pq_with_new_df(pq, df)

        # sample a subset, find maximal element inside
        sample_size = min(length(available_e), sample_size)
        search_elm = sample(collect(available_e), sample_size, replace=false)
        max_gain, best_e, best_e_mg, nevals = lazy_max_marginal_search(nb_list, covered_v, pq, search_elm, df=df)
    
        # update number of function evals
        num_evals += nevals

        # add element if the distorted gain is > 0
        if max_gain > 0

            # update solution
            union!(sol, best_e)
            setdiff!(available_e, best_e)

            # update the covered vertices
            union!(covered_v, nb_list[best_e])
            union!(covered_v, best_e)

            # update record arrays
            gain_array[i+1] = best_e_mg - cost[best_e]
            elm_array[i+1] = best_e
        end
    end
    return gain_array, elm_array, num_evals
end


function unconstrained_distorted_greedy(nb_list, cost)
    """
    # unconstrained_distorted_greedy
    # This function is an implementation of the unconstrained distorted greedy algorithm
    #
    # INPUTS
    #   nb_list     a neighborhood list (dictionary of type Int64 -> Sets(Int64) )
    #   cost        a n vector of element costs (positive)
    #
    # OUTPUTS
    #   gain_array  a length k array of the gains at each iter, i.e. f( e | S ) = g( e | S) + ell( e | S)
    #   elm_array   a length k array of elements added at each iter
    #   num_evals   the number of function evaluations
    #
    """

    # get dimensions
    n = length(cost)

    # initialize set and record arrays
    k = 0
    sol = Set()
    gain_array = zeros(n)
    elm_array = zeros(Int64, n)
    num_evals = 0

    # initialize set of available elements
    available_e = Set(1:n)

    # initialize the number of vertices covered
    covered_v = Set{Int64}()

    for i=0:n-1

        # sample a single element -- forbid choosing e from S, although we allow in analysis
        e = sample(collect(available_e))
        if e in sol # we should never get here - but just in case...
            continue 
        end

        # compute distortion factor
        df = (1 - (1. / n))^(n - (i+1))

        # compute gain
        g_gain = length(setdiff(union(nb_list[e], e), covered_v)) # this is # of nodes adding e would add
        gain = df*g_gain - cost[e]

        # update number of evaluations
        num_evals += 1

        # add element if distorted gain is large enough
        if gain > 0.

            # udpate solution
            union!(sol, e)
            setdiff!(available_e, e)

            # update the covered vertices
            union!(covered_v, nb_list[e])
            union!(covered_v, e)

            # update record arrays
            gain_array[i+1] = g_gain - cost[e]
            elm_array[i+1] = e
        end
    end
    return gain_array, elm_array, num_evals
end

function get_zachary_karate_club()
    """
    # get_zachary_karate_club
    # This function returns the graph of Zachary's karate club.
    # 
    # INPUTS
    #    none
    #
    # OUTPUTS
    #   nb_list     neighborhood list of the graph
    """

    # read in file
    file_name = "data/zachary_karate/edge_list.txt"
    edge_list, header = readdlm(file_name, ',', Int, '\n', header=true)
    n = parse(Int64, header[1])
    m = parse(Int64, header[2])

    # create neighborhood list
    nb_list = Dict( i => Set{Int64}() for i=1:n)
    for i=1:m
        e = edge_list[i,:]
        u = e[1]
        v = e[2]

        # add to nb_list -- assume undirected
        push!(nb_list[u], v)
        push!(nb_list[v], u)
    end 
    return n, nb_list
end

function get_email_eu_core()
    """
    # get_email_eu_core
    # This function returns the graph of the core of the email EU 
    # this also returns a cost, which is the department normalized.
    # 
    # INPUTS
    #    none
    #
    # OUTPUTS
    #   n           number of vertices
    #   nb_list     neighborhood list of the graph
    #   deg         array of degrees (includes self, so a connected graph has min degree >= 2)
    #
    """

    # read in file
    file_name = "data/email_eu_core/email-Eu-core.txt"
    edge_list = readdlm(file_name, ' ', Int, '\n')
    edge_list += ones(Int64, size(edge_list))
    n = maximum(edge_list[:])
    m = size(edge_list)[1]

    # create neighborhood list
    nb_list = Dict( i => Set{Int64}() for i=1:n)
    for i=1:m
        e = edge_list[i,:]
        u = e[1]
        v = e[2]

        # add to nb_list -- this is directed, so u -> v is what we store
        push!(nb_list[u], v)
    end 

    # # create the costs - based on departments
    # file_name = "data/email_eu_core/email-Eu-core-department-labels.txt"
    # department_info = readdlm(file_name, ' ', Int, '\n')
    # cost = department_info[:,2] + ones(size(department_info)[1])
    # cost = cost / maximum(cost)

    # create max a
    deg = [length(union(nb_list[e],e)) for e in 1:n]
    # a_max = maximum(deg ./ cost)

    return n, nb_list, deg
end

function get_email_eu_full()
    """
    # get_email_eu_core
    # This function returns the graph of the core of the email EU 
    # this also returns a cost, which is the department normalized.
    # 
    # INPUTS
    #    none
    #
    # OUTPUTS
    #   n           number of vertices
    #   nb_list     neighborhood list of the graph
    #   deg         array of degrees (includes self, so a connected graph has min degree >= 2)
    #
    """

    # read in file
    file_name = "data/email_eu_full/email-EuAll.txt"
    edge_list = readdlm(file_name, '\t', Int, '\n', comments=true)
    edge_list += ones(Int64, size(edge_list))
    n = maximum(edge_list[:])
    m = size(edge_list)[1]

    # create neighborhood list
    nb_list = Dict( i => Set{Int64}() for i=1:n)
    for i=1:m
        e = edge_list[i,:]
        u = e[1]
        v = e[2]

        # add to nb_list -- this is directed, so u -> v is what we store
        push!(nb_list[u], v)
    end 

    # get max degree
    deg = [length(union(nb_list[e],e)) for e in 1:n]

    return n, nb_list, deg
end

function q_cost(deg, q)
    """
    # q_cost
    # Creates the costs so that
    #   g(e) - c(e) =   q           if q <= g(e) 
    #                   g(e) - 1    otherwise
    # So, setting q = max_degree + 1 yields a cost of all ones
    # setting q = 1 yields that all initial marginal gains are 1
    # Moreover, if a node ever has fewer than q neighbors left
    # that are uncovered, adding the node decreases the total value
    #
    # INPUTS
    #   deg     the array of degrees (count self-edge)
    #   q       a number
    #
    # OUTPUTS
    #   cost    the array of costs
    #
    """
    n = length(deg)
    cost = zeros(n)
    for i=1:n 
        if deg[i] >= q 
            cost[i] = deg[i] - q 
        else 
            cost[i] = 1
        end
    end
    return cost
end
