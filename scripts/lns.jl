#!/usr/bin/env julia
include("greedy_insertion.jl")
include("local_search.jl")

function large_neighborhood_search(
    instance::PDPInstance;
    solution::PDPSolutionVector,
    score::Float64,
    fairness::Function=jain_fairness,
    # runtime parameters
    n_iterations::Int64=200,
    m_repairs::Int64=50,
    # destroy percentage
    alpha::Float64=0.01,
    verbose::Bool=true,
)
    iter_n = 0 # any loop
    iter_score = Vector{Float64}()
    best_solution = solution
    best_score = score
    curr_iter = 0 # outer loop
    while curr_iter < n_iterations
        # destroy: remove alpha% request pairs
        k = floor(alpha * sum([length(x) for x in best_solution]))
        curr_solution = deepcopy(best_solution)
        for _ in 1:k
            # random selection of removed requests
            v_index = rand([k for k in 1:instance.n_vehicles if length(curr_solution[k]) > 0])
            r_index = rand(1:length(curr_solution[v_index]))
            d_req = curr_solution[v_index][r_index]
            pickup_idx, dropoff_idx, _, _ = instance.requests[d_req]
            if !(pickup_idx in curr_solution[v_index] && dropoff_idx in curr_solution[v_index])
                continue
            end
            deleteat!(curr_solution[v_index], findfirst(==(pickup_idx), curr_solution[v_index]))
            deleteat!(curr_solution[v_index], findfirst(==(dropoff_idx), curr_solution[v_index]))
        end

        # repair: greedy insertion heuristic + local search
        _, _, curr_solution, curr_score = greedy_insertion(  # greedy_heuristic_from_partial
            fairness,
            instance,
            curr_solution,
            false
        )
        sub_iter_score, sub_iter_n, curr_solution, curr_score = delta_local_search(
            instance;
            solution=curr_solution,
            score=curr_score,
            fairness=fairness,
            neighborhood_func=delta_neighbor_in_switch_location,
            step_func=delta_step_best,
            n_iterations=m_repairs,
            iter_n=iter_n,
            verbose=false,
        )

        match_iter_score = accumulate(min, sub_iter_score; init=best_score)
        if curr_score < best_score
            best_score = curr_score
            best_solution = curr_solution
        end
        append!(iter_score, match_iter_score)
        for it_idx in iter_n:sub_iter_n-1
            log_iteration(it_idx, iter_score[it_idx+1], verbose)
        end
        iter_n = sub_iter_n
        curr_iter += 1
    end
    best_score = objective_value(fairness, instance, best_solution)
    return (iter_score, iter_n, best_solution, best_score)
end

if abspath(PROGRAM_FILE) == @__FILE__ # if main
    _, _, solution, score = solve_PDPInstance(
        ;
        instance_size="50",
        instance_type="train",
        instance_name="instance1_nreq50_nveh2_gamma50.txt",
        algorithm=greedy_heuristic_one_extend,
        Dict(:verbose => false)...
    )
    test_PDPInstance(
        ;
        instance_size="50",
        instance_type="train",
        instance_name="instance1_nreq50_nveh2_gamma50.txt",
        algorithm=large_neighborhood_search,
        Dict(:solution => solution, :score => score)...,
    )
end
