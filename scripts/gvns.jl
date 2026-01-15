#!/usr/bin/env julia
include("vnd.jl")

function general_variable_neighborhood_search(
    instance::PDPInstance;
    solution::PDPSolutionVector,
    score::Float64,
    fairness::Function=jain_fairness,
    step_func::Function=delta_step_random,
    n_iterations::Int64=200, # outer loop
    verbose::Bool=true,
)
    _, _, best_solution, best_score = variable_neighborhood_descent(
        instance;
        solution=solution,
        score=score,
        fairness=fairness,
        verbose=false
    )

    iter_n = 0 # any loop
    iter_score = Vector{Float64}()
    neighbor_list = [delta_neighbor_between_switch_request]
    curr_iter = 0 # outer loop
    while curr_iter <= n_iterations
        neighbor_index = 1
        while neighbor_index <= length(neighbor_list)
            curr_solution, curr_score = delta_get_neighbor_solution(
                fairness,
                neighbor_list[neighbor_index],
                step_func,
                instance,
                best_solution,
                best_score,
            )
            _, _, curr_solution, curr_score = variable_neighborhood_descent(
                instance;
                solution=curr_solution,
                score=curr_score,
                fairness=fairness,
                verbose=false
            )
            if curr_score < best_score
                best_score = curr_score
                best_solution = curr_solution
                neighbor_index = 1
                curr_iter = 0
            else
                neighbor_index += 1
            end
            push!(iter_score, best_score)
            log_iteration(iter_n, best_score, verbose)
            iter_n += 1
        end
        curr_iter += 1
    end
    log_result(instance, best_solution, best_score, verbose)
    return (iter_score, iter_n, best_solution, best_score)
end

if abspath(PROGRAM_FILE) == @__FILE__ # if main
    _, _, solution, score = solve_PDPInstance(
        ;
        instance_size="50",
        instance_type="train",
        instance_name="instance1_nreq50_nveh2_gamma50.txt",
        algorithm=greedy_heuristic_one_extend_random,
        Dict(:verbose => false)...
    )
    test_PDPInstance(
        ;
        instance_size="50",
        instance_type="train",
        instance_name="instance1_nreq50_nveh2_gamma50.txt",
        algorithm=general_variable_neighborhood_search,
        Dict(:solution => solution, :score => score)...,
    )
end