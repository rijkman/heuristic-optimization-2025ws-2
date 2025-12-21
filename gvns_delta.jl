#!/usr/bin/env julia
include("vnd_delta.jl")

function delta_general_variable_neighborhood_search(
    instance::PDPInstance;
    solution::PDPSolutionVector,
    score::Float64,
    verbose::Bool=true,
)
    iter_n = 0
    iter_score = Vector{Float64}()
    _, _, best_solution, best_score = delta_variable_neighborhood_descent(instance; solution=solution, score=score)
    step_func = delta_step_random
    neighbor_list = [delta_neighbor_between_switch_request]
    max_tries = 100
    curr_tries = 0
    while curr_tries <= max_tries
        neighbor_index = 1
        while neighbor_index <= length(neighbor_list)
            curr_solution, curr_score = delta_get_neighbor_solution(instance, best_solution, best_score, neighbor_list[neighbor_index], step_func)
            _, _, curr_solution, curr_score = delta_variable_neighborhood_descent(instance; solution=curr_solution, score=curr_score, verbose=false)
            if curr_score < best_score
                best_score = curr_score
                best_solution = curr_solution
                neighbor_index = 1
                curr_tries = 0
            else
                neighbor_index += 1
            end
            push!(iter_score, best_score)
            log_iteration(iter_n, best_score, verbose)
            iter_n += 1
        end
        curr_tries += 1
    end
    log_result(instance, best_solution, best_score, verbose)
    return (iter_score, iter_n, best_solution, best_score)
end

if abspath(PROGRAM_FILE) == @__FILE__ # if main
    _, _, solution, score = solve_PDPInstance(
        ;
        path_dir="instances",
        instance_size="50",
        instance_type="train",
        instance_name="instance1_nreq50_nveh2_gamma50.txt",
        algorithm=greedy_heuristic_one_extend_random,
        Dict(:verbose => false)...
    )
    test_PDPInstance(
        ;
        path_dir="instances",
        instance_size="50",
        instance_type="train",
        instance_name="instance1_nreq50_nveh2_gamma50.txt",
        algorithm=delta_general_variable_neighborhood_search,
        Dict(:solution => solution, :score => score)...,
    )
end