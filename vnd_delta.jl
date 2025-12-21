#!/usr/bin/env julia
include("solution_iter_delta.jl")
include("greedy_random.jl")

function delta_variable_neighborhood_descent(
    instance::PDPInstance;
    solution::PDPSolutionVector,
    score::Float64,
    verbose::Bool=true,
)
    iter_n = 0
    iter_score = Vector{Float64}()
    best_solution = solution
    best_score = score
    step_func = delta_step_best # step_random, step_first, step_best
    neighbor_list = [delta_neighbor_in_switch_location, delta_neighbor_in_subsequence]
    neighbor_index = 1
    while neighbor_index <= length(neighbor_list)
        curr_solution, curr_score = delta_get_neighbor_solution(instance, best_solution, best_score, neighbor_list[neighbor_index], step_func)
        if curr_score < best_score
            best_score = curr_score
            best_solution = curr_solution
            neighbor_index = 1
        else
            neighbor_index += 1
        end
        push!(iter_score, best_score)
        log_iteration(iter_n, best_score, verbose)
        iter_n += 1
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
        algorithm=delta_variable_neighborhood_descent,
        Dict(:solution => solution, :score => score)...,
    )
end