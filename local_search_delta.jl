#!/usr/bin/env julia
include("solution_iter_delta.jl")
include("greedy_random.jl")
function delta_local_search(
    instance::PDPInstance;
    solution::PDPSolutionVector,
    score::Float64,
    neighborhood_func::Function,
    step_func::Function,
    stop_time::Float64=60.0,
    stop_percentage::Float64=70.0,
    stop_steps::Int64=200,
    verbose::Bool=true,
)
    iter_n = 0
    iter_score = Vector{Float64}()
    start_time::Float64 = time()
    no_improvement_steps::Int64 = 0
    best_solution = solution
    best_score = score
    while (time() - start_time < stop_time) && ((best_score / score) * 100 > stop_percentage) && (no_improvement_steps < stop_steps)
        curr_solution, curr_score = delta_get_neighbor_solution(instance, best_solution, best_score, neighborhood_func, step_func)
        if curr_score < best_score
            best_score = curr_score
            best_solution = curr_solution
            no_improvement_steps = 0
        else
            no_improvement_steps += 1
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
        algorithm=delta_local_search,
        Dict(:solution => solution,
            :score => score,
            :neighborhood_func => delta_neighbor_in_switch_location,
            :step_func => delta_step_best)...,
    )
end