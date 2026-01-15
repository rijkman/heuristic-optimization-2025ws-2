#!/usr/bin/env julia
include("greedy_random.jl")

function delta_local_search(
    instance::PDPInstance;
    solution::PDPSolutionVector,
    score::Float64,
    fairness::Function=jain_fairness,
    neighborhood_func::Function=delta_neighbor_in_switch_location,
    step_func::Function=delta_step_best,
    n_iterations::Int64=200,
    iter_n::Int64=0, # any loop == outer loop
    verbose::Bool=true,
)
    iter_score = Vector{Float64}()
    no_improvement_steps::Int64 = 0
    best_solution = solution
    best_score = score
    n_iterations += iter_n
    while iter_n < n_iterations
        curr_solution, curr_score = delta_get_neighbor_solution(
            fairness,
            neighborhood_func,
            step_func,
            instance,
            best_solution,
            best_score
        )
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
        algorithm=delta_local_search,
        Dict(:solution => solution, :score => score)...,
    )
end