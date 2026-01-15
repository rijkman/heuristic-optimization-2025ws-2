#!/usr/bin/env julia
include("greedy.jl")

function greedy_heuristic_one_extend_random(
    instance::PDPInstance;
    fairness::Function=jain_fairness,
    lookahead_gamma::Union{Int64,Nothing}=nothing,
    verbose::Bool=true,
)
    solution = [Int64[] for _ in 1:instance.n_vehicles]
    distances = [0 for _ in 1:instance.n_vehicles]
    loads = [0 for _ in 1:instance.n_vehicles]
    open_pickups = Set(collect(1:instance.n_requests))
    open_dropoffs = [Int64[] for _ in 1:instance.n_vehicles]
    served_requests = 0
    rand_alpha = 0.1
    return greedy_heuristic_from_solution_random(
        fairness,
        instance,
        solution,
        distances,
        loads,
        open_pickups,
        open_dropoffs,
        served_requests,
        rand_alpha,
        lookahead_gamma,
        verbose,
    )
end

function greedy_heuristic_from_solution_random(
    fairness::Function,
    instance::PDPInstance,
    best_solution::PDPSolutionVector,
    best_distances::Vector{Int64},
    best_loads::Vector{Int64},
    open_pickups::Set{Int64},
    open_dropoffs::Vector{Vector{Int64}},
    served_requests::Int64,
    rand_alpha::Float64,
    lookahead_gamma::Union{Int64,Nothing},
    verbose::Bool,
)
    iter_n = 0
    iter_score = Vector{Float64}()
    while served_requests < instance.γ
        # loop over all possible solutions for [one-step location extensions]
        candidate_steps = Vector{Tuple{Float64,Int64,Int64,Int64}}()
        for route_k in 1:instance.n_vehicles
            open_locs = vcat(open_pickups..., open_dropoffs[route_k]...)
            for loc_i in open_locs
                # hypothetically assign ith location to kth car
                _, _, is_pickup, load_i = instance.requests[loc_i]
                can_visit = (best_loads[route_k] + load_i) <= instance.capacity || !is_pickup
                if can_visit
                    # check resulting objective score
                    curr_score, curr_distance = delta_objective_value_construct(
                        fairness,
                        instance,
                        best_solution,
                        best_distances,
                        route_k,
                        loc_i,
                    )
                    push!(candidate_steps, (curr_score, curr_distance, route_k, loc_i))
                end
            end
        end
        candidate_scores = first.(candidate_steps)
        cmin, cmax = minimum(candidate_scores), maximum(candidate_scores)
        restrict_threshold = cmin + rand_alpha * (cmax - cmin)
        restrict_steps = filter(step -> step[1] <= restrict_threshold, candidate_steps)
        rand_score, rand_distance, rand_car, rand_loc = rand(restrict_steps)
        # remove pickup or dropoff from remaining requests locations
        served_requests += solution_step_update!(
            instance,
            best_solution,
            best_distances,
            best_loads,
            rand_car,
            rand_loc,
            rand_distance,
            open_pickups,
            open_dropoffs,
        )
        push!(iter_score, rand_score)
        log_iteration(iter_n, rand_score, verbose)
        iter_n += 1
        # quit after gamma steps if called by pilot
        if lookahead_gamma != nothing && iter_n == (lookahead_gamma - 1)
            break
        end
    end
    best_solution = solution_clean!(instance, best_solution)
    best_score = objective_value(fairness, instance, best_solution)
    log_result(instance, best_solution, best_score, verbose)
    return (iter_score, iter_n, best_solution, best_score)
end

if abspath(PROGRAM_FILE) == @__FILE__ # if main
    Random.seed!(1)
    test_PDPInstance(
        ;
        instance_size="50",
        instance_type="train",
        instance_name="instance1_nreq50_nveh2_gamma50.txt",
        algorithm=greedy_heuristic_one_extend_random,
    )
end
